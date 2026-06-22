from __future__ import annotations

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.utils import convert_to_bigram_key

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the KV cache.
"""

import heapq
import hashlib
import logging
import math
import os
import re
import sys
import threading
import time
from collections import defaultdict
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple, Union

import torch

from sglang.srt.mem_cache.memory_pool import move_kv_cache_native

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Anchor KV Entry for position-aligned non-prefix reuse.
#
# AgentTemplateKV treats the KVCOMM-style exact-content/RoPE copy as the low
# level mechanism, then adds device-resident future-use protection so coding
# agent templates can turn prefetch hints into real GPU hits.
# ---------------------------------------------------------------------------
class AnchorKVEntry:
    """Stores KV cache for an anchor block, keyed by signature + position."""

    def __init__(
        self,
        signature: str,
        token_ids: torch.Tensor,
        kv_indices: torch.Tensor,
        start_pos: int,
        code_content_signature: str = "",
        source_node: Optional["TreeNode"] = None,
        prefix_context_signature: str = "",
        # Per-placeholder anchor pool fields (Duke 2026 KVCOMM-style).
        # `slot_id` is the placeholder taxonomy key (e.g. "plan",
        # "architecture", "extra_context"); pool_embedding is the single
        # L2-normalized MiniLM embedding of the slot's full text used for
        # k-NN lookup.  last_access_time is the LRU key (kept even on
        # code-anchor entries because the same dataclass is shared).
        slot_id: str = "",
        slot_label: str = "",
        pool_embedding: Optional[torch.Tensor] = None,
        embedding_text: str = "",
        last_access_time: float = 0.0,
    ):
        self.signature = signature
        self.code_content_signature = code_content_signature
        self.token_ids = token_ids
        self.kv_indices = kv_indices
        self.start_pos = start_pos
        self.prefix_context_signature = prefix_context_signature
        self.ref_count = 1
        self.source_node = source_node
        self.prefetch_protected_until = 0.0
        self.prefetch_steps_remaining = 0
        self.prefetch_lock_held = False
        self.prefetch_hit_count = 0
        # Optional chunk embeddings for semantic suffix-copy length decider.
        # Shape [N_chunks, D] when populated; None when semantic suffix is
        # disabled, embedder failed to load, or entry too short to chunk.
        self.chunk_embeddings = None
        # Per-placeholder k-NN reuse fields.  See comment above.
        self.slot_id = slot_id
        self.slot_label = slot_label
        self.pool_embedding = pool_embedding
        self.embedding_text = embedding_text
        self.last_access_time = last_access_time

    def __repr__(self):
        return (
            f"AnchorKVEntry(sig={self.signature!r:.30}, "
            f"start_pos={self.start_pos}, len={len(self.token_ids)}, "
            f"slot={self.slot_id!r}, ref={self.ref_count})"
        )


def _token_prefix_signature(token_ids: list[int] | torch.Tensor, end_pos: int) -> str:
    if end_pos <= 0:
        return "sha1:"
    if isinstance(token_ids, torch.Tensor):
        values = token_ids[:end_pos].detach().cpu().tolist()
    else:
        values = token_ids[:end_pos]
    payload = ",".join(str(int(x)) for x in values)
    return "sha1:" + hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _placeholder_knn_search(
    pool_entries: List[AnchorKVEntry],
    query_embedding: torch.Tensor,
    top_k: int = 4,
    min_similarity: float = 0.70,
) -> List[Tuple[AnchorKVEntry, float]]:
    """Per-placeholder embedding k-NN search (Duke 2026 KVCOMM-style).

    Args:
        pool_entries: list of AnchorKVEntry with non-None `pool_embedding`.
        query_embedding: 1-D L2-normalized tensor of shape [D].
        top_k: at most this many neighbors returned.
        min_similarity: floor on cosine similarity (results below are dropped).

    Returns:
        List of (entry, cosine) tuples, sorted descending by cosine.
        Empty list when pool is empty or no entry passes the floor.
    """
    if not pool_entries:
        return []
    # Filter out entries with no embedding (e.g. embedder failed to load at
    # store time).
    valid = [e for e in pool_entries if e.pool_embedding is not None]
    if not valid:
        return []
    embeddings = torch.stack([e.pool_embedding for e in valid]).to(
        dtype=query_embedding.dtype,
    )
    # L2-normalize defensively in case any entry wasn't normalized at store
    # time (e.g. model moved to GPU between calls).
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    q = torch.nn.functional.normalize(
        query_embedding.view(1, -1).to(embeddings.device), p=2, dim=1,
    )
    sims = (embeddings @ q.T).squeeze(1)  # [N]
    k = min(top_k, sims.numel())
    top_sims, top_idx = torch.topk(sims, k=k)
    out: List[Tuple[AnchorKVEntry, float]] = []
    for sim, idx in zip(top_sims.tolist(), top_idx.tolist()):
        if sim < min_similarity:
            break  # sorted descending; remaining entries also below floor
        out.append((valid[idx], float(sim)))
    return out

from sglang.srt.disaggregation.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
)
from sglang.srt.mem_cache.anchor_match import (
    AnchorMetadata,
    AnchorMatchResult,
    build_anchor_metadata,
    select_best_match,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.evict_policy import (
    EvictionStrategy,
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
    SLRUStrategy,
    TieredPriorityStrategy,
)
from sglang.srt.mem_cache.hicache_storage import get_hash_str, hash_str_to_int64

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class RadixKey:
    def __init__(
        self,
        token_ids: List[int],
        extra_key: Optional[str] = None,
        is_bigram: bool = False,
    ):
        # token ids sequence
        self.token_ids = token_ids
        # extra key (e.g. lora_id, cache_salt)
        self.extra_key = extra_key
        # is bigram key
        self.is_bigram = is_bigram

    def __len__(self) -> int:
        return len(self.token_ids)

    def __iter__(self) -> Iterator[int]:
        return iter(self.token_ids)

    def __getitem__(self, idx: Union[int, slice]) -> "RadixKey":
        if isinstance(idx, slice):
            return RadixKey(self.token_ids[idx], self.extra_key)
        return RadixKey([self.token_ids[idx]], self.extra_key)

    def __repr__(self) -> str:
        preview = self.token_ids[:10]
        return f"RadixKey(extra_key={self.extra_key!r}, token_ids={preview}{'...' if len(self.token_ids) > 10 else ''})"


# ---------------------------------------------------------------------------
# Role type constants for token-type-aware PriorityStrategy
# These are set on TreeNode.role_type during insertion to identify which
# tier of the multi-agent prefix the node belongs to.
# ---------------------------------------------------------------------------
ROLE_TYPE_SYSTEM = 1   # Tier-0: universal system prompt (shared by ALL agents)
ROLE_TYPE_ROLE   = 2   # Tier-1: role-specific imports/signatures (shared by role)
ROLE_TYPE_TASK   = 3   # Tier-2: workflow-specific task context (unique to workflow)
# 0 = Tier-3 / dynamic suffix or unknown


def infer_role_type_from_key(key: RadixKey) -> int:
    """Infer the role type from the first few tokens of a key.

    Uses simple heuristic based on common token sequences:
      - System prompts often start with "You are" or "You have"
      - Python import lines start with "import " or "from "
    This is best-effort; for precise marking, pass role_type explicitly during insert.
    """
    if not key or len(key) < 3:
        return 0
    try:
        ids = key.token_ids[:8]
        # These are approximate first-token IDs for common English words.
        # In practice, explicit role_type via insert params is more reliable.
        # We use a conservative heuristic: look for patterns that indicate
        # the beginning of a new section.
        # Since exact token IDs are tokenizer-dependent, we return 0 (unknown)
        # and rely on the explicit role_type in insert params.
        return 0
    except Exception:
        return 0


def maybe_bigram_convert(
    is_eagle: bool,
    key: RadixKey,
    value: Optional[torch.Tensor] = None,
) -> Tuple[RadixKey, Optional[torch.Tensor]]:
    if is_eagle and not key.is_bigram:
        key.token_ids = convert_to_bigram_key(key.token_ids)
        key.is_bigram = True
        if value is not None:
            value = value[: len(key)]
    return key, value


def page_align_keys(key: list, page_size) -> list:
    if page_size == 1:
        return key
    page_aligned_len = len(key) // page_size * page_size
    return key[:page_aligned_len]


class TreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.lock_ref = 0
        self.pin_expiry: float = (
            0.0  # absolute expiry time (time.monotonic()), 0 = not pinned
        )
        self.pin_ttl: int = 0  # original TTL in seconds, for refresh-on-hit
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()

        self.hit_count = 0
        # indicating the node is locked to protect from eviction
        # incremented when the node is referenced by a storage operation
        self.host_ref_counter = 0
        # store the host indices of KV cache
        self.host_value: Optional[torch.Tensor] = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None
        # priority for priority-aware eviction
        self.priority = priority
        # Cross-workflow prefix sharing: set of workflow IDs that have accessed
        # this node (including both direct insert and prefix match). Nodes used
        # by multiple workflows are more valuable to keep cached.
        self.workflow_refs: set = set()
        # Token type awareness: "system" for Tier-0 universal prefixes,
        # "role" for Tier-1 role-based prefixes (imports/signatures), 0 for others.
        # PriorityStrategy uses this to give system and role prefixes extra retention boost.
        self.role_type: int = 0
        # DAG-aware convergence protection: number of downstream nodes that depend on
        # this prefix. Higher value = more nodes depend on it = should be evicted later.
        # PriorityStrategy uses this to protect convergence nodes in DAG workflows.
        self.convergence_factor: int = 0
        # DAG-aware critical path distance: distance from this node to the leaf node.
        # Higher distance = further from leaf = needed later = protect.
        # PLANNER: distance=3, ARCHITECT/REVIEWER: distance=2, IMPLEMENTER/TESTER: distance=1.
        # PriorityStrategy uses this to protect the critical path in DAG workflows.
        self.critical_path_distance: int = 1
        self.anchor_type: str = ""
        self.anchor_id: str = ""
        self.code_content_signature: str = ""
        self.anchor_spans: list[dict[str, Any]] = []
        self.reuse_mode: str = ""
        self.reuse_confidence: float = 0.0
        self.syntax_region_type: str = ""
        self.template_task_family: str = ""
        self.template_workflow_signature: str = ""
        self.template_structural_fingerprint: str = ""
        # Prompt-context fields (sglang-kvflow context_aware_confidence)
        self.nesting_depth: int = 0
        self.prompt_position_offset: int = 0
        self.system_prompt_class: str = ""
        self.surrounding_code_hash: str = ""

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def protect_host(self):
        """Protect the host value from eviction."""
        self.host_ref_counter += 1

    def release_host(self):
        """Release the host value, allowing it to be evicted."""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    @lru_cache(maxsize=1)
    def get_prefix_hash_values(self, node: TreeNode) -> List[str]:
        if node is None or node.hash_value is None:
            return []

        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def _check_extra_key(key0: RadixKey, key1: RadixKey):
    if key0.extra_key != key1.extra_key:
        raise ValueError(
            f"_key_match should be run on the same extra key, but got key0.extra_key={key0.extra_key} != key1.extra_key={key1.extra_key}"
        )


def _key_match_page_size1(key0: RadixKey, key1: RadixKey):
    _check_extra_key(key0, key1)
    i = 0
    for k0, k1 in zip(key0.token_ids, key1.token_ids):
        if k0 != k1:
            break
        i += 1
    return i


def _key_match_paged(key0: RadixKey, key1: RadixKey, page_size: int):
    _check_extra_key(key0, key1)
    min_len = min(len(key0), len(key1))

    i = 0
    while i < min_len:
        if key0.token_ids[i : i + page_size] != key1.token_ids[i : i + page_size]:
            break
        i += page_size

    return i


def get_child_key(key: RadixKey, page_size: int = 1):
    if page_size == 1:
        plain_key = key.token_ids[0]
    else:
        plain_key = tuple(key.token_ids[:page_size])
    if key.extra_key is None:
        return plain_key
    else:
        return (key.extra_key, plain_key)


def compute_node_hash_values(node: "TreeNode", page_size: int) -> List[str]:
    """Compute SHA256-based hash values for position-aware identification.

    Args:
        node: The TreeNode to compute hash values for
        page_size: The page size for chunking tokens

    Returns:
        List of SHA256 hex strings, one per page
    """
    hash_values = []

    # Get parent's last hash value if parent exists
    parent_hash = None
    if node.parent is not None and node.parent.hash_value is not None:
        # Check if parent is root by checking if it has empty key
        if len(node.parent.key) > 0 and len(node.parent.hash_value) > 0:
            parent_hash = node.parent.hash_value[-1]

    # Iterate through node's pages
    for start in range(0, len(node.key), page_size):
        page_tokens = node.key.token_ids[start : start + page_size]
        if not page_tokens:
            continue

        # Use SHA256-based chaining via get_hash_str
        hash_val = get_hash_str(page_tokens, prior_hash=parent_hash)
        hash_values.append(hash_val)
        parent_hash = hash_val

    return hash_values


def split_node_hash_value(
    child_hash_value: Optional[List[str]], split_len: int, page_size: int
) -> tuple[Optional[List[str]], Optional[List[str]]]:
    """Split hash_value between parent and child nodes during node splitting.

    Args:
        child_hash_value: The hash_value list from the child node being split
        split_len: The length at which to split (in tokens)
        page_size: The page size for calculating number of pages

    Returns:
        Tuple of (new_node_hash_value, updated_child_hash_value)
    """
    if child_hash_value is None:
        return None, None

    if page_size == 1:
        split_pages = split_len
    else:
        split_pages = split_len // page_size

    new_node_hash = child_hash_value[:split_pages]
    child_hash = child_hash_value[split_pages:]

    return new_node_hash, child_hash


class RadixCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.is_eagle = params.is_eagle
        self.disable_finished_insert = params.disable_finished_insert
        self.eviction_policy = params.eviction_policy.lower()

        self.kv_event_queue = []

        if params.enable_metrics:
            self.init_metrics_collector()

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = get_child_key
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=self.page_size)
            self.get_child_key_fn = partial(get_child_key, page_size=self.page_size)

        if self.eviction_policy == "lru":
            self.eviction_strategy: EvictionStrategy = LRUStrategy()
        elif self.eviction_policy == "lfu":
            self.eviction_strategy: EvictionStrategy = LFUStrategy()
        elif self.eviction_policy == "fifo":
            self.eviction_strategy: EvictionStrategy = FIFOStrategy()
        elif self.eviction_policy == "mru":
            self.eviction_strategy: EvictionStrategy = MRUStrategy()
        elif self.eviction_policy == "filo":
            self.eviction_strategy: EvictionStrategy = FILOStrategy()
        elif self.eviction_policy == "priority":
            self.eviction_strategy: EvictionStrategy = PriorityStrategy()
        elif self.eviction_policy == "slru":
            self.eviction_strategy: EvictionStrategy = SLRUStrategy()
        elif self.eviction_policy == "tiered":
            self.eviction_strategy: EvictionStrategy = TieredPriorityStrategy()

        else:
            raise ValueError(
                f"Unknown eviction policy: {self.eviction_policy}. Supported policies: 'lru', 'lfu', 'fifo', 'mru', 'filo', 'priority', 'slru', 'tiered'."
            )

        self.evictable_leaves = set()
        self.anchor_kv_store: dict[str, list[AnchorKVEntry]] = {}
        self.anchor_kv_store_lock = threading.RLock()
        # Per-placeholder anchor pool (Duke 2026 KVCOMM-style).  Keyed by
        # `slot_id` (e.g. "plan", "architecture", "extra_context"); value is
        # a list of AnchorKVEntry whose pool_embedding is used for k-NN
        # search at request-prefill time.  Independent lock because the
        # access pattern is disjoint from the byte-exact code-anchor pool.
        self.placeholder_anchor_pool: dict[str, list[AnchorKVEntry]] = {}
        self.placeholder_anchor_pool_lock = threading.RLock()
        self.placeholder_pool_max_per_slot: int = int(
            os.environ.get("SGLANG_PLACEHOLDER_POOL_MAX_PER_SLOT", "256")
        )

        self.rope_base = params.rope_base
        self.rope_rotary_dim = params.rope_rotary_dim
        self.rope_is_neox_style = params.rope_is_neox_style
        self.rope_num_kv_heads = params.rope_num_kv_heads
        self.rope_inv_freq: Optional[torch.Tensor] = None

        self.reset()

    @classmethod
    def create_simulated(
        self,
        disable: bool = False,
        mock_allocator: Optional[Any] = None,
        page_size: int = 1,
        enable_kv_cache_events: bool = False,
    ) -> RadixCache:
        """Init a radix cache without memory pools for simulation purpose."""
        params = CacheInitParams(
            disable=disable,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=page_size,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        return RadixCache(params)

    ##### Public API #####

    def reset(self):
        # Initialize root with minimum priority so any real priority overrides it
        self.root_node = TreeNode(priority=-sys.maxsize)
        self.root_node.key = RadixKey(token_ids=[], extra_key=None)
        self.root_node.value = []
        self.root_node.host_value = []
        self.root_node.lock_ref = 1
        self.root_node.hash_value = []
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.evictable_leaves.clear()
        with self.anchor_kv_store_lock:
            self.anchor_kv_store.clear()
        self._record_all_cleared_event()

    def maybe_bigram_convert(
        self, key: RadixKey, value: Optional[torch.Tensor] = None
    ) -> Tuple[RadixKey, Optional[torch.Tensor]]:
        return maybe_bigram_convert(self.is_eagle, key, value)

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find the longest cached prefix of ``key`` in the radix tree.

        The logical namespace for prefix matching is determined by both the
        token id sequence and the optional ``extra_key`` carried by ``RadixKey``.
        Entries that share identical leading token ids but have *different*
        ``extra_key`` values are intentionally kept disjoint and never share
        prefix nodes. This is useful to:

        * Isolate KV cache lines for different LoRA / adapter IDs.
        * Separate requests that intentionally should not share state (e.g.,
          different sampling salt, cache version, or retrieval augmentation
          context) by supplying a distinct ``extra_key``.

        Args:
            params (MatchPrefixParams): Parameters containing the lookup key
                with a list of token ids and an optional ``extra_key`` namespace tag.
                If ``page_size > 1`` the length is internally truncated to a multiple
                of ``page_size`` before matching. Passing an empty key returns an
                empty result with the root as the last node.

        Returns:
            MatchResult: ``device_indices`` is a 1-D ``torch.int64`` tensor of
            the concatenated KV cache indices corresponding to the longest
            cached prefix (may be length 0). ``last_device_node`` and
            ``last_host_node`` (currently the same) are the tree node objects
            representing the terminal node of the matched prefix. This method
            may mutate internal structure by splitting an existing node if the
            match ends inside a stored segment.

        Internal updates:
            * Refreshes access metadata (timestamps) used by the
                configured eviction strategy.
            * If the lookup ends inside a stored segment the node is split once
                to expose a precise boundary; this structural refinement improves
                subsequent match efficiency and does not duplicate data.
        """
        key = params.key
        key, _ = self.maybe_bigram_convert(key)
        req = params.req

        def empty_match_result():
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
            )

        best_node = None
        if req is not None and (getattr(req, "reuse_mode", "") or "") == "lossy":
            best_node, _ = self._resolve_lossy_match(req)
            if not (getattr(req, "lossy_first_reuse_allowed", True)):
                return empty_match_result()

        if self.disable or len(key) == 0:
            return empty_match_result()

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        if len(key) == 0:
            return empty_match_result()

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if best_node is not None:
            value, last_node = self._try_lossy_fuzzy_match(
                req, key, value, last_node, best_node
            )
        # PR3 placeholder k-NN: per-slot embedding k-NN reuse (Duke 2026
        # KVCOMM-style). Runs after byte-exact match; gated by env
        # SGLANG_PLACEHOLDER_KNN_MATCH=1.  Composes with `_try_lossy_fuzzy_
        # match` — both can fire on the same request.
        if req is not None and (
            getattr(req, "placeholder_anchor_token_spans", None) or []
        ):
            value, last_node = self._try_placeholder_knn_lossy_match(
                req, key, value, last_node,
            )
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)
        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
        )

    def insert(
        self,
        params: InsertParams,
        workflow_id: Optional[int] = None,
        role_type: int = 0,
    ) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        priority = params.priority
        chunked = params.chunked

        # workflow_id can come from either the call site or params
        effective_wid = workflow_id if workflow_id is not None else params.workflow_id
        # role_type: which tier of the multi-agent prefix this belongs to
        effective_role_type = role_type or getattr(params, "role_type", 0)
        # convergence_factor: DAG-aware protection for convergence nodes
        effective_convergence_factor = getattr(params, "convergence_factor", 0) or 0
        # critical_path_distance: DAG-aware critical path distance
        effective_crit_distance = getattr(params, "critical_path_distance", 1) or 1
        effective_anchor_type = getattr(params, "anchor_type", "") or ""
        effective_anchor_id = getattr(params, "anchor_id", "") or ""
        effective_code_content_signature = getattr(params, "code_content_signature", "") or ""
        effective_anchor_spans = getattr(params, "anchor_spans", None) or []
        effective_reuse_mode = getattr(params, "reuse_mode", "") or ""
        effective_reuse_confidence = getattr(params, "reuse_confidence", 0.0) or 0.0
        effective_syntax_region_type = getattr(params, "syntax_region_type", "") or ""
        effective_template_task_family = getattr(params, "template_task_family", "") or ""
        effective_template_workflow_signature = getattr(params, "template_workflow_signature", "") or ""
        effective_template_structural_fingerprint = getattr(params, "template_structural_fingerprint", "") or ""
        # Prompt-context fields (sglang-kvflow context_aware_confidence)
        effective_nesting_depth = getattr(params, "nesting_depth", 0) or 0
        effective_prompt_position_offset = getattr(params, "prompt_position_offset", 0) or 0
        effective_system_prompt_class = getattr(params, "system_prompt_class", "") or ""
        effective_surrounding_code_hash = getattr(params, "surrounding_code_hash", "") or ""

        if value is None:
            value = torch.tensor(key.token_ids, dtype=torch.int64)

        key, value = self.maybe_bigram_convert(key, value)

        prefix_len = self._insert_helper(
            self.root_node, key, value, priority, chunked,
            workflow_id=effective_wid,
            role_type=effective_role_type,
            convergence_factor=effective_convergence_factor,
            critical_path_distance=effective_crit_distance,
            anchor_type=effective_anchor_type,
            anchor_id=effective_anchor_id,
            code_content_signature=effective_code_content_signature,
            anchor_spans=effective_anchor_spans,
            reuse_mode=effective_reuse_mode,
            reuse_confidence=effective_reuse_confidence,
            syntax_region_type=effective_syntax_region_type,
            template_task_family=effective_template_task_family,
            template_workflow_signature=effective_template_workflow_signature,
            template_structural_fingerprint=effective_template_structural_fingerprint,
            nesting_depth=effective_nesting_depth,
            prompt_position_offset=effective_prompt_position_offset,
            system_prompt_class=effective_system_prompt_class,
            surrounding_code_hash=effective_surrounding_code_hash,
        )
        return InsertResult(prefix_len=prefix_len)

    def _build_req_anchor_metadata(self, req: Req) -> AnchorMetadata:
        return build_anchor_metadata(
            code_anchor_signature=getattr(req, "code_anchor_signature", "") or "",
            code_content_signature=getattr(req, "code_content_signature", "") or "",
            code_anchor_spans=getattr(req, "code_anchor_spans", None) or [],
            reuse_mode=getattr(req, "reuse_mode", "") or "",
            lossy_alignment_method=getattr(req, "lossy_alignment_method", "") or "",
            template_task_family=getattr(req, "template_task_family", "") or "",
            template_workflow_signature=getattr(req, "template_workflow_signature", "") or "",
            template_structural_fingerprint=getattr(req, "template_structural_fingerprint", "") or "",
            nesting_depth=getattr(req, "nesting_depth", 0) or 0,
            prompt_position_offset=getattr(req, "prompt_position_offset", 0) or 0,
            system_prompt_class=getattr(req, "system_prompt_class", "") or "",
            surrounding_code_hash=getattr(req, "surrounding_code_hash", "") or "",
        )

    def _build_node_anchor_metadata(self, node: TreeNode) -> AnchorMetadata:
        return build_anchor_metadata(
            code_anchor_signature=node.anchor_id,
            code_content_signature=node.code_content_signature,
            code_anchor_spans=node.anchor_spans,
            reuse_mode=node.reuse_mode,
            lossy_alignment_method=node.anchor_type,
            template_task_family=node.template_task_family,
            template_workflow_signature=node.template_workflow_signature,
            template_structural_fingerprint=node.template_structural_fingerprint,
            nesting_depth=node.nesting_depth,
            prompt_position_offset=node.prompt_position_offset,
            system_prompt_class=node.system_prompt_class,
            surrounding_code_hash=node.surrounding_code_hash,
        )

    def _iter_nodes_with_anchor_metadata(self) -> Iterator[TreeNode]:
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            if node.anchor_id or node.anchor_spans:
                yield node
            stack.extend(node.children.values())

    def _merge_anchor_spans(self, node: TreeNode, anchor_spans: list[dict[str, Any]] | None):
        """Attach new span-level signatures without overwriting existing metadata.

        A whole-file request and a selective-span request can share the same
        radix node because the model-visible prompt is identical. The first
        insert may therefore attach whole-file metadata, while a later insert
        carries function/method spans for the same prompt. Keep both sets so
        exact-content matching can find per-span signatures.
        """
        if not anchor_spans:
            return
        if not node.anchor_spans:
            node.anchor_spans = list(anchor_spans)
            return
        seen = {
            (
                str(span.get("anchor_type", "") or ""),
                str(span.get("signature", "") or ""),
                str(span.get("content_signature", "") or ""),
                int(span.get("start_line", 0) or 0),
                int(span.get("end_line", 0) or 0),
                str(span.get("segment_name", "") or ""),
            )
            for span in node.anchor_spans
            if isinstance(span, dict)
        }
        for span in anchor_spans:
            if not isinstance(span, dict):
                continue
            key = (
                str(span.get("anchor_type", "") or ""),
                str(span.get("signature", "") or ""),
                str(span.get("content_signature", "") or ""),
                int(span.get("start_line", 0) or 0),
                int(span.get("end_line", 0) or 0),
                str(span.get("segment_name", "") or ""),
            )
            if key not in seen:
                node.anchor_spans.append(dict(span))
                seen.add(key)

    def _default_syntax_region_type(self, req: Req) -> str:
        spans = getattr(req, "code_anchor_spans", None) or []
        if spans and isinstance(spans[0], dict):
            anchor_type = str(spans[0].get("anchor_type", "") or "")
            if anchor_type:
                return anchor_type
        return "code_anchor"

    def _extract_prefetch_hint_signatures(self, req: Req) -> set[str]:
        signatures: set[str] = set()
        for hint in getattr(req, "codebase_prefetch_hints", None) or []:
            if not isinstance(hint, dict):
                continue
            signature = str(
                hint.get("content_signature")
                or hint.get("code_content_signature")
                or ""
            )
            if signature:
                signatures.add(signature)
        return signatures

    def _agenttemplatekv_enabled(self) -> bool:
        """Master gate: return False if AgentTemplateKV paths are disabled.

        When ``SGLANG_LOSSY_ENABLED=0``, all ``_agenttemplatekv_*``
        and ``agenttemplatekv_prefetch_codebases`` methods are no-ops. This
        lets non-AgentTemplateKV users turn the feature off without changing
        any other behavior. The default is ``1`` (enabled).
        """
        try:
            return os.environ.get("SGLANG_LOSSY_ENABLED", "1") != "0"
        except Exception:
            return True

    def _agenttemplatekv_protect_entry(
        self,
        entry: AnchorKVEntry,
        *,
        req: Optional[Req] = None,
        steps_to_use: int = 1,
        ttl_s: Optional[float] = None,
        max_ancestors: int = 2,
    ) -> bool:
        if not self._agenttemplatekv_enabled():
            return False
        """Keep an exact-content anchor resident for the next coding agent.

        Safety-net cap: if locking the entry would push ``protected_size_``
        above ``_agenttemplatekv_protected_size_cap()``, the protect is
        rejected (treated as a miss). This bounds cumulative state across
        cases so the KV pool never gets starved.

        Capped walk: instead of locking every ancestor up to root, only
        ``max_ancestors`` (default 2) levels are locked, plus the leaf.
        Per-protect cost drops from O(prefix_length) to O(leaf+small).
        """
        # Resolve the TTL: explicit arg takes precedence over the env var.
        # TTL=0 means "no-op": do not even bookkeep the protect; this is
        # the canonical "feature off" semantic. A negative TTL is treated
        # the same as 0.
        if ttl_s is not None:
            ttl = float(ttl_s)
        else:
            try:
                ttl = float(os.environ.get("SGLANG_LOSSY_PREFETCH_TTL_S", "60"))
            except ValueError:
                ttl = 60.0
        if ttl <= 0:
            return False
        now = time.monotonic()
        entry.prefetch_protected_until = max(entry.prefetch_protected_until, now + ttl)
        entry.prefetch_steps_remaining = max(entry.prefetch_steps_remaining, steps_to_use)

        if entry.prefetch_lock_held:
            return False

        node = entry.source_node
        if node is not None and not node.evicted:
            # Safety-net cap: try to FIFO-evict oldest still-locked protected
            # anchors first; only reject if the cap is still exceeded after
            # the eviction sweep.  Without this, anchors churn into the cap
            # at 9.6x its capacity and the pool starves the next request.
            cap = self._agenttemplatekv_protected_size_cap()
            if cap > 0 and self.protected_size_ + len(entry.token_ids) > cap:
                self._agenttemplatekv_evict_oldest_protected(
                    need_tokens=len(entry.token_ids), cap=cap
                )
                if self.protected_size_ + len(entry.token_ids) > cap:
                    # Rate-limiter: only log every 1024 rejects to avoid
                    # log spam in sustained cap-hit regimes. The miss counter
                    # is always incremented; the warning is for human ops.
                    self._agenttemplatekv_reject_count = (
                        getattr(self, "_agenttemplatekv_reject_count", 0) + 1
                    )
                    if self._agenttemplatekv_reject_count % 1024 == 1:
                        logger.warning(
                            "agenttemplatekv_protect rejected "
                            "(suppressing further logs; count=%d): "
                            "protected_size_=%d + token_ids=%d > cap=%d "
                            "(eviction sweep could not free enough)",
                            self._agenttemplatekv_reject_count,
                            self.protected_size_, len(entry.token_ids), cap,
                        )
                    if req is not None:
                        setattr(
                            req,
                            "agenttemplatekv_prefetch_miss_count",
                            getattr(
                                req, "agenttemplatekv_prefetch_miss_count", 0
                            )
                            + 1,
                        )
                    return False
            # Capped walk: only lock the leaf + max_ancestors ancestors when
            # the active cache backend supports it. HiRadixCache subclasses
            # may still expose the older inc_lock_ref(node) signature.
            try:
                result = self.inc_lock_ref(node, max_ancestors=max_ancestors)
            except TypeError as exc:
                if "max_ancestors" not in str(exc):
                    raise
                result = self.inc_lock_ref(node)
            setattr(entry, "_protected_ancestor_nodes", result.locked_nodes or [])
            entry.prefetch_lock_held = True
            return True

        if req is not None:
            setattr(
                req,
                "agenttemplatekv_prefetch_miss_count",
                getattr(req, "agenttemplatekv_prefetch_miss_count", 0) + 1,
            )
        return False

    def _agenttemplatekv_evict_oldest_protected(
        self, need_tokens: int, cap: int
    ) -> int:
        """FIFO evict the oldest still-locked protected anchors to make room
        for a new protect of ``need_tokens`` tokens under the cap.

        Walks ``anchor_kv_store`` under ``anchor_kv_store_lock``, finds
        entries where ``prefetch_lock_held`` is True, sorts by
        ``prefetch_protected_until`` (oldest expiration first), and
        releases the earliest ones until the cap can satisfy the request
        or there are no more candidates.

        Returns the number of tokens freed. Caller should re-check
        ``protected_size_ + need_tokens <= cap`` after this returns.
        """
        if self.disable:
            return 0
        freed = 0
        with self.anchor_kv_store_lock:
            candidates = []
            for entries in self.anchor_kv_store.values():
                for entry in entries:
                    if entry.prefetch_lock_held:
                        candidates.append(
                            (entry.prefetch_protected_until, entry)
                        )
            if not candidates:
                return 0
            # FIFO by expiration timestamp; ties broken by id() for stability.
            candidates.sort(key=lambda x: (x[0], id(x[1])))
            for _ts, entry in candidates:
                if self.protected_size_ + need_tokens <= cap:
                    break
                if self._agenttemplatekv_release_entry(entry):
                    freed += len(entry.token_ids)
                    logger.info(
                        "agenttemplatekv_evict_oldest: released sig=%s "
                        "tokens=%d (freed=%d, still_need=%d, "
                        "protected_size_=%d)",
                        entry.signature,
                        len(entry.token_ids),
                        freed,
                        need_tokens,
                        self.protected_size_,
                    )
        return freed

    def _agenttemplatekv_release_entry(self, entry: AnchorKVEntry) -> bool:
        if not entry.prefetch_lock_held:
            return False
        # Release the capped chain if stored (post-fix path). Fall back to
        # the full walk for entries from older engine versions.
        locked = getattr(entry, "_protected_ancestor_nodes", None)
        if locked is None:
            node = entry.source_node
            if node is not None and not node.evicted:
                # Full walk: matches the original dec_lock_ref contract.
                self.dec_lock_ref(node)
        else:
            # Capped walk: re-derive max_ancestors from the stored list.
            # Each step from leaf to root re-releases the corresponding
            # ancestor; we walk from leaf upward via dec_lock_ref with the
            # same cap used in inc_lock_ref.
            n_ancestors = len(locked)
            deepest = locked[0] if locked else entry.source_node
            if deepest is not None and not deepest.evicted:
                try:
                    self.dec_lock_ref(deepest, max_ancestors=n_ancestors)
                except TypeError as exc:
                    if "max_ancestors" not in str(exc):
                        raise
                    self.dec_lock_ref(deepest)
            setattr(entry, "_protected_ancestor_nodes", [])
        entry.prefetch_lock_held = False
        entry.prefetch_protected_until = 0.0
        entry.prefetch_steps_remaining = 0
        return True

    def _agenttemplatekv_release_expired_prefetch_entries(self, req: Optional[Req] = None):
        now = time.monotonic()
        released = 0
        with self.anchor_kv_store_lock:
            for entries in self.anchor_kv_store.values():
                for entry in entries:
                    if (
                        entry.prefetch_lock_held
                        and entry.prefetch_protected_until > 0
                        and entry.prefetch_protected_until <= now
                    ):
                        if self._agenttemplatekv_release_entry(entry):
                            released += len(entry.token_ids)
        if req is not None and released:
            setattr(
                req,
                "agenttemplatekv_prefetch_expired_tokens",
                getattr(req, "agenttemplatekv_prefetch_expired_tokens", 0) + released,
            )

    def agenttemplatekv_prefetch_codebases(self, req: Req, tokenizer=None, max_hints: int = 8):
        """Backward-compat alias for :class:`AgentTemplateKVCache.prefetch_codebases`.

        New code should construct an :class:`AgentTemplateKVCache` and call
        :meth:`AgentTemplateKVCache.prefetch_codebases` directly. The
        scheduler's isinstance check will then dispatch to the subclass
        implementation. This alias remains so older call sites that call
        ``tree_cache.agenttemplatekv_prefetch_codebases(...)`` keep working.
        """
        if not self._agenttemplatekv_enabled():
            return
        # Defer to the subclass implementation if available; otherwise
        # run the same body inline. The inline path matches the original
        # RadixCache behaviour and is used by tests that build a stock
        # RadixCache with no subclassing.
        from sglang.srt.mem_cache.agenttemplatekv_cache import AgentTemplateKVCache
        if isinstance(self, AgentTemplateKVCache):
            AgentTemplateKVCache.prefetch_codebases(self, req, tokenizer=tokenizer, max_hints=max_hints)
            return
        # Inline fallback (used by tests and stock radix cache that hasn't
        # been subclassed). Identical body to AgentTemplateKVCache.prefetch_codebases.
        self._agenttemplatekv_release_expired_prefetch_entries(req)
        hints = getattr(req, "codebase_prefetch_hints", None) or []
        if not hints:
            return
        for hint in hints[:max_hints]:
            if not isinstance(hint, dict):
                continue
            content_signature = str(
                hint.get("content_signature")
                or hint.get("code_content_signature")
                or ""
            )
            if not content_signature:
                continue
            entries = []
            with self.anchor_kv_store_lock:
                entries = list(self.anchor_kv_store.get(content_signature, []))
            if not entries:
                req.agenttemplatekv_prefetch_miss_count += 1
                continue
            text = hint.get("text") or hint.get("code") or hint.get("content")
            text_token_ids = None
            if text and tokenizer is not None:
                try:
                    text_token_ids = tokenizer.encode(text, add_special_tokens=False)
                except Exception:
                    text_token_ids = None
            matched_entry = None
            for entry in entries:
                if entry.code_content_signature != content_signature:
                    continue
                if text_token_ids is not None and list(entry.token_ids.tolist()) != list(text_token_ids):
                    continue
                matched_entry = entry
                break
            if matched_entry is None:
                req.agenttemplatekv_prefetch_miss_count += 1
                continue
            steps_to_use = int(hint.get("steps_to_use") or 1)
            locked_now = self._agenttemplatekv_protect_entry(
                matched_entry, req=req, steps_to_use=max(1, steps_to_use)
            )
            matched_tokens = len(matched_entry.token_ids)
            req.codebase_prefetch_matched_tokens += matched_tokens
            req.codebase_prefetch_success_count += 1
            req.codebase_prefetch_device_hit_count += 1
            req.agenttemplatekv_prefetch_hit_count += 1
            req.agenttemplatekv_prefetch_protected_tokens += matched_tokens
            if locked_now:
                req.agenttemplatekv_prefetch_newly_protected_tokens += matched_tokens

    def _store_anchor_kv(
        self,
        req: Req,
        kv_indices: torch.Tensor,
        source_node: Optional[TreeNode] = None,
    ):
        """Extract anchor block KV and store for position-aligned reuse.

        Uses ``code_anchor_token_spans`` (if provided) to identify the exact
        token positions of the anchor block within the prompt. Falls back to
        line-based spans with approximate mapping.
        """
        signature = getattr(req, "code_anchor_signature", "") or ""
        content_signature = getattr(req, "code_content_signature", "") or ""
        token_spans = getattr(req, "code_anchor_token_spans", None) or []
        if not signature:
            return

        if not token_spans:
            # The lossy reuse path needs explicit token positions to know
            # which KV slots to copy. Without them we silently can't store
            # the entry, which means the next request can't fuzzy-match
            # against it. Surface this in logs so the misconfiguration is
            # visible in production (Bug C fix).
            logger.warning(
                "[anchor_kv_store] skip store for rid=%s sig=%s content=%s: "
                "missing code_anchor_token_spans on request",
                getattr(req, "rid", "?"), signature, content_signature,
            )
            setattr(req, "lossy_anchor_store_skipped_missing_token_spans", 1)
            return

        token_ids = list(req.origin_input_ids) + list(req.output_ids)
        max_pos = len(token_ids)

        for span in token_spans:
            if not isinstance(span, dict):
                continue
            start = int(span.get("start_token", -1))
            end = int(span.get("end_token", -1))
            if start < 0 or end <= start or end > max_pos:
                continue

            span_token_ids = torch.tensor(
                token_ids[start:end], dtype=torch.int64, device=self.device
            )
            span_kv_indices = kv_indices[start:end].clone()

            entry = AnchorKVEntry(
                signature=signature,
                code_content_signature=content_signature,
                token_ids=span_token_ids,
                kv_indices=span_kv_indices,
                start_pos=start,
                source_node=source_node,
                prefix_context_signature=_token_prefix_signature(token_ids, start),
            )
            segment_content_signature = str(
                span.get("content_signature", "") or content_signature
            )
            if not segment_content_signature:
                continue

            entry.code_content_signature = segment_content_signature
            # Optional semantic suffix-copy length decider: compute and stash
            # chunk embeddings for this anchor so the consumer can decide how
            # much to copy based on per-chunk cosine to the request.
            try:
                from sglang.srt.mem_cache.semantic_suffix import entry_chunks_for
                llm_tokenizer = getattr(self, "tokenizer", None)
                if llm_tokenizer is None:
                    # Reuse the same tokenizer the scheduler used to tokenize
                    # the request, falling back to the radix cache tokenizer.
                    llm_tokenizer = getattr(req, "tokenizer", None)
                entry.chunk_embeddings = entry_chunks_for(
                    entry.token_ids, llm_tokenizer,
                )
            except Exception as ce:  # pragma: no cover - defensive
                logger.debug(
                    "[anchor_kv_store] chunk_embeddings skipped for sig=%s: %s",
                    segment_content_signature, ce,
                )
                entry.chunk_embeddings = None
            with self.anchor_kv_store_lock:
                self.anchor_kv_store.setdefault(segment_content_signature, []).append(entry)
            setattr(
                req,
                "lossy_anchor_store_entry_count",
                getattr(req, "lossy_anchor_store_entry_count", 0) + 1,
            )
            setattr(
                req,
                "lossy_anchor_store_token_count",
                getattr(req, "lossy_anchor_store_token_count", 0) + len(entry.token_ids),
            )
            if segment_content_signature in self._extract_prefetch_hint_signatures(req):
                locked_now = self._agenttemplatekv_protect_entry(
                    entry,
                    req=req,
                    steps_to_use=1,
                )
                matched_tokens = len(entry.token_ids)
                req.agenttemplatekv_prefetch_protected_tokens += matched_tokens
                if locked_now:
                    req.agenttemplatekv_prefetch_newly_protected_tokens += matched_tokens
            logger.info(
                "[anchor_kv_store] stored anchor_sig=%s content_sig=%s start=%d len=%d",
                signature, segment_content_signature, start, end - start,
            )

    # -----------------------------------------------------------------
    # Per-placeholder anchor pool (Duke 2026 KVCOMM-style).
    #
    # While `_store_anchor_kv` writes per-`code_content_signature` entries
    # used by the byte-exact suffix path, this method writes per-`slot_id`
    # entries whose `pool_embedding` is consumed by `_try_placeholder_knn_
    # lossy_match` for embedding k-NN reuse across agents.  Each slot's
    # entries are bounded by `placeholder_pool_max_per_slot` (LRU).
    # -----------------------------------------------------------------

    def _placeholder_store_enabled(self) -> bool:
        """Master switch for write-back.  Default ON; off when env=0 or
        SGLANG_SEMANTIC_SUFFIX_ENABLED=0 (shared embedder model)."""
        from sglang.srt.mem_cache.semantic_suffix import is_enabled as _ss_enabled
        if not _ss_enabled():
            return False
        return os.environ.get("SGLANG_PLACEHOLDER_STORE_ENABLED", "1") == "1"

    def _decode_placeholder_span(self, span_token_ids: torch.Tensor,
                                  fallback_tokenizer=None) -> str:
        """Decode a span's tokens to text using whichever tokenizer is
        available on this cache / request.  Returns empty string on
        failure or when no tokenizer is reachable."""
        tok = (
            getattr(self, "tokenizer", None)
            or fallback_tokenizer
        )
        if tok is None:
            return ""
        try:
            return tok.decode(
                span_token_ids.detach().cpu().tolist(),
                skip_special_tokens=True,
            )
        except Exception:
            return ""

    def _placeholder_f1(self, predicted_text: str, actual_text: str) -> float:
        """Wrapper around `text_utils.token_f1` so we don't import the
        benchmark at runtime."""
        from sglang.srt.mem_cache.text_utils import token_f1
        try:
            return token_f1(predicted_text, actual_text)
        except Exception:
            return 0.0

    def _evict_placeholder_pool_slot_locked(self, slot_id: str) -> None:
        """LRU trim on a slot's list to <= placeholder_pool_max_per_slot.
        Caller must hold `placeholder_anchor_pool_lock`."""
        entries = self.placeholder_anchor_pool.get(slot_id, [])
        if len(entries) <= self.placeholder_pool_max_per_slot:
            return
        # Sort by last_access_time descending; keep the freshest.
        entries.sort(key=lambda e: e.last_access_time, reverse=True)
        self.placeholder_anchor_pool[slot_id] = entries[
            : self.placeholder_pool_max_per_slot
        ]

    def _store_placeholder_anchor_kv(
        self,
        req: Req,
        kv_indices: torch.Tensor,
        source_node: Optional[TreeNode] = None,
    ) -> None:
        """Write per-placeholder KV blocks + their MiniLM pool embeddings
        into the per-slot pool.  Mirrors `_store_anchor_kv` but keyed by
        `slot_id` and using a single embedding per slot text (not per-
        chunk).  Skips entries whose predicted text diverges from the
        actually-prefilled text (F1 < SGLANG_PLACEHOLDER_STORE_MIN_F1).
        """
        if not self._placeholder_store_enabled():
            return
        spans = getattr(req, "placeholder_anchor_token_spans", None) or []
        if not spans:
            return
        from sglang.srt.mem_cache.semantic_suffix import embed_single_text

        token_ids = list(req.origin_input_ids) + list(req.output_ids)
        max_pos = len(token_ids)
        stored = 0
        skipped_low_f1 = 0
        skipped_invalid = 0
        min_f1 = float(
            os.environ.get("SGLANG_PLACEHOLDER_STORE_MIN_F1", "0.60")
        )
        for span in spans:
            if not isinstance(span, dict):
                skipped_invalid += 1
                continue
            slot_id = str(span.get("slot_id", "") or "")
            start = int(span.get("start_token", -1))
            end = int(span.get("end_token", -1))
            text = str(span.get("text", "") or "")
            if not slot_id or start < 0 or end <= start or end > max_pos:
                skipped_invalid += 1
                continue

            span_token_ids = torch.tensor(
                token_ids[start:end], dtype=torch.int64, device=self.device
            )
            span_kv_indices = kv_indices[start:end].clone()

            actual_text = self._decode_placeholder_span(
                span_token_ids,
                fallback_tokenizer=getattr(req, "tokenizer", None),
            )
            # Use actual_text as fallback when client didn't supply text.
            embed_target = text or actual_text
            # If we have no tokenizer we can't compute F1 — fall back to
            # permissive "accept" (same shape as the v10c doc's known
            # limitation for chunk_embeddings; the entry will still be
            # validated by the k-NN read path's cosine floor).
            if text and actual_text:
                f1_score = self._placeholder_f1(text, actual_text)
            else:
                f1_score = 1.0  # unknown — accept by default
            if text and actual_text and f1_score < min_f1:
                logger.info(
                    "[placeholder_anchor_pool] skip store slot=%s start=%d "
                    "len=%d: F1=%.3f < %.3f",
                    slot_id, start, end - start, f1_score, min_f1,
                )
                skipped_low_f1 += 1
                continue

            try:
                emb = embed_single_text(embed_target)
            except Exception as ce:  # pragma: no cover - defensive
                logger.debug(
                    "[placeholder_anchor_pool] embed_single_text failed for "
                    "slot=%s: %s",
                    slot_id, ce,
                )
                emb = None

            entry = AnchorKVEntry(
                signature=(
                    f"placeholder:{slot_id}:"
                    f"{str(span.get('content_signature', ''))[:16]}"
                ),
                code_content_signature=str(
                    span.get("content_signature", "") or ""
                ),
                token_ids=span_token_ids,
                kv_indices=span_kv_indices,
                start_pos=start,
                source_node=source_node,
                slot_id=slot_id,
                slot_label=str(span.get("label", "") or ""),
                pool_embedding=emb,
                embedding_text=embed_target,
                last_access_time=time.monotonic(),
            )
            with self.placeholder_anchor_pool_lock:
                self.placeholder_anchor_pool.setdefault(slot_id, []).append(entry)
                self._evict_placeholder_pool_slot_locked(slot_id)
            stored += 1

        setattr(req, "placeholder_anchor_store_entry_count", stored)
        setattr(req, "placeholder_anchor_store_skipped_low_f1_count", skipped_low_f1)
        setattr(req, "placeholder_anchor_store_skipped_invalid_count", skipped_invalid)
        if stored or skipped_low_f1 or skipped_invalid:
            logger.info(
                "[placeholder_anchor_pool] rid=%s stored=%d skipped_low_f1=%d "
                "skipped_invalid=%d",
                getattr(req, "rid", "?"), stored, skipped_low_f1, skipped_invalid,
            )

    def _resolve_lossy_match(self, req: Req) -> AnchorMatchResult:
        request_meta = self._build_req_anchor_metadata(req)
        candidate_nodes = list(self._iter_nodes_with_anchor_metadata())
        best_node, best_result = select_best_match(
            request_meta,
            ((node, self._build_node_anchor_metadata(node)) for node in candidate_nodes),
        )
        is_first = getattr(req, "lossy_first_match_reason", None) is None
        setattr(req, "lossy_candidate_count", len(candidate_nodes))
        setattr(req, "lossy_final_match_reason", best_result.match_reason)
        setattr(req, "lossy_final_rejected_reason", best_result.rejected_reason)
        setattr(req, "lossy_final_reuse_allowed", best_result.reuse_allowed)
        setattr(req, "lossy_final_reuse_confidence", best_result.reuse_confidence)
        setattr(req, "lossy_final_matched_anchor_signature", best_result.matched_anchor_signature)
        setattr(req, "lossy_final_matched_content_signature", best_result.matched_content_signature)
        setattr(req, "lossy_final_syntax_region_type", best_result.syntax_region_type)
        setattr(req, "lossy_final_matched_node_id", getattr(best_node, "id", None))
        # context_aware_confidence modifier telemetry
        setattr(req, "lossy_predicted_distance", best_result.predicted_distance)
        setattr(req, "lossy_context_aware_confidence", best_result.reuse_confidence)
        setattr(req, "lossy_context_aware_multiplier", best_result.context_aware_multiplier)
        if is_first:
            setattr(req, "lossy_first_match_reason", best_result.match_reason)
            setattr(req, "lossy_first_rejected_reason", best_result.rejected_reason)
            setattr(req, "lossy_first_reuse_allowed", best_result.reuse_allowed)
            setattr(req, "lossy_first_reuse_confidence", best_result.reuse_confidence)
            setattr(req, "lossy_first_matched_anchor_signature", best_result.matched_anchor_signature)
            setattr(req, "lossy_first_matched_content_signature", best_result.matched_content_signature)
            setattr(req, "lossy_first_syntax_region_type", best_result.syntax_region_type)
            setattr(req, "lossy_first_matched_node_id", getattr(best_node, "id", None))
        setattr(req, "lossy_match_reason", best_result.match_reason)
        setattr(req, "lossy_rejected_reason", best_result.rejected_reason)
        setattr(req, "lossy_reuse_allowed", best_result.reuse_allowed)
        setattr(req, "lossy_reuse_confidence", best_result.reuse_confidence)
        setattr(req, "lossy_matched_anchor_signature", best_result.matched_anchor_signature)
        setattr(req, "lossy_matched_content_signature", best_result.matched_content_signature)
        setattr(req, "lossy_syntax_region_type", best_result.syntax_region_type)
        setattr(req, "lossy_matched_node_id", getattr(best_node, "id", None))
        logger.info(
            "[lossy_reuse] rid=%s candidates=%d req_sig=%s allowed=%s reason=%s rejected=%s matched_sig=%s first=%s",
            getattr(req, "rid", ""),
            len(candidate_nodes),
            getattr(req, "code_anchor_signature", "") or "",
            best_result.reuse_allowed,
            best_result.match_reason,
            best_result.rejected_reason,
            best_result.matched_anchor_signature,
            is_first,
        )
        return best_node, best_result

    def _ensure_rope_inv_freq(self):
        if self.rope_inv_freq is None and self.rope_rotary_dim > 0:
            self.rope_inv_freq = 1.0 / (
                self.rope_base
                ** (
                    torch.arange(0, self.rope_rotary_dim, 2, dtype=torch.float32)
                    / self.rope_rotary_dim
                )
            )

    def _compute_rope_cos_sin_for_delta(
        self, delta_positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin for RoPE delta rotation.

        Args:
            delta_positions: [num_tokens] int tensor of new_pos - old_pos.

        Returns:
            cos: [num_tokens, rotary_dim] float32
            sin: [num_tokens, rotary_dim] float32
        """
        self._ensure_rope_inv_freq()
        inv_freq = self.rope_inv_freq.to(device=delta_positions.device)
        freqs = torch.einsum("i,j->ij", delta_positions.float(), inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        return cos, sin

    def _apply_rope_delta_to_keys(
        self, k_buffer, dst_slots: torch.Tensor, delta_positions: torch.Tensor
    ):
        """Apply RoPE delta rotation to key cache at dst_slots.

        Reads keys from the cache at dst_slots, applies a RoPE rotation
        corresponding to delta_positions, and writes them back. Only keys
        are rotated (values are position-independent).

        Args:
            k_buffer: list of [max_tokens, num_kv_heads * head_size] tensors per layer.
            dst_slots: [num_tokens] int tensor of slot indices.
            delta_positions: [num_tokens] int tensor of position deltas.
        """
        cos, sin = self._compute_rope_cos_sin_for_delta(delta_positions)
        cos = cos.to(device=dst_slots.device)
        sin = sin.to(device=dst_slots.device)
        head_size = self.rope_rotary_dim
        is_neox = self.rope_is_neox_style

        dst_flat = dst_slots.view(-1).long()
        num_tokens = dst_flat.shape[0]

        # Timing via CUDA events for micro-benchmarking
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        for k_cache in k_buffer:
            # k_cache: [max_tokens, num_kv_heads, head_size] (3D)
            k_selected = k_cache[dst_flat]  # [num_tokens, num_kv_heads, head_size]
            from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb

            k_rotated = apply_rotary_emb(k_selected, cos, sin, is_neox)
            k_cache[dst_flat] = k_rotated

        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)

        logger.warning(
            "[rope_delta] rotated %d tokens x %d layers in %.3f ms (%.3f ms/layer)",
            num_tokens,
            len(k_buffer),
            elapsed_ms,
            elapsed_ms / len(k_buffer) if k_buffer else 0,
        )

    def _apply_rope_delta_to_head(
        self, k_buffer, dst_slots: torch.Tensor, head_len: int, delta: int,
    ):
        """Head-only RoPE delta rotation (Phase 2.1, EPIC-inspired).

        Only the first `head_len` tokens at dst_slots are rotated to encode
        the correct global position.  The remaining tokens retain their
        chunk-local position-0 RoPE.

        Reduces rotation cost from O(entry_len x layer_num) to
        O(head_len x layer_num).  For entry_len=2245, head_len=2,
        layer_num=28: 62,860 -> 56 ops (~1120x cheaper).

        EPIC (ICML 2025) shows k=2 is sufficient for <=7% accuracy loss
        on standard benchmarks.

        Returns: number of tokens actually rotated (0 if no-op).
        """
        if head_len <= 0 or delta == 0:
            return 0
        n = int(dst_slots.shape[0])
        head_len = min(int(head_len), n)
        if head_len <= 0:
            return 0
        head_dst = dst_slots[:head_len].contiguous()
        delta_tensor = torch.full(
            (head_len,), int(delta), dtype=torch.long, device=head_dst.device,
        )
        self._apply_rope_delta_to_keys(k_buffer, head_dst, delta_tensor)
        return head_len

    def _try_lossy_fuzzy_match(
        self,
        req: Req,
        key: RadixKey,
        exact_values: List[torch.Tensor],
        exact_node: TreeNode,
        best_node: Optional[TreeNode],
    ) -> Tuple[List[torch.Tensor], TreeNode]:
        """Extend cache match via anchor KV store with active RoPE rotation.

        Unlike the original position-aligned approach, this method can reuse
        cached KV from ANY position by applying a RoPE delta rotation to
        re-encode the keys for the new position.

        Controlled by the ``SGLANG_LOSSY_FUZZY_MATCH`` environment variable.
        """
        lossy_env = os.environ.get("SGLANG_LOSSY_FUZZY_MATCH", "0")
        if best_node is None or lossy_env != "1":
            return exact_values, exact_node

        exact_len = sum(len(v) for v in exact_values) if exact_values else 0
        if exact_len >= len(key):
            return exact_values, exact_node

        multi_anchor_enabled = os.environ.get("SGLANG_LOSSY_MULTI_ANCHOR", "0") == "1"
        copied_anchor_count = int(getattr(req, "lossy_anchor_multi_copy_count", 0) or 0)

        match_reason = getattr(req, "lossy_first_match_reason", "")
        if match_reason not in (
            "exact_anchor_signature",
            "exact_code_content_signature",
            "span_overlap_high",
            "span_overlap_medium",
        ) and not (multi_anchor_enabled and copied_anchor_count > 0):
            return exact_values, exact_node

        matched_content_sig = getattr(req, "lossy_first_matched_content_signature", "") or ""
        if not matched_content_sig:
            matched_content_sig = getattr(req, "code_content_signature", "") or ""
        if not matched_content_sig:
            return exact_values, exact_node

        key_tokens = key.token_ids
        token_spans = getattr(req, "code_anchor_token_spans", None) or []
        candidate_content_sigs = [matched_content_sig]
        for span in token_spans:
            span_sig = str(span.get("content_signature", "") or "")
            if span_sig and span_sig not in candidate_content_sigs:
                candidate_content_sigs.append(span_sig)
        entries_by_sig = []
        lookup_entries = 0
        with self.anchor_kv_store_lock:
            for content_sig in candidate_content_sigs:
                entries = list(self.anchor_kv_store.get(content_sig, []))
                lookup_entries += len(entries)
                if entries:
                    entries_by_sig.append((content_sig, entries))
        setattr(req, "lossy_anchor_store_lookup_entries", lookup_entries)
        if not entries_by_sig:
            setattr(req, "lossy_anchor_match_fail_reason", "no_anchor_store_entry")
            return exact_values, exact_node

        skip_check = os.environ.get("SGLANG_LOSSY_SKIP_TOKEN_CHECK", "0") == "1"
        token_mismatch_count = 0
        span_shape_mismatch_count = 0
        prefix_covers_count = 0
        total_copy_len = int(getattr(req, "lossy_anchor_suffix_copy_len", 0) or 0)
        total_planned_copy_len = int(
            getattr(req, "lossy_anchor_suffix_copy_planned_len", 0) or 0
        )
        total_gap_len = int(getattr(req, "lossy_anchor_match_gap_len", 0) or 0)
        total_gap_recompute_len = int(getattr(req, "lossy_anchor_gap_recompute_len", 0) or 0)
        any_copy_truncated = bool(getattr(req, "lossy_anchor_suffix_copy_truncated", False))

        for req_content_signature, entries in entries_by_sig:
            for entry in entries:
                if (
                    not req_content_signature
                    or not entry.code_content_signature
                    or req_content_signature != entry.code_content_signature
                ):
                    continue
                entry_len = len(entry.token_ids)

                # Determine the anchor's position in the current request.
                # Use code_anchor_token_spans if available; otherwise fall back
                # to entry.start_pos (legacy position-aligned path).
                anchor_pos = None
                matched_span_max_suffix_copy_len = 0
                matched_span_suffix_recompute_head_len = 0
                for span in token_spans:
                    s_start = span.get("start_token", -1)
                    s_end = span.get("end_token", -1)
                    span_content_signature = str(
                        span.get("content_signature", "") or req_content_signature
                    )
                    if span_content_signature != entry.code_content_signature:
                        continue
                    if s_start < 0 or s_end > len(key_tokens) or s_end - s_start != entry_len:
                        span_shape_mismatch_count += 1
                        continue
                    if skip_check:
                        anchor_pos = s_start
                        matched_span_max_suffix_copy_len = int(
                            span.get("max_suffix_copy_len", 0) or 0
                        )
                        matched_span_suffix_recompute_head_len = int(
                            span.get("suffix_recompute_head_len", 0) or 0
                        )
                        break
                    span_tokens = torch.tensor(
                        key_tokens[s_start:s_end],
                        dtype=entry.token_ids.dtype,
                        device=entry.token_ids.device,
                    )
                    if torch.equal(span_tokens, entry.token_ids):
                        anchor_pos = s_start
                        matched_span_max_suffix_copy_len = int(
                            span.get("max_suffix_copy_len", 0) or 0
                        )
                        matched_span_suffix_recompute_head_len = int(
                            span.get("suffix_recompute_head_len", 0) or 0
                        )
                        break
                    token_mismatch_count += 1

                if anchor_pos is None:
                    # Fallback: position-aligned token match at entry.start_pos
                    entry_end = entry.start_pos + entry_len
                    if entry_end > len(key_tokens) or entry.start_pos < 0:
                        continue
                    if not skip_check:
                        current_span = torch.tensor(
                            key_tokens[entry.start_pos : entry_end],
                            dtype=entry.token_ids.dtype,
                            device=entry.token_ids.device,
                        )
                        if not torch.equal(current_span, entry.token_ids):
                            token_mismatch_count += 1
                            continue
                    anchor_pos = entry.start_pos

                anchor_end = anchor_pos + entry_len
                if exact_len >= anchor_end:
                    prefix_covers_count += 1
                    continue

                suffix_start = max(0, exact_len - anchor_pos)
                planned_copy_len = entry_len - suffix_start
                max_planned_suffix_copy_len = int(
                    os.environ.get("SGLANG_LOSSY_MAX_PLANNED_SUFFIX_COPY_LEN", "0")
                    or 0
                )
                if (
                    max_planned_suffix_copy_len > 0
                    and planned_copy_len > max_planned_suffix_copy_len
                ):
                    setattr(req, "lossy_rejected_reason", "planned_suffix_copy_too_long")
                    setattr(req, "lossy_anchor_match_fail_reason", "planned_suffix_copy_too_long")
                    setattr(req, "lossy_anchor_suffix_copy_len", 0)
                    setattr(req, "lossy_anchor_suffix_copy_planned_len", planned_copy_len)
                    setattr(req, "lossy_anchor_suffix_copy_truncated", False)
                    logger.info(
                        "[agenttemplatekv] reject content_sig=%s planned_copy_len=%d max_planned=%d",
                        req_content_signature,
                        planned_copy_len,
                        max_planned_suffix_copy_len,
                    )
                    continue

                recompute_gap_enabled = os.environ.get("SGLANG_LOSSY_RECOMPUTE_GAP", "0") == "1"
                stage_recompute_enabled = os.environ.get("SGLANG_LOSSY_STAGE_RECOMPUTE_GAP", "0") == "1"
                suffix_recompute_head_len = max(
                    0,
                    matched_span_suffix_recompute_head_len
                    or int(os.environ.get("SGLANG_LOSSY_SUFFIX_RECOMPUTE_HEAD_LEN", "0") or 0),
                )
                target_prefix_for_copy = min(
                    anchor_end,
                    anchor_pos + min(suffix_recompute_head_len, entry_len),
                )
                planned_recompute_gap_len = max(0, target_prefix_for_copy - exact_len)
                max_recompute_gap_len = int(
                    os.environ.get("SGLANG_LOSSY_MAX_RECOMPUTE_GAP_LEN", "0") or 0
                )
                if (
                    recompute_gap_enabled
                    and max_recompute_gap_len > 0
                    and planned_recompute_gap_len > max_recompute_gap_len
                ):
                    setattr(req, "lossy_rejected_reason", "recompute_gap_too_long")
                    setattr(req, "lossy_anchor_match_fail_reason", "recompute_gap_too_long")
                    setattr(req, "lossy_anchor_match_gap_len", max(0, anchor_pos - exact_len))
                    setattr(req, "lossy_anchor_gap_recompute_len", planned_recompute_gap_len)
                    setattr(req, "lossy_anchor_suffix_recompute_head_len", target_prefix_for_copy - anchor_pos)
                    setattr(req, "lossy_anchor_suffix_copy_len", 0)
                    setattr(req, "lossy_anchor_suffix_copy_planned_len", planned_copy_len)
                    setattr(req, "lossy_anchor_context_copy_ready", False)
                    setattr(req, "lossy_anchor_context_aligned", False)
                    setattr(
                        req,
                        "lossy_anchor_context_align_fail_reason",
                        "planned_recompute_gap_too_long",
                    )
                    logger.info(
                        "[agenttemplatekv] reject content_sig=%s recompute_gap=%d max_recompute_gap=%d",
                        req_content_signature,
                        planned_recompute_gap_len,
                        max_recompute_gap_len,
                    )
                    continue
                staged_target_prefix_len = int(
                    getattr(req, "lossy_anchor_context_target_prefix_len", 0) or 0
                )
                staged_copy_ready = (
                    stage_recompute_enabled
                    and staged_target_prefix_len == target_prefix_for_copy
                    and exact_len >= target_prefix_for_copy
                    and getattr(req, "lossy_anchor_context_align_stage", None)
                    == "recompute_gap_chunk"
                )
                if staged_copy_ready and copied_anchor_count <= 0:
                    # The request already carries the first-stage recompute
                    # length as metadata.  Treat this second-stage suffix copy
                    # as the first successful copy in the aggregate so the
                    # telemetry is not counted twice.
                    total_gap_recompute_len = 0
                if recompute_gap_enabled:
                    target_prefix_sig = _token_prefix_signature(key_tokens, anchor_pos)
                    if (
                        entry.prefix_context_signature
                        and target_prefix_sig != entry.prefix_context_signature
                        and not staged_copy_ready
                    ):
                        if exact_len < anchor_pos:
                            staged_gap = target_prefix_for_copy - exact_len
                            if stage_recompute_enabled:
                                rejected_reason = "context_aligned_staging"
                                fail_reason = "staged_recompute_gap_pending"
                                target_prefix_len = target_prefix_for_copy
                            else:
                                rejected_reason = "context_aligned_not_supported"
                                fail_reason = "staged_recompute_gap_not_supported"
                                target_prefix_len = 0
                            setattr(req, "lossy_rejected_reason", rejected_reason)
                            setattr(req, "lossy_anchor_match_gap_len", staged_gap)
                            setattr(req, "lossy_anchor_gap_recompute_len", staged_gap)
                            setattr(
                                req,
                                "lossy_anchor_suffix_recompute_head_len",
                                target_prefix_for_copy - anchor_pos,
                            )
                            if copied_anchor_count <= 0:
                                setattr(req, "lossy_anchor_suffix_copy_len", 0)
                            setattr(req, "lossy_anchor_context_copy_ready", False)
                            setattr(req, "lossy_anchor_context_aligned", False)
                            setattr(
                                req,
                                "lossy_anchor_context_align_fail_reason",
                                fail_reason,
                            )
                            setattr(req, "lossy_anchor_match_fail_reason", rejected_reason)
                            setattr(req, "lossy_anchor_context_target_prefix_len", target_prefix_len)
                            setattr(req, "lossy_anchor_context_prefix_signature_match", False)
                            if copied_anchor_count > 0:
                                return exact_values, exact_node
                            continue
                        setattr(req, "lossy_rejected_reason", "context_aligned_prefix_mismatch")
                        setattr(req, "lossy_anchor_match_gap_len", 0)
                        setattr(req, "lossy_anchor_gap_recompute_len", getattr(req, "lossy_anchor_gap_recompute_len", 0))
                        setattr(req, "lossy_anchor_suffix_copy_len", 0)
                        setattr(req, "lossy_anchor_context_aligned", False)
                        setattr(
                            req,
                            "lossy_anchor_context_align_fail_reason",
                            "prefix_context_mismatch",
                        )
                        setattr(req, "lossy_anchor_match_fail_reason", "context_aligned_prefix_mismatch")
                        setattr(req, "lossy_anchor_context_prefix_signature_match", False)
                        continue
                setattr(req, "lossy_anchor_context_prefix_signature_match", True)

                max_suffix_copy_len = int(os.environ.get("SGLANG_LOSSY_MAX_SUFFIX_COPY_LEN", "0") or 0)
                copy_caps = [
                    cap for cap in (max_suffix_copy_len, matched_span_max_suffix_copy_len) if cap > 0
                ]
                copy_len = min([planned_copy_len, *copy_caps]) if copy_caps else planned_copy_len
                if copy_len <= 0:
                    continue

                # Semantic suffix-copy length decider: replace the hand-tuned
                # caps above with a content-derived length. Compute the request's
                # chunk embeddings at the candidate position and compare to the
                # entry's stored chunk embeddings; take the longest prefix where
                # every chunk cosine >= SGLANG_SEMANTIC_SUFFIX_MIN_COSINE.
                semantic_copy_len = copy_len  # default: no semantic constraint
                semantic_min_cos = 0.0
                semantic_truncated = False
                try:
                    from sglang.srt.mem_cache.semantic_suffix import (
                        is_enabled as semantic_enabled,
                        request_chunks_for,
                        cosine_profile,
                    )
                    if (
                        semantic_enabled()
                        and getattr(entry, "chunk_embeddings", None) is not None
                        and entry.chunk_embeddings.numel() > 0
                    ):
                        req_tokens_at_pos = key_tokens[anchor_pos:anchor_pos + len(entry.token_ids)]
                        req_token_tensor = torch.as_tensor(
                            req_tokens_at_pos, dtype=torch.long,
                        )
                        llm_tokenizer = getattr(self, "tokenizer", None)
                        if llm_tokenizer is None:
                            llm_tokenizer = getattr(req, "tokenizer", None)
                        req_chunks = request_chunks_for(
                            req_token_tensor, llm_tokenizer,
                        )
                        if req_chunks is not None and req_chunks.numel() > 0:
                            sem_len = cosine_profile(
                                req_chunks, entry.chunk_embeddings,
                            )
                            # Compute the min cosine across the kept prefix
                            # for telemetry.
                            from sglang.srt.mem_cache.semantic_suffix import (
                                min_cosine as _min_cos,
                                chunk_tokens as _chunk_tokens,
                            )
                            n_keep = sem_len // _chunk_tokens()
                            if n_keep > 0:
                                sims = (req_chunks[:n_keep] * entry.chunk_embeddings[:n_keep]).sum(dim=-1)
                                semantic_min_cos = float(sims.min().item())
                            # Apply semantic cap. Round DOWN to chunk boundary.
                            if sem_len > 0 and sem_len < copy_len:
                                semantic_copy_len = sem_len
                                semantic_truncated = True
                except Exception as se:  # pragma: no cover - defensive
                    logger.debug(
                        "[agenttemplatekv] semantic suffix check skipped: %s",
                        se,
                    )
                    semantic_copy_len = copy_len
                # Final cap: the smallest of planned, env caps, span cap, and
                # semantic cap.
                if semantic_copy_len < copy_len:
                    copy_len = semantic_copy_len
                if copy_len <= 0:
                    setattr(req, "lossy_anchor_suffix_copy_semantic_len", 0)
                    setattr(req, "lossy_anchor_suffix_copy_semantic_min_cosine", 0.0)
                    setattr(req, "lossy_anchor_suffix_copy_semantic_truncated", True)
                    continue

                gap_len = max(0, anchor_pos - exact_len)
                max_gap = int(os.environ.get("SGLANG_LOSSY_MAX_ZERO_GAP", "16"))
                if gap_len > 0 and recompute_gap_enabled:
                    staged_gap = target_prefix_for_copy - exact_len
                    if stage_recompute_enabled:
                        rejected_reason = "context_aligned_staging"
                        fail_reason = "staged_recompute_gap_pending"
                        target_prefix_len = target_prefix_for_copy
                    else:
                        rejected_reason = "context_aligned_not_supported"
                        fail_reason = "staged_recompute_gap_not_supported"
                        target_prefix_len = 0
                    setattr(req, "lossy_rejected_reason", rejected_reason)
                    setattr(req, "lossy_anchor_match_gap_len", gap_len)
                    setattr(req, "lossy_anchor_gap_recompute_len", staged_gap)
                    setattr(
                        req,
                        "lossy_anchor_suffix_recompute_head_len",
                        target_prefix_for_copy - anchor_pos,
                    )
                    if copied_anchor_count <= 0:
                        setattr(req, "lossy_anchor_suffix_copy_len", 0)
                    setattr(req, "lossy_anchor_context_copy_ready", False)
                    setattr(req, "lossy_anchor_context_aligned", False)
                    setattr(
                        req,
                        "lossy_anchor_context_align_fail_reason",
                        fail_reason,
                    )
                    setattr(req, "lossy_anchor_match_fail_reason", rejected_reason)
                    setattr(req, "lossy_anchor_context_target_prefix_len", target_prefix_len)
                    logger.info(
                        "[agenttemplatekv] context-aligned reuse requested; staging recompute gap before suffix copy: "
                        "content_sig=%s gap_len=%d max_gap=%d",
                        req_content_signature,
                        gap_len,
                        max_gap,
                    )
                    if copied_anchor_count > 0:
                        return exact_values, exact_node
                    continue
                if gap_len > max_gap:
                    setattr(req, "lossy_rejected_reason", "agenttemplatekv_large_zero_gap")
                    setattr(req, "lossy_anchor_match_gap_len", gap_len)
                    setattr(req, "agenttemplatekv_rejected_large_gap_count", 1)
                    logger.info(
                        "[agenttemplatekv] reject content_sig=%s gap_len=%d max_gap=%d",
                        req_content_signature,
                        gap_len,
                        max_gap,
                    )
                    continue

                total_new = gap_len + copy_len

                new_slots = self.token_to_kv_pool_allocator.alloc(total_new)
                if new_slots is None:
                    logger.warning("[anchor_kv] alloc failed for content_sig=%s", req_content_signature)
                    continue

                kvcache = self.token_to_kv_pool_allocator.get_kvcache()

                if gap_len > 0:
                    gap_slots = new_slots[:gap_len]
                    for layer_id in range(kvcache.layer_num):
                        kvcache.get_key_buffer(layer_id)[gap_slots] = 0
                        kvcache.get_value_buffer(layer_id)[gap_slots] = 0

                # Copy anchor KV
                src_kv = entry.kv_indices[suffix_start : suffix_start + copy_len]
                dst_kv = new_slots[gap_len : gap_len + copy_len]
                move_kv_cache_native(kvcache.k_buffer, kvcache.v_buffer, dst_kv, src_kv)
                with self.anchor_kv_store_lock:
                    entry.ref_count += 1
                    # Track so cache_finished_req can decrement on natural
                    # request finish (prevents anchor_kv_store leak across
                    # multi-case runs).  setattr keeps the change off the
                    # public Req API; unit tests build SimpleNamespace reqs
                    # and do not assert on this attribute.
                    if req is not None:
                        _consumed = getattr(req, "_consumed_anchor_entries", None)
                        if _consumed is None:
                            setattr(req, "_consumed_anchor_entries", [entry])
                        else:
                            _consumed.append(entry)

                # Apply RoPE delta rotation: key positions must match the new
                # absolute positions in this request. Delta = new_pos - old_pos.
                # Since RoPE is additive in 2D, R(new) = R(delta) * R(old).
                delta = (exact_len + gap_len) - (entry.start_pos + suffix_start)
                if delta != 0 and self.rope_rotary_dim > 0:
                    delta_tensor = torch.full(
                        (copy_len,), delta, dtype=torch.long, device=dst_kv.device
                    )
                    self._apply_rope_delta_to_keys(kvcache.k_buffer, dst_kv, delta_tensor)
                    logger.info(
                        "[anchor_kv] RoPE delta=%d for sig=%s (old_pos=%d, new_pos=%d)",
                        delta, req_content_signature,
                        entry.start_pos + suffix_start, exact_len + gap_len,
                    )

                extended = exact_values + [new_slots]
                copied_anchor_count += 1
                total_copy_len += copy_len
                total_planned_copy_len += planned_copy_len
                total_gap_len += gap_len
                total_gap_recompute_len += (
                    int(getattr(req, "lossy_anchor_gap_recompute_len", 0) or 0)
                    if staged_copy_ready
                    else 0
                )
                any_copy_truncated = any_copy_truncated or copy_len < planned_copy_len
                # Track per-entry semantic suffix telemetry (sum across multi-anchor).
                setattr(
                    req,
                    "lossy_anchor_suffix_copy_semantic_len",
                    int(getattr(req, "lossy_anchor_suffix_copy_semantic_len", 0) or 0)
                    + int(semantic_copy_len),
                )
                if semantic_min_cos > 0:
                    prev = float(getattr(req, "lossy_anchor_suffix_copy_semantic_min_cosine", 1.0) or 1.0)
                    setattr(
                        req,
                        "lossy_anchor_suffix_copy_semantic_min_cosine",
                        min(prev, semantic_min_cos),
                    )
                if semantic_truncated:
                    setattr(req, "lossy_anchor_suffix_copy_semantic_truncated", True)
                setattr(req, "lossy_anchor_match_used", True)
                setattr(req, "lossy_anchor_match_len", total_copy_len)
                setattr(req, "lossy_anchor_match_gap_len", total_gap_len)
                staged_recompute_len = int(
                    getattr(req, "lossy_anchor_gap_recompute_len", 0) or 0
                )
                setattr(req, "lossy_anchor_gap_recompute_len", total_gap_recompute_len)
                setattr(req, "lossy_anchor_suffix_copy_len", total_copy_len)
                setattr(req, "lossy_anchor_suffix_copy_planned_len", total_planned_copy_len)
                setattr(req, "lossy_anchor_suffix_copy_cap_len", min(copy_caps) if copy_caps else 0)
                setattr(req, "lossy_anchor_suffix_copy_truncated", any_copy_truncated)
                setattr(
                    req,
                    "lossy_anchor_suffix_recompute_head_len",
                    max(0, suffix_start),
                )
                setattr(req, "lossy_anchor_multi_copy_count", copied_anchor_count)
                setattr(req, "lossy_anchor_context_copy_ready", staged_copy_ready)
                setattr(req, "lossy_anchor_context_aligned", gap_len == 0 or staged_copy_ready)
                setattr(req, "lossy_anchor_context_align_fail_reason", None)
                setattr(req, "lossy_anchor_match_fail_reason", None)
                setattr(req, "lossy_anchor_match_signature", entry.signature)
                setattr(req, "lossy_anchor_match_content_signature", req_content_signature)
                setattr(req, "lossy_anchor_rope_delta", delta)
                setattr(
                    req,
                    "codebase_prefetch_device_hit_count",
                    getattr(req, "codebase_prefetch_device_hit_count", 0) + 1,
                )
                setattr(
                    req,
                    "agenttemplatekv_prefetch_hit_count",
                    getattr(req, "agenttemplatekv_prefetch_hit_count", 0) + 1,
                )
                if entry.prefetch_lock_held:
                    entry.prefetch_hit_count += 1
                    setattr(
                        req,
                        "agenttemplatekv_prefetch_consumed_count",
                        getattr(req, "agenttemplatekv_prefetch_consumed_count", 0) + 1,
                    )
                    entry.prefetch_steps_remaining -= 1
                    if entry.prefetch_steps_remaining <= 0:
                        self._agenttemplatekv_release_entry(entry)
                elif getattr(req, "agenttemplatekv_prefetch_protected_tokens", 0) > 0:
                    setattr(
                        req,
                        "agenttemplatekv_prefetch_consumed_count",
                        getattr(req, "agenttemplatekv_prefetch_consumed_count", 0) + 1,
                    )
                exact_values = extended
                exact_len += total_new
                if not multi_anchor_enabled:
                    return extended, exact_node

        setattr(req, "lossy_anchor_token_mismatch_count", token_mismatch_count)
        setattr(req, "lossy_anchor_span_shape_mismatch_count", span_shape_mismatch_count)
        setattr(req, "lossy_anchor_prefix_covers_count", prefix_covers_count)
        if copied_anchor_count > 0:
            setattr(req, "lossy_anchor_match_fail_reason", None)
            return exact_values, exact_node
        if not getattr(req, "lossy_anchor_match_fail_reason", None):
            if prefix_covers_count:
                setattr(req, "lossy_anchor_match_fail_reason", "prefix_already_covers_anchor")
            elif token_mismatch_count:
                setattr(req, "lossy_anchor_match_fail_reason", "token_mismatch")
            elif span_shape_mismatch_count:
                setattr(req, "lossy_anchor_match_fail_reason", "span_shape_mismatch")
            else:
                setattr(req, "lossy_anchor_match_fail_reason", "no_usable_anchor_entry")
        return exact_values, exact_node

    # -----------------------------------------------------------------
    # Per-placeholder k-NN read path (Duke 2026 KVCOMM-style).
    #
    # `_try_placeholder_knn_lossy_match` runs *after* `_try_lossy_fuzzy_match`
    # in `match_prefix` (when SGLANG_PLACEHOLDER_KNN_MATCH=1).  For each
    # slot in `req.placeholder_anchor_token_spans` it (a) embeds the slot's
    # text, (b) does per-slot embedding k-NN search, (c) copies the best
    # neighbor's KV into the current prefill stream with a RoPE delta
    # rotation.  v1 takes only the single-best neighbor; soft-weighted
    # K-nearest reconstruction is Phase 2.
    # -----------------------------------------------------------------

    def _try_placeholder_knn_lossy_match(
        self,
        req: Req,
        key: RadixKey,
        exact_values: List[torch.Tensor],
        exact_node: TreeNode,
    ) -> Tuple[List[torch.Tensor], TreeNode]:
        """Per-placeholder embedding k-NN reuse.  Returns updated
        ``(exact_values, exact_node)`` — caller treats the returned node
        as the new radix tail.  Disabled by default; flips on via
        ``SGLANG_PLACEHOLDER_KNN_MATCH=1``.

        Telemetry (set on req):
          - placeholder_kv_prefill_matched_slots
          - placeholder_kv_prefill_skipped_tokens
          - placeholder_knn_topk_similarity_mean
          - placeholder_anchor_pool_hit_count
          - placeholder_anchor_pool_miss_count
        """
        if os.environ.get("SGLANG_PLACEHOLDER_KNN_MATCH", "0") != "1":
            return exact_values, exact_node
        spans = getattr(req, "placeholder_anchor_token_spans", None) or []
        if not spans:
            return exact_values, exact_node
        try:
            from sglang.srt.mem_cache.semantic_suffix import (
                embed_single_text,
                is_enabled,
                load_embedder,
            )
        except Exception as ie:  # pragma: no cover - defensive
            logger.debug("[placeholder_knn] import failed: %s", ie)
            return exact_values, exact_node
        if not is_enabled():
            return exact_values, exact_node
        emb = load_embedder()
        if emb is None:
            return exact_values, exact_node
        top_k = int(os.environ.get("SGLANG_PLACEHOLDER_KNN_TOPK", "4"))
        min_cos = float(
            os.environ.get("SGLANG_PLACEHOLDER_KNN_MIN_COSINE", "0.70")
        )
        max_slot_len = int(
            os.environ.get("SGLANG_PLACEHOLDER_KNN_MAX_SLOT_LEN", "4096")
        )
        # Cost-aware abort guard: when entry_len × layer_num exceeds this
        # threshold, the slot's RoPE delta rotation would cost more GPU
        # time than the dense prefill it would skip.  Default 114688
        # (28 layers × 4096 tokens) targets Qwen2.5-7B's breakeven.
        # Set to 0 to disable the guard (matches v10c convention).
        max_rope_ops = int(
            os.environ.get("SGLANG_PLACEHOLDER_KNN_MAX_ROPE_OPS", "114688")
        )
        # Head-only RoPE rotation (Phase 2.1, EPIC-inspired).  Only the
        # first `head_tokens` of each placeholder slot get rotated to the
        # correct global position; the rest keep their original (chunk-
        # local position-0) RoPE.  k=2 is EPIC's recommended default with
        # <=7% accuracy loss on standard benchmarks.  Set to 0 to disable
        # (full rotation, v12 back-compat).  For entry_len <= head_tokens,
        # behaves identically to full rotation (no-op slice wrapper).
        head_tokens = int(
            os.environ.get("SGLANG_PLACEHOLDER_KNN_HEAD_TOKENS", "2")
        )
        # Phase 2.5: skip copy when most of the slot is already in the
        # prefix cache.  For high overlap ratios the alloc + move_kv +
        # RoPE overhead of the trimmed copy exceeds the prefill saving,
        # so we let dense prefill handle the few new tokens.  Default 0.5
        # (skip when >50% of slot is already cached).  Set to 1.0 to
        # disable.
        max_overlap_ratio = float(
            os.environ.get(
                "SGLANG_PLACEHOLDER_KNN_MAX_OVERLAP_RATIO", "0.5"
            )
        )
        # Phase 2.6: cost-vs-prefill gate (CacheBlend HKVD-style).  Skip
        # the copy when alloc + move_kv + RoPE cost exceeds the prefill
        # saving × margin.  Calibrated defaults from v16 per-agent cost
        # analysis (see MULTI_AGENT_PLACEHOLDER_V16_TRIM_RESULTS.md):
        #   - 20-30 ms per copy call (CPU dispatch + cuda sync dominates
        #     for small copy_lens)
        #   - ~4 μs/token move_kv_cache tiled kernel (28 layers)
        #   - ~40 μs/token prefill on Qwen2.5-7B / RTX 4090
        #   - ~2 μs/layer head-only RoPE for 2 tokens
        # Set COPY_COST_GUARD_ENABLED=0 to disable and let the legacy
        # max_rope_ops safety net alone.
        cost_guard_enabled = (
            os.environ.get(
                "SGLANG_PLACEHOLDER_KNN_COPY_COST_GUARD_ENABLED", "1"
            )
            == "1"
        )
        copy_skip_margin = float(
            os.environ.get(
                "SGLANG_PLACEHOLDER_KNN_COPY_SKIP_MARGIN", "1.0"
            )
        )
        copy_launch_overhead_us = float(
            os.environ.get(
                "SGLANG_PLACEHOLDER_KNN_COPY_LAUNCH_OVERHEAD_US", "20000"
            )
        )
        copy_move_per_token_us = float(
            os.environ.get(
                "SGLANG_PLACEHOLDER_KNN_COPY_MOVE_PER_TOKEN_US", "4"
            )
        )
        copy_prefill_per_token_us = float(
            os.environ.get(
                "SGLANG_PLACEHOLDER_KNN_COPY_PREFILL_PER_TOKEN_US", "40"
            )
        )
        copy_rope_per_layer_us = float(
            os.environ.get(
                "SGLANG_PLACEHOLDER_KNN_COPY_ROPE_PER_LAYER_US", "2"
            )
        )
        # Wrap the rest in try/except so a single bad span cannot crash
        # the request pipeline.  Any exception becomes a "no-op" and
        # sets req.placeholder_anchor_pool_skipped_invalid_count.
        try:
            return self._try_placeholder_knn_lossy_match_body(
                req, exact_values, exact_node, spans, emb,
                top_k, min_cos, max_slot_len, max_rope_ops, head_tokens,
                max_overlap_ratio,
                cost_guard_enabled, copy_skip_margin,
                copy_launch_overhead_us, copy_move_per_token_us,
                copy_prefill_per_token_us, copy_rope_per_layer_us,
            )
        except Exception as ke:  # pragma: no cover - defensive
            logger.warning(
                "[placeholder_knn] unexpected error rid=%s: %s",
                getattr(req, "rid", "?"), ke,
            )
            setattr(req, "placeholder_anchor_pool_skipped_invalid_count",
                    getattr(req, "placeholder_anchor_pool_skipped_invalid_count", 0) + len(spans))
            return exact_values, exact_node

    def _try_placeholder_knn_lossy_match_body(
        self,
        req: Req,
        exact_values: List[torch.Tensor],
        exact_node: TreeNode,
        spans: List[Dict[str, Any]],
        emb,
        top_k: int,
        min_cos: float,
        max_slot_len: int,
        max_rope_ops: int = 0,
        head_tokens: int = 2,  # Phase 2.1: EPIC-inspired head-only RoPE
        max_overlap_ratio: float = 0.5,  # Phase 2.5: skip when overlap_ratio > this
        # Phase 2.6: cost-vs-prefill gate (CacheBlend HKVD-style).
        # Default False in the body signature so existing unit tests
        # don't trip the new gate; production enables via env var
        # SGLANG_PLACEHOLDER_KNN_COPY_COST_GUARD_ENABLED=1 in the wrapper.
        cost_guard_enabled: bool = False,
        copy_skip_margin: float = 1.0,
        copy_launch_overhead_us: float = 20000,
        copy_move_per_token_us: float = 4,
        copy_prefill_per_token_us: float = 40,
        copy_rope_per_layer_us: float = 2,
    ) -> Tuple[List[torch.Tensor], TreeNode]:
        from sglang.srt.mem_cache.semantic_suffix import embed_single_text as _est

        # `prefix_len` is the length that the prefix cache matched BEFORE
        # any k-NN copy in this body call.  We use this for the
        # `start < prefix_len` guard (not the running `exact_len` which
        # also includes prior k-NN-copied slots).  With multiple slots in
        # a single request, the second slot's start can be < the running
        # `exact_len` (which now includes the first slot's k-NN copy)
        # without indicating a real conflict — the slot's content is
        # genuinely new; it's just AFTER the prefix in absolute token
        # position, not AFTER the running exact_len.
        prefix_len = (
            sum(int(v.numel()) for v in exact_values) if exact_values else 0
        )
        exact_len = prefix_len
        matched_slots = 0
        skipped_tokens = 0
        sims_total: List[float] = []
        consumed_entries: List[AnchorKVEntry] = []
        miss_count = 0
        skipped_invalid = 0
        # The original code accepted `key` to bound k-NN lookups to the
        # prompt length, but the body helper no longer needs it; keep
        # `key_tokens_len = None` so spans are always attempted (they were
        # already filtered by the outer caller).
        key_tokens_len: Optional[int] = None

        for span in spans:
            if not isinstance(span, dict):
                skipped_invalid += 1
                continue
            slot_id = str(span.get("slot_id", "") or "")
            start = int(span.get("start_token", -1))
            end = int(span.get("end_token", -1))
            text = str(span.get("text", "") or "")
            if (
                not slot_id
                or start < 0
                or end <= start
                or (key_tokens_len is not None and start >= key_tokens_len)
            ):
                skipped_invalid += 1
                continue
            query_emb = _est(text or " ", emb=emb)
            if query_emb is None:
                miss_count += 1
                continue

            with self.placeholder_anchor_pool_lock:
                pool = list(self.placeholder_anchor_pool.get(slot_id, []))
            if not pool:
                miss_count += 1
                continue

            neighbors = _placeholder_knn_search(
                pool, query_emb, top_k=top_k, min_similarity=min_cos,
            )
            if not neighbors:
                miss_count += 1
                continue

            best, best_sim = neighbors[0]
            sims_total.append(best_sim)
            entry_len = min(len(best.token_ids), end - start, max_slot_len)
            if entry_len <= 0:
                miss_count += 1
                continue

            # Phase 2.4: trim the k-NN copy to only the post-prefix
            # portion of the slot. When start < prefix_len, the prefix
            # cache has [start, prefix_len) of the slot already; we
            # only copy [prefix_len, end). When start >= prefix_len,
            # copy_len == entry_len (no trim).
            #
            # The KV layout stays contiguous: prefix indices cover
            # [0, prefix_len), and the trimmed new_slots cover
            # [prefix_len, prefix_len + copy_len). The flashinfer
            # attention backend handles this correctly.
            #
            # The two extremes:
            #  - start < prefix_len: copy_len = entry_len - overlap_len
            #  - start >= prefix_len: copy_len = entry_len, copy_offset=0
            overlap_len = max(0, prefix_len - start)
            copy_offset = overlap_len
            copy_len = entry_len - overlap_len
            # Phase 2.5: skip copy when most of the slot is already in
            # the prefix cache.  The cost of alloc + move_kv_cache + head
            # RoPE for a small (entry_len - overlap_len) trim is on par
            # with the prefill saving for that many tokens (~20-30 ms in
            # practice, dominated by per-launch overhead).  When the
            # overlap ratio is high, the right move is to let dense
            # prefill handle the few new tokens rather than pay the copy
            # overhead.  Set max_overlap_ratio=1.0 to disable.
            if (
                max_overlap_ratio < 1.0
                and entry_len > 0
                and overlap_len / entry_len > max_overlap_ratio
            ):
                setattr(
                    req,
                    "placeholder_knn_skipped_high_overlap_count",
                    getattr(
                        req,
                        "placeholder_knn_skipped_high_overlap_count",
                        0,
                    ) + 1,
                )
                logger.info(
                    "[placeholder_knn] rid=%s slot=%s skip-high-overlap: "
                    "overlap_ratio=%.3f > %.3f (overlap=%d entry=%d)",
                    getattr(req, "rid", "?"), slot_id,
                    overlap_len / entry_len, max_overlap_ratio,
                    overlap_len, entry_len,
                )
                continue
            # Phase 2.6: cost-vs-prefill gate (CacheBlend HKVD-style).
            # The empirical per-copy cost is dominated by CPU dispatch +
            # cuda sync (~20ms); the prefill saving scales with
            # copy_len.  Below the crossover (small copy_len), the copy
            # is net-negative; dense prefill handles the few new tokens
            # cheaper.  Cost model is parameterized via env vars and
            # can be calibrated per hardware/model.
            if cost_guard_enabled and copy_len > 0:
                try:
                    layer_num = (
                        self.token_to_kv_pool_allocator
                        .get_kvcache().layer_num
                    )
                except Exception:  # pragma: no cover - defensive
                    layer_num = 0
                eff_head_tokens = (
                    copy_len
                    if head_tokens <= 0
                    else min(head_tokens, copy_len)
                )
                rope_cost_us = (
                    eff_head_tokens * layer_num * copy_rope_per_layer_us
                    if layer_num > 0
                    else 0
                )
                copy_cost_us = (
                    copy_launch_overhead_us
                    + copy_len * copy_move_per_token_us
                    + rope_cost_us
                )
                prefill_saving_us = copy_len * copy_prefill_per_token_us
                if copy_cost_us > prefill_saving_us * copy_skip_margin:
                    setattr(
                        req,
                        "placeholder_anchor_pool_skipped_cost_count",
                        getattr(
                            req,
                            "placeholder_anchor_pool_skipped_cost_count",
                            0,
                        ) + 1,
                    )
                    logger.info(
                        "[placeholder_knn] rid=%s slot=%s skip-cost: "
                        "copy_cost=%.1fus > prefill_saving=%.1fus × "
                        "margin=%.2f (copy_len=%d head=%d layers=%d)",
                        getattr(req, "rid", "?"), slot_id,
                        copy_cost_us, prefill_saving_us, copy_skip_margin,
                        copy_len, eff_head_tokens, layer_num,
                    )
                    continue
            if copy_len <= 0:
                # Slot entirely within prefix region, or anchor shorter
                # than overlap. Nothing useful to copy; fall back to the
                # skip path (no behavior change from v15 here).
                skipped_invalid += 1
                continue

            # Cost-aware abort guard (Phase 2).  In Phase 2.1 the cost is
            # based on head-only rotation (head_len x layer_num) not
            # full-slot rotation.  With head_tokens=2 and layer_num=28,
            # cost = 56 << 114688 default, so the guard is effectively
            # a safety net (almost never fires).
            if max_rope_ops > 0:
                try:
                    layer_num = (
                        self.token_to_kv_pool_allocator
                        .get_kvcache().layer_num
                    )
                except Exception:  # pragma: no cover - defensive
                    layer_num = 0
                eff_head_len = (
                    copy_len
                    if head_tokens <= 0
                    else min(head_tokens, copy_len)
                )
                cost = eff_head_len * layer_num if layer_num > 0 else 0
                if cost > max_rope_ops:
                    logger.info(
                        "[placeholder_knn] rid=%s slot=%s abort: "
                        "entry_len=%d head_len=%d layer_num=%d cost=%d "
                        "> threshold=%d (cosine=%.3f)",
                        getattr(req, "rid", "?"), slot_id, entry_len,
                        eff_head_len, layer_num, cost, max_rope_ops,
                        best_sim,
                    )
                    setattr(
                        req,
                        "placeholder_anchor_pool_skipped_cost_count",
                        getattr(
                            req,
                            "placeholder_anchor_pool_skipped_cost_count",
                            0,
                        ) + 1,
                    )
                    continue

            new_slots = self.token_to_kv_pool_allocator.alloc(copy_len)
            if new_slots is None:
                logger.warning(
                    "[placeholder_knn] alloc failed for slot=%s len=%d",
                    slot_id, copy_len,
                )
                miss_count += 1
                continue

            try:
                kvcache = self.token_to_kv_pool_allocator.get_kvcache()
            except Exception as ae:  # pragma: no cover - defensive
                logger.warning(
                    "[placeholder_knn] get_kvcache failed: %s", ae,
                )
                continue

            src_kv = best.kv_indices[copy_offset : copy_offset + copy_len]
            dst_kv = new_slots[:copy_len]
            # Phase 2.2: route through MHATokenToKVPool.move_kv_cache
            # dispatcher so we get the triton-tiled kernel at default
            # SGLANG_NATIVE_MOVE_KV_CACHE=False (envs.SGLANG_NATIVE_MOVE_KV_CACHE
            # = EnvBool(False) at environ.py:213). Falls back to
            # move_kv_cache_native when env=1 or when kvcache lacks the
            # dispatcher (legacy stubs / tests).
            copy_method = (
                "native"
                if os.environ.get("SGLANG_NATIVE_MOVE_KV_CACHE", "0") == "1"
                else "tiled"
            )
            try:
                kvcache.move_kv_cache(dst_kv, src_kv)
            except AttributeError:
                # Pre-Phase-2.2 test stubs may expose only
                # k_buffer/v_buffer.  Re-route to the eager native loop
                # so behavior is identical to v13.
                move_kv_cache_native(
                    kvcache.k_buffer, kvcache.v_buffer, dst_kv, src_kv,
                )
                copy_method = "native"
            except Exception as me:  # pragma: no cover - defensive
                logger.warning(
                    "[placeholder_knn] move_kv_cache failed (method=%s): %s",
                    copy_method, me,
                )
                setattr(
                    req,
                    "placeholder_anchor_pool_copy_error_count",
                    getattr(
                        req,
                        "placeholder_anchor_pool_copy_error_count",
                        0,
                    ) + 1,
                )
                continue
            # Record which path served this slot, for end-to-end
            # diagnostics.
            setattr(req, "placeholder_knn_copy_method", copy_method)

            # Apply RoPE delta: anchor was stored at best.start_pos; this
            # request's slot starts at `start`.  delta = new_pos - old_pos.
            # Phase 2.1: head-only rotation (EPIC-inspired).  Only the
            # first `head_tokens` tokens are rotated; the rest retain
            # chunk-local position-0 RoPE.  Cost: head_tokens x layer_num
            # vs entry_len x layer_num (~1000x cheaper for typical slots).
            #
            # Phase 2.4: the first token of the trimmed `dst_kv` lives at
            # global position `start + copy_offset = prefix_len`. Its
            # corresponding anchor position is `best.start_pos + copy_offset`.
            # So delta = (start + copy_offset) - (best.start_pos + copy_offset)
            # which algebraically simplifies to start - best.start_pos.
            # When copy_offset=0 (no trim), the formula is byte-identical
            # to v15.
            delta = (start + copy_offset) - (best.start_pos + copy_offset)
            rotated = 0
            if delta != 0 and getattr(self, "rope_rotary_dim", 0) > 0:
                try:
                    if head_tokens <= 0:
                        # Explicitly disabled: full rotation (v12 back-compat).
                        delta_tensor = torch.full(
                            (copy_len,), delta, dtype=torch.long,
                            device=dst_kv.device,
                        )
                        self._apply_rope_delta_to_keys(
                            kvcache.k_buffer, dst_kv, delta_tensor,
                        )
                        rotated = copy_len
                    else:
                        rotated = self._apply_rope_delta_to_head(
                            kvcache.k_buffer, dst_kv, head_tokens, delta,
                        )
                except Exception as re_:  # pragma: no cover - defensive
                    logger.debug(
                        "[placeholder_knn] rope delta skipped: %s", re_,
                    )
            setattr(req, "placeholder_knn_head_rotation_tokens", rotated)
            setattr(
                req, "placeholder_knn_head_rotation_total_ops",
                rotated * len(kvcache.k_buffer) if rotated else 0,
            )

            with self.placeholder_anchor_pool_lock:
                best.ref_count += 1
                best.last_access_time = time.monotonic()
            consumed_entries.append(best)

            exact_values = exact_values + [new_slots]
            exact_len += copy_len
            matched_slots += 1
            skipped_tokens += copy_len
            # Phase 2.4: track the cumulative overlap (prefix_len -
            # start) for diagnostics.  Zero when no trim.
            setattr(
                req,
                "placeholder_kv_prefill_overlap_tokens",
                getattr(
                    req,
                    "placeholder_kv_prefill_overlap_tokens",
                    0,
                ) + overlap_len,
            )

        setattr(req, "placeholder_kv_prefill_matched_slots", matched_slots)
        setattr(req, "placeholder_kv_prefill_skipped_tokens", skipped_tokens)
        setattr(req, "placeholder_anchor_pool_hit_count", matched_slots)
        setattr(
            req, "placeholder_anchor_pool_miss_count",
            getattr(req, "placeholder_anchor_pool_miss_count", 0) + miss_count,
        )
        setattr(req, "placeholder_anchor_pool_skipped_invalid_count", skipped_invalid)
        if sims_total:
            setattr(
                req, "placeholder_knn_topk_similarity_mean",
                float(sum(sims_total) / len(sims_total)),
            )
        if consumed_entries:
            existing = getattr(req, "_consumed_placeholder_entries", None)
            if existing is None:
                setattr(req, "_consumed_placeholder_entries", consumed_entries)
            else:
                existing.extend(consumed_entries)
        if matched_slots:
            logger.info(
                "[placeholder_knn] rid=%s matched=%d skipped_tokens=%d "
                "miss=%d invalid=%d sim_mean=%.3f",
                getattr(req, "rid", "?"), matched_slots, skipped_tokens,
                miss_count, skipped_invalid,
                float(sims_total[0]) if sims_total else 0.0,
            )
        return exact_values, exact_node

    def cache_finished_req(self, req: Req, is_insert: bool = True,
                           tokenizer=None):
        """Cache request when it finishes.

        Args:
            req: the request object whose KV is being cached.
            is_insert: whether to insert into the radix tree.
            tokenizer: optional LLM tokenizer passed through from the
                scheduler.  Used by placeholder k-NN write-back for F1
                validation of dense-prefilled text.  When None, falls
                back to getattr(req, "tokenizer", None) (also typically
                None in production); the write-back path treats missing
                tokenizer as "cannot compute F1" and skips the F1 guard
                (matches v10c behavior).
        """
        # Stash the tokenizer on the cache instance so subsequent
        # `_decode_placeholder_span` and `_store_placeholder_anchor_kv`
        # calls can find it without prop drilling through every method.
        if tokenizer is not None:
            self.tokenizer = tokenizer
        # AgentTemplateKV hooks (must run even if insertion is disabled).
        # 1) Make TTL release reachable: warmup requests carry no hints,
        #    so locks from prior cases' planner+mode4 would otherwise never
        #    expire.  Trigger the sweep here on every request finish.
        try:
            self._agenttemplatekv_release_expired_prefetch_entries(req)
        except Exception as _e:
            logger.warning("release_expired_prefetch_entries failed: %s", _e)
        # 2) Decrement ref_count for entries this request consumed via
        #    lossy reuse.  Without this, ref_count only ever goes up and
        #    anchor_kv_store grows unboundedly across cases.
        try:
            consumed = getattr(req, "_consumed_anchor_entries", None)
            if consumed:
                self._decrement_consumed_anchor_refs(consumed)
        except Exception as _e:
            logger.warning("decrement_consumed_anchor_refs failed: %s", _e)
        # 3) Optional per-case state dump for debugging the protected-anchor
        #    accumulation.  Enable with SGLANG_DBGCASE=1.
        if os.environ.get("SGLANG_DBGCASE") == "1":
            try:
                with self.anchor_kv_store_lock:
                    _total = sum(len(v) for v in self.anchor_kv_store.values())
                    _held = sum(
                        1
                        for v in self.anchor_kv_store.values()
                        for e in v
                        if e.prefetch_lock_held
                    )
                logger.info(
                    "[dbgcase] rid=%s protected=%d evictable=%d held=%d total=%d",
                    getattr(req, "rid", "?"),
                    self.protected_size_,
                    self.evictable_size_,
                    _held,
                    _total,
                )
            except Exception as _e:
                logger.warning("dbgcase log failed: %s", _e)

        # 4) Clear the consumed-entries list on the request so it doesn't
        # leak across request reuse (the req object may be recycled by the
        # paged scheduler). Drop the attribute entirely rather than leaving
        # an empty list, to keep the SimpleNamespace tests' hasattr path
        # explicit.
        if hasattr(req, "_consumed_anchor_entries"):
            try:
                delattr(req, "_consumed_anchor_entries")
            except Exception as _e:
                logger.warning("del req._consumed_anchor_entries failed: %s", _e)

        # In deterministic mode, disable finished request insertion to radix cache
        if self.disable_finished_insert:
            is_insert = False

        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Maybe convert to bigram keys for EAGLE
        keys = convert_to_bigram_key(token_ids) if self.is_eagle else token_ids
        keys = page_align_keys(keys, self.page_size)
        values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        # Radix Cache takes one ref in memory pool
        # Extract workflow_id from request ID if available (e.g., rid="wf0-step2-abc123").
        # This enables cross-workflow prefix sharing: nodes accessed by multiple workflows
        # get boosted priority in PriorityStrategy, making shared prefixes sticky.
        workflow_id: Optional[int] = None
        if hasattr(req, "rid") and req.rid:
            m = re.match(r"wf(\d+)", req.rid)
            if m:
                workflow_id = int(m.group(1))

        if is_insert:
            priority = getattr(req, "priority", 0) or 0
            role_type = getattr(req, "role_type", 0) or 0
            convergence_factor = getattr(req, "convergence_factor", 0) or 0
            critical_path_distance = getattr(req, "critical_path_distance", 1) or 1
            code_anchor_signature = getattr(req, "code_anchor_signature", "") or ""
            code_content_signature = getattr(req, "code_content_signature", "") or ""
            code_anchor_spans = getattr(req, "code_anchor_spans", None) or []
            reuse_mode = getattr(req, "reuse_mode", "") or ""
            lossy_alignment_method = getattr(req, "lossy_alignment_method", "") or ""
            template_task_family = getattr(req, "template_task_family", "") or ""
            template_workflow_signature = getattr(req, "template_workflow_signature", "") or ""
            template_structural_fingerprint = getattr(req, "template_structural_fingerprint", "") or ""
            lossy_match = (
                self._resolve_lossy_match(req)[1]
                if reuse_mode == "lossy"
                else AnchorMatchResult(
                    reuse_allowed=False,
                    reuse_confidence=0.0,
                    syntax_region_type=self._default_syntax_region_type(req),
                )
            )
            result = self.insert(
                InsertParams(key=radix_key, value=values, priority=priority,
                             role_type=role_type, convergence_factor=convergence_factor,
                             critical_path_distance=critical_path_distance,
                             anchor_type=lossy_alignment_method,
                             anchor_id=code_anchor_signature,
                             code_content_signature=code_content_signature,
                             anchor_spans=code_anchor_spans,
                             reuse_mode=reuse_mode,
                             reuse_confidence=lossy_match.reuse_confidence if lossy_match.reuse_allowed else 0.0,
                             syntax_region_type=lossy_match.syntax_region_type or self._default_syntax_region_type(req),
                             template_task_family=template_task_family,
                             template_workflow_signature=template_workflow_signature,
                             template_structural_fingerprint=template_structural_fingerprint),
                workflow_id=workflow_id,
            )
            new_prefix_len = result.prefix_len
            # Free the duplicates that were already in the tree
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : new_prefix_len]
            )
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : len(keys)]
            )

        # free the unaligned tail
        self.token_to_kv_pool_allocator.free(kv_indices[len(keys) :])

        # Store anchor KV for position-aligned non-prefix reuse. Re-match the
        # just-inserted full key so AgentTemplateKV can pin the actual radix
        # node behind future-use codebase anchors.
        source_node = None
        if is_insert:
            source_node = self.match_prefix(
                MatchPrefixParams(key=radix_key)
            ).last_device_node
        self._store_anchor_kv(req, kv_indices, source_node=source_node)
        # PR2 placeholder k-NN: write per-placeholder anchor blocks to the
        # per-slot pool. No-op when SGLANG_PLACEHOLDER_STORE_ENABLED=0 or
        # when the request did not declare placeholder_anchor_token_spans.
        self._store_placeholder_anchor_kv(req, kv_indices, source_node=source_node)

        # Remove req slot release the cache lock
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Maybe convert to bigram keys for EAGLE
        keys = convert_to_bigram_key(token_ids) if self.is_eagle else token_ids
        keys = page_align_keys(keys, self.page_size)
        values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        # Radix Cache takes one ref in memory pool
        code_anchor_spans = getattr(req, "code_anchor_spans", None) or []
        reuse_mode = getattr(req, "reuse_mode", "") or ""
        lossy_match = (
            self._resolve_lossy_match(req)[1]
            if reuse_mode == "lossy"
            else AnchorMatchResult(
                reuse_allowed=False,
                reuse_confidence=0.0,
                syntax_region_type=self._default_syntax_region_type(req),
            )
        )
        result = self.insert(
            InsertParams(
                key=radix_key,
                value=values,
                chunked=chunked,
                priority=getattr(req, "priority", 0) or 0,
                anchor_type=getattr(req, "lossy_alignment_method", "") or "",
                anchor_id=getattr(req, "code_anchor_signature", "") or "",
                code_content_signature=getattr(req, "code_content_signature", "") or "",
                anchor_spans=code_anchor_spans,
                reuse_mode=reuse_mode,
                reuse_confidence=lossy_match.reuse_confidence if lossy_match.reuse_allowed else 0.0,
                syntax_region_type=lossy_match.syntax_region_type or self._default_syntax_region_type(req),
                template_task_family=getattr(req, "template_task_family", "") or "",
                template_workflow_signature=getattr(req, "template_workflow_signature", "") or "",
                template_structural_fingerprint=getattr(req, "template_structural_fingerprint", "") or "",
            )
        )
        new_prefix_len = result.prefix_len

        self.token_to_kv_pool_allocator.free(
            kv_indices[req.cache_protected_len : new_prefix_len]
        )

        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )
        assert len(new_indices) == len(keys), f"{len(new_indices)=}, {len(keys)=}"

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        # The cache_protected_len is not always equal to len(req.prefix_indices)
        # since for page_size > 1, the partial part is added to req.prefix_indices, but that part of kv indices is not added to the tree.
        # It should be freed in the next cache_unfinished_req and final cache_finished_req to avoid memory leak.
        # So we introduce this `cache_protected_len` field to make sure the partial part can be freed correctly.
        req.cache_protected_len = len(new_indices)

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        # - page_size != 1: there is a partial page at the end, keep the full kv_indices
        # - eagle case: bigram keys will only cache len - 1 kv indices
        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices

        req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper()

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        # When force=True, bypass the lock_ref check: walk the entire
        # tree and free leaves regardless of lock state. This recovers
        # from transient lock-pressure OOMs where all large leaves are
        # held by in-flight prefill batches (lock_ref=3) and the normal
        # `evictable_leaves` set is empty. Gated by
        # SGLANG_RADIX_FORCE_EVICT=1 in common.py.
        if getattr(params, "force", False):
            return self._force_evict_locked(num_tokens, start_time)
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            if x.value is None:
                self.evictable_leaves.discard(x)
                continue

            self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)
            # Hold the lock across _delete_leaf so a concurrent
            # protect_entry cannot race with the ref_count GC inside
            # _decrement_anchor_refs.  RLock supports re-entry, so the
            # inner `with self.anchor_kv_store_lock:` in
            # _decrement_anchor_refs is harmless.
            with self.anchor_kv_store_lock:
                self._delete_leaf(x)

            if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

            self._record_remove_event(x)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def _force_evict_locked(self, num_tokens: int, start_time: float) -> EvictResult:
        """Free leaves regardless of lock_ref. Used to recover from
        transient lock-pressure OOMs.

        Walks the entire radix tree, gathers all leaf nodes with
        live KV data (including those with lock_ref > 0), and
        frees them in LRU order until ``num_tokens`` is satisfied.
        Each freed leaf has its ``value`` cleared so the
        ``evicted`` ``@property`` returns True; this protects
        against the in-flight request's later ``dec_lock_ref``
        re-adding a dead node to ``evictable_leaves`` (because
        ``_update_leaf_status`` short-circuits on
        ``evicted == True``).

        Trade-off: an in-flight request whose KV cache we evict
        will recompute (or fail) on its next decode step. In the
        prefill-dominated pass@1 workload, the in-flight request
        is the one that's about to allocate the 8K space, so
        force-evicting the previous case's leaves is safe: the
        previous case has already completed prefill and either
        finished or has only its decode output to re-derive.
        """
        # DFS to gather all leaves with KV data. We cannot use
        # ``cur.evicted`` (a property returning ``value is None``)
        # as a skip signal: the root_node also has ``value is None``
        # and is therefore "evicted" by that property, which would
        # cause us to skip the root and miss all live leaves
        # beneath it. Instead, identify a real leaf as a node that
        # has a non-None value (a live KV cache).
        all_leaves: list = []
        stack = [self.root_node]
        while stack:
            cur = stack.pop()
            if cur.value is not None:
                all_leaves.append(cur)
            if len(cur.children) > 0:
                for child in cur.children.values():
                    stack.append(child)
        # Sort by eviction priority (LRU: oldest first)
        all_leaves.sort(key=self.eviction_strategy.get_priority)

        num_evicted = 0
        for x in all_leaves:
            if num_evicted >= num_tokens:
                break
            if x.value is None:
                continue
            # Save the value reference, then clear the node's value
            # so the ``evicted`` @property returns True and any
            # concurrent _update_leaf_status / dec_lock_ref from the
            # in-flight request sees the dead state.
            val = x.value
            x.value = None
            self.token_to_kv_pool_allocator.free(val)
            num_evicted += len(val)
            with self.anchor_kv_store_lock:
                self._delete_leaf(x)
            self._record_remove_event(x)

        self.update_eviction_metrics(num_evicted, start_time)
        logger.warning(
            "[radix_cache] force_evict_locked freed %d tokens (target %d) from %d leaves; "
            "this recovers from a lock-pressure OOM but the in-flight request that held the "
            "lock will need to re-derive its KV cache",
            num_evicted, num_tokens, len(all_leaves),
        )
        return EvictResult(num_tokens_evicted=num_evicted)

    def inc_lock_ref(
        self, node: TreeNode, max_ancestors: Optional[int] = None
    ) -> IncLockRefResult:
        """Increment ``lock_ref`` along the ancestor chain from ``node``.

        Default (``max_ancestors=None``) walks all the way to root, matching
        the original RadixCache contract. Pass ``max_ancestors=N`` to cap
        the walk at N levels from ``node``; the returned
        ``locked_nodes`` field then enumerates the locked nodes for
        symmetric release via :meth:`dec_lock_ref` with the same
        ``max_ancestors`` value.

        AgentTemplateKV uses ``max_ancestors=2`` so a deep ``source_node``
        (~14k tokens) only locks the leaf + 2 ancestors, not the entire
        prefix path back to root.
        """
        if self.disable:
            return IncLockRefResult(delta=0)
        cap = math.inf if max_ancestors is None else int(max_ancestors)
        delta = 0
        locked: list = []
        steps = 0
        cur = node
        # ``steps <= cap`` so max_ancestors=2 locks leaf + 2 ancestors
        # (3 nodes total). Default cap=math.inf walks to root.
        while cur is not None and cur != self.root_node and steps <= cap:
            if cur.lock_ref == 0:
                self.evictable_size_ -= len(cur.key)
                self.protected_size_ += len(cur.key)
                delta -= len(cur.key)
            cur.lock_ref += 1
            self._update_leaf_status(cur)
            locked.append(cur)
            cur = cur.parent
            steps += 1
        # When capped, return the locked list so the caller can release
        # exactly those nodes. For a full walk (default), the caller
        # already tracks the top-level node and dec_lock_ref will walk the
        # full path itself, so we leave locked_nodes=None to keep
        # IncLockRefResult lightweight in the hot path.
        return IncLockRefResult(
            delta=delta, locked_nodes=locked if math.isfinite(cap) else None
        )

    # ------------------------------------------------------------------
    # AgentTemplateKV device-first protected-anchor helpers
    # ------------------------------------------------------------------

    def _agenttemplatekv_protected_size_cap(self) -> int:
        """Return the protected-anchor size cap in tokens.  A new protect
        that would push ``protected_size_`` above this cap is rejected.

        Default: ``0.5 * SGLANG_MAX_TOTAL_TOKENS`` (32 768 when the pool is
        65 536).  Overridable via ``SGLANG_LOSSY_PROTECTED_FRAC``
        (float) or ``SGLANG_LOSSY_PROTECTED_MAX_TOKENS`` (int).
        Returns 0 to disable the cap.
        """
        override = os.environ.get("SGLANG_LOSSY_PROTECTED_MAX_TOKENS")
        if override is not None:
            try:
                return max(0, int(override))
            except ValueError:
                pass
        frac_override = os.environ.get("SGLANG_LOSSY_PROTECTED_FRAC")
        try:
            frac = (
                float(frac_override)
                if frac_override is not None
                else 0.5
            )
        except ValueError:
            frac = 0.5
        # Prefer the allocator's authoritative pool size over the env var.
        # The env var was the legacy sizing reference but is silently wrong
        # when the user tunes the pool size via the CLI --max-total-tokens
        # without also setting SGLANG_MAX_TOTAL_TOKENS. CacheInitParams is
        # the right source; fall back to the env var for back-compat.
        max_total = 0
        allocator = getattr(self, "token_to_kv_pool_allocator", None)
        if allocator is not None and hasattr(allocator, "size"):
            max_total = int(allocator.size)
        if max_total <= 0:
            max_total = int(os.environ.get("SGLANG_MAX_TOTAL_TOKENS", "65536"))
        return max(0, int(frac * max_total))

    def dec_lock_ref(
        self,
        node: TreeNode,
        params: Optional[DecLockRefParams] = None,
        max_ancestors: Optional[int] = None,
    ) -> DecLockRefResult:
        """Decrement ``lock_ref`` along the ancestor chain from ``node``.

        Default (``max_ancestors=None``) walks all the way to root, matching
        the original RadixCache contract. Pass ``max_ancestors=N`` to cap
        the walk at N levels (symmetric with :meth:`inc_lock_ref`); this is
        used by AgentTemplateKV to release the chain locked by a prior
        capped ``inc_lock_ref`` call.
        """
        if self.disable:
            return DecLockRefResult(delta=0)
        cap = math.inf if max_ancestors is None else int(max_ancestors)
        delta = 0
        steps = 0
        cur = node
        # Symmetric with inc_lock_ref: ``steps <= cap``.
        while cur is not None and cur != self.root_node and steps <= cap:
            if cur.lock_ref == 1:
                self.evictable_size_ += len(cur.key)
                self.protected_size_ -= len(cur.key)
                delta += len(cur.key)
            cur.lock_ref -= 1
            self._update_leaf_status(cur)
            if cur.parent is None:
                assert (
                    cur is self.root_node
                ), f"This request holds the node from another tree"
            cur = cur.parent
            steps += 1
        return DecLockRefResult(delta=delta)

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        return self.protected_size_

    def all_values_flatten(self):
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    ##### Internal Helper Functions #####

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        access_time = time.monotonic()
        node.last_access_time = access_time

        child_key = self.get_child_key_fn(key)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = access_time
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # new_node -> child
        # New node inherits child's priority (represents shared prefix)
        new_node = TreeNode(priority=child.priority)
        new_node.hit_count = child.hit_count
        new_node.workflow_refs.update(child.workflow_refs)
        new_node.role_type = child.role_type
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].clone()
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        # Propagate the 8 anchor / context-anchor fields from child to new_node.
        # Without this, the prefix side of a split becomes "anchor-blind" and
        # `select_best_match` in anchor_match.py will not consider it.
        if child.anchor_id:
            new_node.anchor_id = child.anchor_id
            new_node.anchor_type = child.anchor_type
            new_node.code_content_signature = child.code_content_signature
            new_node.anchor_spans = list(child.anchor_spans)
            new_node.nesting_depth = child.nesting_depth
            new_node.prompt_position_offset = child.prompt_position_offset
            new_node.system_prompt_class = child.system_prompt_class
            new_node.surrounding_code_hash = child.surrounding_code_hash

        # Split hash_value if it was already computed, otherwise leave as None
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        return new_node

    def _inc_hit_count(self, node: TreeNode, chunked: bool = False):
        # Skip the hit count update for chunked requests to avoid self-referencing
        # inflation where a chunked request increments hit_count on nodes it created
        # in previous chunks.
        if chunked:
            return
        node.hit_count += 1

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        priority: int = 0,
        chunked: bool = False,
        workflow_id: Optional[int] = None,
        role_type: int = 0,
        convergence_factor: int = 0,
        critical_path_distance: int = 1,
        anchor_type: str = "",
        anchor_id: str = "",
        code_content_signature: str = "",
        anchor_spans: list[dict[str, Any]] | None = None,
        reuse_mode: str = "",
        reuse_confidence: float = 0.0,
        syntax_region_type: str = "",
        template_task_family: str = "",
        template_workflow_signature: str = "",
        # Prompt-context fields (sglang-kvflow context_aware_confidence)
        nesting_depth: int = 0,
        prompt_position_offset: int = 0,
        system_prompt_class: str = "",
        surrounding_code_hash: str = "",
        template_structural_fingerprint: str = "",
    ):
        # Convert None priority to 0
        if priority is None:
            priority = 0
        access_time = time.monotonic()
        node.last_access_time = access_time
        # Update priority along the path (take max to propagate higher priority)
        old_priority = node.priority
        node.priority = max(node.priority, priority)
        # Debug logging for KVFlow priority tracking
        if priority > 0 and node.priority != old_priority:
            logger.debug(
                f"[KVFlow-Priority] node_key={node.key!r:.50}, "
                f"priority={priority}, old={old_priority}, new={node.priority}, "
                f"role_type={role_type}, crit_dist={critical_path_distance}"
            )
        # Cross-workflow sharing: record which workflow touched this node.
        # Nodes used by multiple workflows get priority boost in PriorityStrategy.
        if workflow_id is not None:
            node.workflow_refs.add(workflow_id)
        # Token-type awareness: mark system/role nodes for priority retention boost.
        if role_type > 0 and node.role_type == 0:
            node.role_type = role_type
        # DAG-aware convergence protection: mark convergence nodes for priority retention.
        # Nodes with higher convergence_factor are protected from eviction.
        if convergence_factor > 0 and node.convergence_factor == 0:
            node.convergence_factor = convergence_factor
        # DAG-aware critical path distance: mark nodes on the critical path for protection.
        if critical_path_distance > 1 and node.critical_path_distance == 1:
            node.critical_path_distance = critical_path_distance
        if anchor_type and not node.anchor_type:
            node.anchor_type = anchor_type
        if anchor_id and not node.anchor_id:
            node.anchor_id = anchor_id
        if code_content_signature and not node.code_content_signature:
            node.code_content_signature = code_content_signature
        self._merge_anchor_spans(node, anchor_spans)
        if reuse_mode and not node.reuse_mode:
            node.reuse_mode = reuse_mode
        if reuse_confidence > 0 and node.reuse_confidence == 0.0:
            node.reuse_confidence = reuse_confidence
        if syntax_region_type and not node.syntax_region_type:
            node.syntax_region_type = syntax_region_type
        if template_task_family and not node.template_task_family:
            node.template_task_family = template_task_family
        if template_workflow_signature and not node.template_workflow_signature:
            node.template_workflow_signature = template_workflow_signature
        if template_structural_fingerprint and not node.template_structural_fingerprint:
            node.template_structural_fingerprint = template_structural_fingerprint
        if nesting_depth and not node.nesting_depth:
            node.nesting_depth = nesting_depth
        if prompt_position_offset and not node.prompt_position_offset:
            node.prompt_position_offset = prompt_position_offset
        if system_prompt_class and not node.system_prompt_class:
            node.system_prompt_class = system_prompt_class
        if surrounding_code_hash and not node.surrounding_code_hash:
            node.surrounding_code_hash = surrounding_code_hash
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = access_time
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                new_node.priority = max(new_node.priority, priority)
                self._inc_hit_count(new_node, chunked)
                if workflow_id is not None:
                    new_node.workflow_refs.add(workflow_id)
                if role_type > 0 and new_node.role_type == 0:
                    new_node.role_type = role_type
                if convergence_factor > 0 and new_node.convergence_factor == 0:
                    new_node.convergence_factor = convergence_factor
                if critical_path_distance > 1 and new_node.critical_path_distance == 1:
                    new_node.critical_path_distance = critical_path_distance
                if anchor_type and not new_node.anchor_type:
                    new_node.anchor_type = anchor_type
                if anchor_id and not new_node.anchor_id:
                    new_node.anchor_id = anchor_id
                if code_content_signature and not new_node.code_content_signature:
                    new_node.code_content_signature = code_content_signature
                self._merge_anchor_spans(new_node, anchor_spans)
                if reuse_mode and not new_node.reuse_mode:
                    new_node.reuse_mode = reuse_mode
                if reuse_confidence > 0 and new_node.reuse_confidence == 0.0:
                    new_node.reuse_confidence = reuse_confidence
                if syntax_region_type and not new_node.syntax_region_type:
                    new_node.syntax_region_type = syntax_region_type
                if template_task_family and not new_node.template_task_family:
                    new_node.template_task_family = template_task_family
                if template_workflow_signature and not new_node.template_workflow_signature:
                    new_node.template_workflow_signature = template_workflow_signature
                if template_structural_fingerprint and not new_node.template_structural_fingerprint:
                    new_node.template_structural_fingerprint = template_structural_fingerprint
                if nesting_depth and not new_node.nesting_depth:
                    new_node.nesting_depth = nesting_depth
                if prompt_position_offset and not new_node.prompt_position_offset:
                    new_node.prompt_position_offset = prompt_position_offset
                if system_prompt_class and not new_node.system_prompt_class:
                    new_node.system_prompt_class = system_prompt_class
                if surrounding_code_hash and not new_node.surrounding_code_hash:
                    new_node.surrounding_code_hash = surrounding_code_hash
                node = new_node
            else:
                node.priority = max(node.priority, priority)
                self._inc_hit_count(node, chunked)
                if workflow_id is not None:
                    node.workflow_refs.add(workflow_id)
                if role_type > 0 and node.role_type == 0:
                    node.role_type = role_type
                if convergence_factor > 0 and node.convergence_factor == 0:
                    node.convergence_factor = convergence_factor
                if critical_path_distance > 1 and node.critical_path_distance == 1:
                    node.critical_path_distance = critical_path_distance
                if anchor_type and not node.anchor_type:
                    node.anchor_type = anchor_type
                if anchor_id and not node.anchor_id:
                    node.anchor_id = anchor_id
                if code_content_signature and not node.code_content_signature:
                    node.code_content_signature = code_content_signature
                self._merge_anchor_spans(node, anchor_spans)
                if reuse_mode and not node.reuse_mode:
                    node.reuse_mode = reuse_mode
                if reuse_confidence > 0 and node.reuse_confidence == 0.0:
                    node.reuse_confidence = reuse_confidence
                if syntax_region_type and not node.syntax_region_type:
                    node.syntax_region_type = syntax_region_type
                if template_task_family and not node.template_task_family:
                    node.template_task_family = template_task_family
                if template_workflow_signature and not node.template_workflow_signature:
                    node.template_workflow_signature = template_workflow_signature
                if template_structural_fingerprint and not node.template_structural_fingerprint:
                    node.template_structural_fingerprint = template_structural_fingerprint
                if nesting_depth and not node.nesting_depth:
                    node.nesting_depth = nesting_depth
                if prompt_position_offset and not node.prompt_position_offset:
                    node.prompt_position_offset = prompt_position_offset
                if system_prompt_class and not node.system_prompt_class:
                    node.system_prompt_class = system_prompt_class
                if surrounding_code_hash and not node.surrounding_code_hash:
                    node.surrounding_code_hash = surrounding_code_hash
            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            self._inc_hit_count(new_node, chunked)
            if workflow_id is not None:
                new_node.workflow_refs.add(workflow_id)
            if role_type > 0:
                new_node.role_type = role_type
            if convergence_factor > 0:
                new_node.convergence_factor = convergence_factor
            if critical_path_distance > 0:
                new_node.critical_path_distance = critical_path_distance
            if anchor_type:
                new_node.anchor_type = anchor_type
            if anchor_id:
                new_node.anchor_id = anchor_id
            if code_content_signature:
                new_node.code_content_signature = code_content_signature
            self._merge_anchor_spans(new_node, anchor_spans)
            if reuse_mode:
                new_node.reuse_mode = reuse_mode
            if reuse_confidence > 0:
                new_node.reuse_confidence = reuse_confidence
            if syntax_region_type:
                new_node.syntax_region_type = syntax_region_type
            if template_task_family:
                new_node.template_task_family = template_task_family
            if template_workflow_signature:
                new_node.template_workflow_signature = template_workflow_signature
            if template_structural_fingerprint:
                new_node.template_structural_fingerprint = template_structural_fingerprint
            if nesting_depth:
                new_node.nesting_depth = nesting_depth
            if prompt_position_offset:
                new_node.prompt_position_offset = prompt_position_offset
            if system_prompt_class:
                new_node.system_prompt_class = system_prompt_class
            if surrounding_code_hash:
                new_node.surrounding_code_hash = surrounding_code_hash
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)
            # Hash will be computed lazily during event emission
            self._record_store_event(new_node)
        return total_prefix_length

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key.token_ids[:10],
                f"r={current_node.lock_ref}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

    def _delete_leaf(self, node):
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        # GC: decrement ref_count on every AnchorKVEntry whose original
        # content signature matches this node. The node's KV slots are
        # about to be freed, so any entry that "lived" on them is now
        # potentially stale. We use content_signature as the join key
        # because AnchorKVEntry does not carry a back-pointer to the
        # source TreeNode. Entries that match but still have ref_count > 0
        # after this decrement are kept (they were reused elsewhere and
        # will get their own GC signal later).
        self._decrement_anchor_refs(node)

        self.evictable_size_ -= len(node.key)
        if node in self.evictable_leaves:
            self.evictable_leaves.remove(node)
        self._update_leaf_status(node.parent)

    def _decrement_anchor_refs(self, node: TreeNode) -> None:
        """Decrement ref_count on every AnchorKVEntry whose content signature
        matches the deleted node's. Drop entries that reach 0.

        Without this, `ref_count` only ever goes up (incremented in
        `_try_lossy_fuzzy_match`), causing the anchor store to leak entries
        whose source KV slots are now freed.
        """
        sig = getattr(node, "code_content_signature", "") or ""
        if not sig:
            return
        with self.anchor_kv_store_lock:
            entries = self.anchor_kv_store.get(sig)
            if not entries:
                return
            survivors = []
            for entry in entries:
                if entry.ref_count > 0:
                    entry.ref_count -= 1
                if entry.ref_count > 0:
                    survivors.append(entry)
                else:
                    self._agenttemplatekv_release_entry(entry)
                    logger.info(
                        "[anchor_kv_store] GC drop entry sig=%s content=%s start_pos=%d",
                        entry.signature, entry.code_content_signature, entry.start_pos,
                    )
            if survivors:
                self.anchor_kv_store[sig] = survivors
            else:
                self.anchor_kv_store.pop(sig, None)

    def _decrement_consumed_anchor_refs(self, consumed) -> None:
        """Decrement ref_count on every entry this request consumed via lossy
        reuse. Drop entries that reach 0.

        Called from ``cache_finished_req`` with the
        ``req._consumed_anchor_entries`` list populated by
        ``_try_lossy_fuzzy_match``. This is the natural request-finish path
        (vs. ``_decrement_anchor_refs`` which only fires from leaf eviction).
        """
        if not consumed:
            return
        with self.anchor_kv_store_lock:
            for entry in consumed:
                if entry is None:
                    continue
                if entry.ref_count > 0:
                    entry.ref_count -= 1
                if entry.ref_count <= 0:
                    self._agenttemplatekv_release_entry(entry)
                    sig = entry.code_content_signature
                    if sig in self.anchor_kv_store:
                        survivors = [
                            e for e in self.anchor_kv_store[sig] if e is not entry
                        ]
                        if survivors:
                            self.anchor_kv_store[sig] = survivors
                        else:
                            self.anchor_kv_store.pop(sig, None)

    def _update_leaf_status(self, node: TreeNode):
        if node.evicted or node.lock_ref > 0:
            if node in self.evictable_leaves:
                self.evictable_leaves.remove(node)
            return

        for child in node.children.values():
            if not child.evicted:
                if node in self.evictable_leaves:
                    self.evictable_leaves.remove(node)
                return

        if node not in self.evictable_leaves:
            self.evictable_leaves.add(node)

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size

    def _record_store_event(self, node: TreeNode):
        # One BlockStored per ``page_size`` chunk.
        if self.enable_kv_cache_events:
            # Compute hash_value lazily if not already set
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)

            # Get parent's last hash value for first page
            parent_block_hash = None
            if node.parent is not None and node.parent != self.root_node:
                if (
                    node.parent.hash_value is not None
                    and len(node.parent.hash_value) > 0
                ):
                    parent_block_hash = hash_str_to_int64(node.parent.hash_value[-1])

            page_index = 0
            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key.token_ids[start : start + self.page_size]
                if not page_tokens:
                    continue

                block_hash = hash_str_to_int64(node.hash_value[page_index])

                self.kv_event_queue.append(
                    BlockStored(
                        block_hashes=[block_hash],
                        parent_block_hash=parent_block_hash,
                        token_ids=page_tokens,
                        block_size=len(page_tokens),
                        lora_id=None,
                        medium=MEDIUM_GPU,
                    )
                )

                parent_block_hash = block_hash
                page_index += 1

    def _record_remove_event(self, node: TreeNode):
        # One BlockRemoved per chunk.
        if self.enable_kv_cache_events:
            # Compute hash_value lazily if not already set (must match what was stored)
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)

            page_index = 0
            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key.token_ids[start : start + self.page_size]
                if not page_tokens:
                    continue

                block_hash = hash_str_to_int64(node.hash_value[page_index])

                self.kv_event_queue.append(
                    BlockRemoved(block_hashes=[block_hash], medium=MEDIUM_GPU)
                )

                page_index += 1

    def _record_all_cleared_event(self):
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

    def take_events(self):
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events


if __name__ == "__main__":
    tree = RadixCache.create_simulated()

    # Example token id sequences (as lists of ints)
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 3], extra_key=None)))
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 3], extra_key=None)))
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 4, 5], extra_key=None)))
    tree.insert(
        InsertParams(key=RadixKey(token_ids=[1, 2, 4, 5, 6, 7], extra_key=None))
    )
    tree.insert(
        InsertParams(key=RadixKey(token_ids=[8, 9, 10, 11, 12], extra_key=None))
    )
    tree.pretty_print()

    print(
        tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=[1, 2, 3, 13, 14], extra_key=None))
        )
    )
