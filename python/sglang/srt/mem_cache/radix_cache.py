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
import logging
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
    ):
        self.signature = signature
        self.code_content_signature = code_content_signature
        self.token_ids = token_ids
        self.kv_indices = kv_indices
        self.start_pos = start_pos
        self.ref_count = 1
        self.source_node = source_node
        self.prefetch_protected_until = 0.0
        self.prefetch_steps_remaining = 0
        self.prefetch_lock_held = False
        self.prefetch_hit_count = 0

    def __repr__(self):
        return (
            f"AnchorKVEntry(sig={self.signature!r:.30}, "
            f"start_pos={self.start_pos}, len={len(self.token_ids)}, ref={self.ref_count})"
        )

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

    def _agenttemplatekv_protect_entry(
        self,
        entry: AnchorKVEntry,
        *,
        req: Optional[Req] = None,
        steps_to_use: int = 1,
        ttl_s: Optional[float] = None,
        max_ancestors: int = 2,
    ) -> bool:
        """Keep an exact-content anchor resident for the next coding agent.

        Safety-net cap: if locking the entry would push ``protected_size_``
        above ``_agenttemplatekv_protected_size_cap()``, the protect is
        rejected (treated as a miss). This bounds cumulative state across
        cases so the KV pool never gets starved.

        Capped walk: instead of locking every ancestor up to root, only
        ``max_ancestors`` (default 2) levels are locked, plus the leaf.
        Per-protect cost drops from O(prefix_length) to O(leaf+small).
        """
        now = time.monotonic()
        ttl = (
            float(ttl_s)
            if ttl_s is not None
            else float(os.environ.get("SGLANG_AGENTTEMPLATEKV_PREFETCH_TTL_S", "60"))
        )
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
                    logger.warning(
                        "agenttemplatekv_protect rejected: "
                        "protected_size_=%d + token_ids=%d > cap=%d "
                        "(eviction sweep could not free enough)",
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
            # Capped walk: only lock the leaf + max_ancestors ancestors.
            locked = self._inc_lock_ref_capped(node, max_ancestors=max_ancestors)
            setattr(entry, "_protected_ancestor_nodes", locked)
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
                self.dec_lock_ref(node)
        else:
            for locked_node in locked:
                if locked_node is not None and not locked_node.evicted:
                    self._dec_lock_ref_one(locked_node)
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
        """Device-first codebase prefetch for AgentTemplateKV.

        This path does not depend on HiCache. It checks whether exact-content
        anchors from earlier agents are already resident on device, pins them
        for the next template step, and records a real device hit.
        """
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
            )
            segment_content_signature = str(
                span.get("content_signature", "") or content_signature
            )
            if not segment_content_signature:
                continue

            entry.code_content_signature = segment_content_signature
            with self.anchor_kv_store_lock:
                self.anchor_kv_store.setdefault(segment_content_signature, []).append(entry)
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

        match_reason = getattr(req, "lossy_first_match_reason", "")
        if match_reason not in (
            "exact_anchor_signature",
            "exact_code_content_signature",
            "span_overlap_high",
            "span_overlap_medium",
        ):
            return exact_values, exact_node

        matched_content_sig = getattr(req, "lossy_first_matched_content_signature", "") or ""
        if not matched_content_sig:
            matched_content_sig = getattr(req, "code_content_signature", "") or ""
        if not matched_content_sig:
            return exact_values, exact_node

        with self.anchor_kv_store_lock:
            entries = list(self.anchor_kv_store.get(matched_content_sig, []))
        if not entries:
            return exact_values, exact_node

        key_tokens = key.token_ids
        token_spans = getattr(req, "code_anchor_token_spans", None) or []
        skip_check = os.environ.get("SGLANG_LOSSY_SKIP_TOKEN_CHECK", "0") == "1"
        req_content_signature = matched_content_sig

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
            for span in token_spans:
                s_start = span.get("start_token", -1)
                s_end = span.get("end_token", -1)
                span_content_signature = str(
                    span.get("content_signature", "") or req_content_signature
                )
                if span_content_signature != entry.code_content_signature:
                    continue
                if s_start < 0 or s_end > len(key_tokens) or s_end - s_start != entry_len:
                    continue
                if skip_check:
                    anchor_pos = s_start
                    break
                span_tokens = torch.tensor(
                    key_tokens[s_start:s_end],
                    dtype=entry.token_ids.dtype,
                    device=entry.token_ids.device,
                )
                if torch.equal(span_tokens, entry.token_ids):
                    anchor_pos = s_start
                    break

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
                        continue
                anchor_pos = entry.start_pos

            anchor_end = anchor_pos + entry_len
            if exact_len >= anchor_end:
                continue

            suffix_start = max(0, exact_len - anchor_pos)
            copy_len = entry_len - suffix_start
            if copy_len <= 0:
                continue

            gap_len = max(0, anchor_pos - exact_len)
            max_gap = int(os.environ.get("SGLANG_AGENTTEMPLATEKV_MAX_ZERO_GAP", "16"))
            if gap_len > max_gap:
                setattr(req, "lossy_rejected_reason", "agenttemplatekv_large_zero_gap")
                setattr(req, "lossy_anchor_match_gap_len", gap_len)
                setattr(req, "agenttemplatekv_rejected_large_gap_count", 1)
                logger.info(
                    "[agenttemplatekv] reject content_sig=%s gap_len=%d max_gap=%d",
                    matched_content_sig,
                    gap_len,
                    max_gap,
                )
                continue

            total_new = gap_len + copy_len

            new_slots = self.token_to_kv_pool_allocator.alloc(total_new)
            if new_slots is None:
                logger.warning("[anchor_kv] alloc failed for content_sig=%s", matched_content_sig)
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
                    delta, matched_content_sig,
                    entry.start_pos + suffix_start, exact_len + gap_len,
                )

            extended = exact_values + [new_slots]
            setattr(req, "lossy_anchor_match_used", True)
            setattr(req, "lossy_anchor_match_len", copy_len)
            setattr(req, "lossy_anchor_match_gap_len", gap_len)
            setattr(req, "lossy_anchor_match_signature", entry.signature)
            setattr(req, "lossy_anchor_match_content_signature", matched_content_sig)
            setattr(req, "lossy_anchor_rope_delta", delta)
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
            return extended, exact_node

        return exact_values, exact_node

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        """Cache request when it finishes."""
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
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

            self._record_remove_event(x)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            node = node.parent
        return IncLockRefResult(delta=delta)

    # ------------------------------------------------------------------
    # AgentTemplateKV device-first protected-anchor helpers
    # ------------------------------------------------------------------
    def _inc_lock_ref_capped(
        self, node: TreeNode, max_ancestors: int = 2
    ) -> list:
        """Like ``inc_lock_ref`` but stops after ``max_ancestors`` steps from
        ``node`` toward root, instead of walking all the way to root.

        This bounds the per-protect cost when a deep ``source_node`` (the
        full-prompt leaf, ~14k tokens) is protected: only the leaf + 2
        ancestors are locked, not the entire prefix path back to root.
        Returns the list of nodes that were locked (for symmetric release).
        """
        if self.disable:
            return []
        locked = []
        steps = 0
        cur = node
        while cur is not None and cur != self.root_node and steps <= max_ancestors:
            if cur.lock_ref == 0:
                self.evictable_size_ -= len(cur.key)
                self.protected_size_ += len(cur.key)
            cur.lock_ref += 1
            self._update_leaf_status(cur)
            locked.append(cur)
            cur = cur.parent
            steps += 1
        return locked

    def _dec_lock_ref_one(self, node: TreeNode) -> None:
        """Single-node decrement of ``lock_ref``, mirroring the body of
        ``dec_lock_ref`` but for exactly one level (not a full walk to root).
        Used by ``_agenttemplatekv_release_entry`` to release the capped
        chain stored by ``_inc_lock_ref_capped``.
        """
        if self.disable or node is None:
            return
        if node.lock_ref == 1:
            self.evictable_size_ += len(node.key)
            self.protected_size_ -= len(node.key)
        if node.lock_ref > 0:
            node.lock_ref -= 1
        self._update_leaf_status(node)

    def _agenttemplatekv_protected_size_cap(self) -> int:
        """Return the protected-anchor size cap in tokens.  A new protect
        that would push ``protected_size_`` above this cap is rejected.

        Default: ``0.5 * SGLANG_MAX_TOTAL_TOKENS`` (32 768 when the pool is
        65 536).  Overridable via ``SGLANG_AGENTTEMPLATEKV_PROTECTED_FRAC``
        (float) or ``SGLANG_AGENTTEMPLATEKV_PROTECTED_MAX_TOKENS`` (int).
        Returns 0 to disable the cap.
        """
        override = os.environ.get("SGLANG_AGENTTEMPLATEKV_PROTECTED_MAX_TOKENS")
        if override is not None:
            try:
                return max(0, int(override))
            except ValueError:
                pass
        frac_override = os.environ.get("SGLANG_AGENTTEMPLATEKV_PROTECTED_FRAC")
        try:
            frac = (
                float(frac_override)
                if frac_override is not None
                else 0.5
            )
        except ValueError:
            frac = 0.5
        max_total = int(os.environ.get("SGLANG_MAX_TOTAL_TOKENS", "65536"))
        return max(0, int(frac * max_total))

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
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
        if anchor_spans and not node.anchor_spans:
            node.anchor_spans = list(anchor_spans)
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
                if anchor_spans and not new_node.anchor_spans:
                    new_node.anchor_spans = list(anchor_spans)
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
                if anchor_spans and not node.anchor_spans:
                    node.anchor_spans = list(anchor_spans)
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
            if anchor_spans:
                new_node.anchor_spans = list(anchor_spans)
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
