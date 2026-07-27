from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.kvcomm.config import KVCommFeatureConfig
from sglang.srt.mem_cache.kvcomm.manager import KVCommManager
from sglang.srt.mem_cache.kvcomm.radix_backend import RoPEConfig
from sglang.srt.mem_cache.kvcomm.types import ResidencyTier, token_ids_hash
from sglang.srt.mem_cache.kvcomm_exact import (
    ExactMiddleCanaryController,
    ExactMiddleCase,
    ExactMiddlePhase,
)
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    kv_cache_copy_required,
)


class Cache:
    def __init__(self, size=64):
        self.layer_num = 1
        self.keys = [torch.arange(size * 8).reshape(size, 2, 4).float()]
        self.values = [self.keys[0].clone() + 1000]

    def move_kv_cache(self, target, source):
        self.keys[0][target] = self.keys[0][source].clone()
        self.values[0][target] = self.values[0][source].clone()

    def get_key_buffer(self, layer):
        return self.keys[layer]


class Allocator:
    def __init__(self):
        self.cache = Cache()
        self.next_slot = 20
        self.freed = []

    def alloc(self, length):
        value = torch.arange(self.next_slot, self.next_slot + length)
        self.next_slot += length
        return value

    def free(self, indices):
        self.freed.extend(indices.tolist())

    def get_kvcache(self):
        return self.cache

    def get_cpu_copy(self, indices):
        return [
            (
                self.cache.keys[layer][indices].clone(),
                self.cache.values[layer][indices].clone(),
            )
            for layer in range(self.cache.layer_num)
        ]

    def load_cpu_copy(self, payload, indices):
        for layer, (keys, values) in enumerate(payload):
            self.cache.keys[layer][indices] = keys
            self.cache.values[layer][indices] = values


class ReqPool:
    def __init__(self):
        self.req_to_token = torch.zeros((4, 32), dtype=torch.int32)

    def write(self, indices, values):
        self.req_to_token[indices] = values


def _case(
    source,
    target,
    source_start=2,
    target_start=3,
    length=3,
    target_uses=None,
    allow_target_prefix_bypass=False,
):
    return ExactMiddleCase(
        case_id="shifted",
        source_prompt_hash=token_ids_hash(source),
        target_prompt_hash=token_ids_hash(target),
        segment_token_hash=token_ids_hash(
            source[source_start : source_start + length]
        ),
        source_prefix_token_hash=token_ids_hash(source[:source_start]),
        target_prefix_token_hash=token_ids_hash(target[:target_start]),
        source_start=source_start,
        target_start=target_start,
        length=length,
        content_hash="shared-segment",
        allow_shifted_copy=True,
        allow_target_prefix_bypass=allow_target_prefix_bypass,
        policy_label="coding_aware",
        target_uses=target_uses,
    )


def _controller(
    *,
    host_overflow_enabled=False,
    ordinary_prefix_reuse_enabled=False,
):
    source = (1, 2, 3, 4, 5, 8, 9)
    target = (1, 7, 2, 3, 4, 5, 9)
    manager = KVCommManager(KVCommFeatureConfig(core_enabled=True))
    allocator = Allocator()
    pool = ReqPool()
    pool.req_to_token[0, : len(source)] = torch.arange(len(source))
    controller = ExactMiddleCanaryController(
        manager=manager,
        allocator=allocator,
        req_to_token_pool=pool,
        model_id="test",
        cache_dtype="fp32",
        rope=RoPEConfig(rotary_dim=0, base=10_000, is_neox_style=True),
        cases=(_case(source, target),),
        host_overflow_enabled=host_overflow_enabled,
        ordinary_prefix_reuse_enabled=ordinary_prefix_reuse_enabled,
    )
    return controller, source, target, allocator, pool


def _req(tokens, pool_index=0, prefix=()):
    req = SimpleNamespace(
        origin_input_ids=list(tokens),
        fill_ids=list(tokens),
        kv_committed_len=len(tokens),
        req_pool_idx=pool_index,
        prefix_indices=torch.tensor(prefix, dtype=torch.int64),
    )
    req.set_extend_input_len = lambda value: setattr(
        req, "extend_input_len", value
    )
    return req


def test_manifest_flag_enables_all_layer_kv_copy(monkeypatch):
    args = SimpleNamespace(speculative_algorithm=None)
    monkeypatch.delenv("SGLANG_KVCOMM_EXACT_CANARY_MANIFEST", raising=False)
    assert not kv_cache_copy_required(args)
    monkeypatch.setenv("SGLANG_KVCOMM_EXACT_CANARY_MANIFEST", "/tmp/reuse.json")
    assert kv_cache_copy_required(args)


def test_shifted_controller_materializes_and_commits_middle_span():
    controller, source, target, allocator, pool = _controller()
    handle = controller.maybe_materialize_source(_req(source))
    assert handle is not None
    owned_source = handle.backend_ref.indices.clone()

    target_req = _req(target, pool_index=1, prefix=(11, 12, 13))
    state = controller.maybe_attach_target(target_req)
    assert state is not None
    assert controller.stage_prefix_length(target_req) == 0
    stats = controller.copy_into_request(target_req)
    assert stats is not None and stats.mechanically_valid
    assert stats.copied_k_tokens == 3
    assert state.phase == ExactMiddlePhase.DENSE_SUFFIX
    copied = pool.req_to_token[1, 3:6].long()
    assert torch.equal(
        allocator.cache.values[0][copied],
        allocator.cache.values[0][owned_source],
    )
    controller.finish_request(target_req)
    assert controller.manager.store.lease_count == 0


def test_missing_source_falls_back_without_target_allocation():
    controller, _, target, allocator, _ = _controller()
    assert controller.maybe_attach_target(_req(target, pool_index=1)) is None
    assert allocator.next_slot == 20


def test_source_capacity_failure_skips_materialization():
    controller, source, _, allocator, _ = _controller()
    allocator.alloc = lambda length: None
    allocator.available_size = lambda: 0
    assert controller.maybe_materialize_source(_req(source)) is None
    assert controller.manager.store.record_count == 0


def test_source_capacity_overflow_uses_host_and_loads_on_target():
    controller, source, target, allocator, pool = _controller(
        host_overflow_enabled=True
    )
    original_alloc = allocator.alloc
    failed_once = False

    def fail_device_source_once(length):
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            return None
        return original_alloc(length)

    allocator.alloc = fail_device_source_once
    expected_values = allocator.cache.values[0][2:5].clone()
    handle = controller.maybe_materialize_source(_req(source))

    assert handle is not None
    assert handle.residency == ResidencyTier.HOST
    assert controller.owned_device_tokens == 0
    target_req = _req(target, pool_index=1, prefix=(11, 12, 13))
    state = controller.maybe_attach_target(target_req)
    assert state is not None and state.source.residency == ResidencyTier.HOST
    assert controller.stage_prefix_length(target_req) == 0
    stats = controller.copy_into_request(target_req)
    assert stats is not None and stats.mechanically_valid
    copied = pool.req_to_token[1, 3:6].long()
    assert torch.equal(allocator.cache.values[0][copied], expected_values)


def test_target_capacity_failure_falls_back_dense():
    controller, source, target, allocator, _ = _controller()
    assert controller.maybe_materialize_source(_req(source)) is not None
    target_req = _req(target, pool_index=1, prefix=(11, 12, 13))
    state = controller.maybe_attach_target(target_req)
    assert state is not None
    allocator.alloc = lambda length: None
    assert controller.copy_into_request(target_req) is None
    assert state.phase == ExactMiddlePhase.FALLBACK_DENSE
    assert state.fallback_reason == "target_allocation_capacity"


def test_controller_identifies_registered_source_and_target_prompts():
    controller, source, target, _, _ = _controller()
    assert controller.is_source_request(_req(source))
    assert not controller.is_source_request(_req(target))
    assert controller.is_target_request(_req(target))
    assert not controller.is_target_request(_req(source))


def test_ordinary_prefix_reuse_is_opt_in_and_stops_before_middle():
    controller, source, target, _, _ = _controller()
    assert controller.ordinary_prefix_match_limit(_req(source)) == 0
    assert controller.ordinary_prefix_match_limit(_req(target)) == 0

    dual, source, target, _, _ = _controller(
        ordinary_prefix_reuse_enabled=True
    )
    assert dual.ordinary_prefix_match_limit(_req(source)) is None
    assert dual.ordinary_prefix_match_limit(_req(target)) == 3


def test_target_prefix_bypass_copies_only_uncached_repository_tail():
    source = (1, 2, 3, 4, 5, 8, 9)
    target = (1, 7, 2, 3, 4, 5, 9)
    manager = KVCommManager(KVCommFeatureConfig(core_enabled=True))
    allocator = Allocator()
    pool = ReqPool()
    pool.req_to_token[0, : len(source)] = torch.arange(len(source))
    case = _case(
        source,
        target,
        target_uses=2,
        allow_target_prefix_bypass=True,
    )
    controller = ExactMiddleCanaryController(
        manager=manager,
        allocator=allocator,
        req_to_token_pool=pool,
        model_id="test",
        cache_dtype="fp32",
        rope=RoPEConfig(rotary_dim=0, base=10_000, is_neox_style=True),
        cases=(case,),
        ordinary_prefix_reuse_enabled=True,
    )
    handle = controller.maybe_materialize_source(_req(source))
    assert handle is not None
    owned_source = handle.backend_ref.indices.clone()
    assert controller.ordinary_prefix_match_limit(_req(target)) == 6

    partial = _req(target, pool_index=1, prefix=(11, 12, 13, 14))
    assert controller.maybe_attach_target(partial) is not None
    assert controller.copy_ready(partial)
    stats = controller.copy_into_request(partial)
    assert stats is not None and stats.copied_k_tokens == 2
    copied = pool.req_to_token[1, 4:6].long()
    assert torch.equal(
        allocator.cache.values[0][copied],
        allocator.cache.values[0][owned_source[1:]],
    )
    controller.finish_request(partial)

    complete = _req(
        target,
        pool_index=2,
        prefix=(11, 12, 13, 14, 15, 16),
    )
    state = controller.maybe_attach_target(complete)
    assert state is not None
    assert not controller.copy_ready(complete)
    assert controller.stage_prefix_length(complete) is None
    assert state.phase == ExactMiddlePhase.DENSE_SUFFIX
    controller.finish_request(complete)
    assert manager.store.record_count == 0


def test_configured_target_use_count_releases_owned_source():
    source = (1, 2, 3, 4, 5, 8, 9)
    target = (1, 7, 2, 3, 4, 5, 9)
    manager = KVCommManager(KVCommFeatureConfig(core_enabled=True))
    allocator = Allocator()
    pool = ReqPool()
    pool.req_to_token[0, : len(source)] = torch.arange(len(source))
    controller = ExactMiddleCanaryController(
        manager=manager,
        allocator=allocator,
        req_to_token_pool=pool,
        model_id="test",
        cache_dtype="fp32",
        rope=RoPEConfig(rotary_dim=0, base=10_000, is_neox_style=True),
        cases=(_case(source, target, target_uses=1),),
    )
    controller.maybe_materialize_source(_req(source))
    target_req = _req(target, pool_index=1, prefix=(11, 12, 13))
    controller.maybe_attach_target(target_req)
    controller.copy_into_request(target_req)
    controller.finish_request(target_req)
    assert manager.store.record_count == 0
    assert controller.owned_device_tokens == 0


def test_repeated_target_use_holds_source_lease_until_last_target():
    source = (1, 2, 3, 4, 5, 8, 9)
    target = (1, 7, 2, 3, 4, 5, 9)
    manager = KVCommManager(KVCommFeatureConfig(core_enabled=True))
    allocator = Allocator()
    pool = ReqPool()
    pool.req_to_token[0, : len(source)] = torch.arange(len(source))
    controller = ExactMiddleCanaryController(
        manager=manager,
        allocator=allocator,
        req_to_token_pool=pool,
        model_id="test",
        cache_dtype="fp32",
        rope=RoPEConfig(rotary_dim=0, base=10_000, is_neox_style=True),
        cases=(_case(source, target, target_uses=2),),
    )
    controller.maybe_materialize_source(_req(source))
    assert manager.store.lease_count == 1

    first = _req(target, pool_index=1, prefix=(11, 12, 13))
    assert controller.maybe_attach_target(first) is not None
    controller.copy_into_request(first)
    controller.finish_request(first)
    assert manager.store.record_count == 1
    assert manager.store.lease_count == 1

    second = _req(target, pool_index=2, prefix=(11, 12, 13))
    assert controller.maybe_attach_target(second) is not None
    controller.copy_into_request(second)
    controller.finish_request(second)
    assert manager.store.record_count == 0
    assert manager.store.lease_count == 0


def test_shifted_case_requires_explicit_opt_in():
    source = (1, 2, 3, 4)
    target = (1, 9, 2, 3)
    with pytest.raises(ValueError, match="manifest version 2"):
        ExactMiddleCase(
            case_id="bad",
            source_prompt_hash=token_ids_hash(source),
            target_prompt_hash=token_ids_hash(target),
            segment_token_hash=token_ids_hash(source[1:3]),
            source_prefix_token_hash=token_ids_hash(source[:1]),
            target_prefix_token_hash=token_ids_hash(target[:2]),
            source_start=1,
            target_start=2,
            length=2,
            content_hash="segment",
        )


def test_dynamic_v3_sidecar_adds_source_target_and_releases(tmp_path):
    source = (1, 2, 3, 4, 5, 8, 9)
    target = (1, 7, 2, 3, 4, 5, 9)
    case = _case(source, target, target_uses=1)
    manifest = tmp_path / "dynamic.json"
    base = {
        "version": 3,
        "model_id": "test",
        "cache_dtype": "fp32",
        "lease_ttl_s": 30,
        "ledger_path": str(tmp_path / "ledger.jsonl"),
        "rope": {
            "rotary_dim": 0,
            "base": 10_000,
            "is_neox_style": True,
        },
        "sources": [],
        "cases": [],
        "release_source_ids": [],
    }
    manifest.write_text(json.dumps(base), encoding="utf-8")

    manager = KVCommManager(KVCommFeatureConfig(core_enabled=True))
    allocator = Allocator()
    pool = ReqPool()
    reclaimed = []
    pool.req_to_token[0, : len(source)] = torch.arange(len(source))
    controller = ExactMiddleCanaryController.from_manifest(
        manifest,
        manager=manager,
        allocator=allocator,
        req_to_token_pool=pool,
        model_id="test",
        cache_dtype="fp32",
        reclaim_device_tokens=reclaimed.append,
    )
    assert not controller.is_source_request(_req(source))

    source_id = "dynamic-source"
    base["sources"].append(
        {
            "source_id": source_id,
            "source_prompt_hash": case.source_prompt_hash,
            "segment_token_hash": case.segment_token_hash,
            "source_prefix_token_hash": case.source_prefix_token_hash,
            "source_start": case.source_start,
            "length": case.length,
            "content_hash": case.content_hash,
            "policy_label": case.policy_label,
        }
    )
    base["cases"].append(
        {
            "case_id": case.case_id,
            "source_id": source_id,
            "source_prompt_hash": case.source_prompt_hash,
            "target_prompt_hash": case.target_prompt_hash,
            "segment_token_hash": case.segment_token_hash,
            "source_prefix_token_hash": case.source_prefix_token_hash,
            "target_prefix_token_hash": case.target_prefix_token_hash,
            "source_start": case.source_start,
            "target_start": case.target_start,
            "length": case.length,
            "content_hash": case.content_hash,
            "policy_label": case.policy_label,
            "target_uses": 1,
        }
    )
    replacement = manifest.with_suffix(".new")
    replacement.write_text(json.dumps(base), encoding="utf-8")
    replacement.replace(manifest)

    assert controller.is_source_request(_req(source))
    controller.maybe_materialize_source(_req(source))
    target_req = _req(target, pool_index=1, prefix=(11, 12, 13))
    assert controller.is_target_request(target_req)
    controller.maybe_attach_target(target_req)
    controller.copy_into_request(target_req)
    controller.finish_request(target_req)
    assert manager.store.record_count == 0
    assert controller.owned_device_tokens == 0
    assert reclaimed == [case.length, case.length]
