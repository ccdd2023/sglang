from __future__ import annotations

import types
import unittest

from sglang.srt.mem_cache.approx_kv.cachecraft_capability import (
    CACHECRAFT_DISPATCH_SYMBOL,
    CacheCraftServerCapability,
    _inspect_module_source,
    inspect_scheduler_dispatch_capability,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


class TestCacheCraftServerCapability(unittest.TestCase):
    def test_real_schedule_batch_has_no_cachecraft_dispatch_yet(self):
        # This is the honest, currently-true audit finding: the real
        # installed scheduler request path does not route CacheCraft-tagged
        # requests to CacheCraft's decision logic. If this test ever starts
        # failing because someone wired real scheduler dispatch, that is
        # good news -- update this test alongside the wiring, do not
        # silence it.
        capability = inspect_scheduler_dispatch_capability()
        self.assertFalse(capability.supported)
        self.assertFalse(bool(capability))
        self.assertIn(CACHECRAFT_DISPATCH_SYMBOL, capability.reason)
        self.assertIn("cachecraft", capability.reason)

    def test_reports_supported_when_dispatch_symbol_present(self):
        # Proves the probe is a genuine source inspection (not a
        # hard-coded False): a module whose source really does mention the
        # dispatch call is reported as supported.
        fake_module = types.ModuleType("fake_schedule_batch")
        fake_module.__dict__["source_text"] = (
            "def process_batch(self):\n"
            "    if self.approx_kv_metadata is not None:\n"
            "        restore_request_via_cachecraft(tree_cache, self)\n"
        )

        def fake_getsource(module):
            return module.source_text

        import sglang.srt.mem_cache.approx_kv.cachecraft_capability as capability_mod

        original_getsource = capability_mod.inspect.getsource
        capability_mod.inspect.getsource = fake_getsource
        try:
            capability = _inspect_module_source(fake_module)
        finally:
            capability_mod.inspect.getsource = original_getsource
        self.assertTrue(capability.supported)
        self.assertTrue(bool(capability))

    def test_reports_unsupported_when_dispatch_symbol_absent(self):
        fake_module = types.ModuleType("fake_schedule_batch_no_dispatch")
        fake_module.__dict__["source_text"] = (
            "def process_batch(self):\n"
            "    if self.approx_kv_metadata is not None:\n"
            "        restore_request_prefix(tree_cache, self)\n"
        )

        def fake_getsource(module):
            return module.source_text

        import sglang.srt.mem_cache.approx_kv.cachecraft_capability as capability_mod

        original_getsource = capability_mod.inspect.getsource
        capability_mod.inspect.getsource = fake_getsource
        try:
            capability = _inspect_module_source(fake_module)
        finally:
            capability_mod.inspect.getsource = original_getsource
        self.assertFalse(capability.supported)
        self.assertIn(CACHECRAFT_DISPATCH_SYMBOL, capability.reason)

    def test_source_introspection_failure_is_reported_not_raised(self):
        fake_module = types.ModuleType("fake_module_without_source")

        def failing_getsource(module):
            raise OSError("no source available")

        import sglang.srt.mem_cache.approx_kv.cachecraft_capability as capability_mod

        original_getsource = capability_mod.inspect.getsource
        capability_mod.inspect.getsource = failing_getsource
        try:
            capability = _inspect_module_source(fake_module)
        finally:
            capability_mod.inspect.getsource = original_getsource
        self.assertFalse(capability.supported)
        self.assertIn("unable to introspect", capability.reason)

    def test_capability_is_frozen_dataclass_with_bool_protocol(self):
        supported = CacheCraftServerCapability(supported=True, reason="ok")
        blocked = CacheCraftServerCapability(supported=False, reason="nope")
        self.assertTrue(bool(supported))
        self.assertFalse(bool(blocked))
        with self.assertRaises(Exception):
            supported.supported = False  # frozen dataclass


if __name__ == "__main__":
    unittest.main()
