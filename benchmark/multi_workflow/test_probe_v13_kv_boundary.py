from benchmark.multi_workflow.probe_v13_kv_boundary import zone_slices


def test_zone_slices_cover_span_without_overlap():
    zones = zone_slices(53, 16)
    assert zones == {
        "head": slice(0, 16),
        "interior": slice(16, 37),
        "tail": slice(37, 53),
    }
    positions = [
        index
        for span in zones.values()
        for index in range(span.start, span.stop)
    ]
    assert positions == list(range(53))
