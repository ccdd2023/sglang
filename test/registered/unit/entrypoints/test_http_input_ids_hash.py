from sglang.srt.utils.input_identity import input_ids_sha256


def test_input_ids_sha256_matches_common_agent_encoding() -> None:
    # The common bridge hashes every signed token id as little-endian int64.
    assert input_ids_sha256([1, 2, 32000]) == (
        "2c9f203d8bdd52a862ad8a7993643262609ffe80830413f2af736b6baf7cf51e"
    )


def test_input_ids_sha256_ignores_text_and_batch_inputs() -> None:
    assert input_ids_sha256(None) is None
    assert input_ids_sha256([[1, 2], [3, 4]]) is None
