"""Small helpers for auditable token-identity measurements."""

from __future__ import annotations

import hashlib
from typing import List, Optional, Union


def input_ids_sha256(
    input_ids: Optional[Union[List[List[int]], List[int]]],
) -> Optional[str]:
    """Hash the exact single-request token sequence consumed by /generate."""
    if not input_ids or not isinstance(input_ids[0], int):
        return None
    digest = hashlib.sha256()
    for token_id in input_ids:
        digest.update(int(token_id).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()
