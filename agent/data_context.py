"""
data_context.py — module-level store for the current chunk's raw dimuon
data. Claude only ever sees a lightweight summary (n_events, mean mass,
Z-peak stats) in the conversation; the raw reco_mass array lives here and
is fetched by tools on demand.
"""

_current_chunk_id: str | None = None
_current_chunk: dict | None = None  # expects at least a "reco_mass" key


def set_chunk(chunk_id: str, chunk_data: dict) -> None:
    global _current_chunk_id, _current_chunk
    _current_chunk_id = chunk_id
    _current_chunk = chunk_data


def get_reco_mass():
    if _current_chunk is None:
        raise RuntimeError(
            f"No chunk data set (current chunk id: {_current_chunk_id}). "
            f"Was set_chunk() called before analysis?"
        )
    return _current_chunk["reco_mass"]


def get_n_events() -> int:
    return len(get_reco_mass())


def current_chunk_id() -> str | None:
    return _current_chunk_id
