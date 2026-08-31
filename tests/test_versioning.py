"""config_hash must be stable across formatting and sensitive to content."""
from __future__ import annotations

from swing.common.versioning import config_hash


def test_hash_is_deterministic():
    assert config_hash() == config_hash()


def test_hash_is_hex_and_short():
    h = config_hash()
    assert len(h) == 16 and all(c in "0123456789abcdef" for c in h)


def test_hash_tracks_semantic_content_not_formatting(tmp_path, monkeypatch):
    import swing.common.versioning as v

    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("x: 1\ny: 2\n")
    b.write_text("y: 2\n\n\nx:   1\n")   # same data, different formatting
    assert v._canonical(a) == v._canonical(b)

    c = tmp_path / "c.yaml"
    c.write_text("x: 1\ny: 3\n")          # different data
    assert v._canonical(a) != v._canonical(c)
