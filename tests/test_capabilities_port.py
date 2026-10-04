"""Lock the capability script to a loopback socket."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch


def _load_script():
    path = (
        Path(__file__).resolve().parents[1] / "scripts" / "test_model_capabilities.py"
    )
    spec = importlib.util.spec_from_file_location("test_model_capabilities", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_find_free_port_binds_loopback_only():
    """Port discovery must not listen on every network interface."""
    module = _load_script()
    bound = {}

    class _Socket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def bind(self, address):
            bound["address"] = address

        def getsockname(self):
            return ("127.0.0.1", 9)

    with patch.object(
        module.socket,
        "socket",
        side_effect=lambda *_args, **_kwargs: _Socket(),
    ):
        assert module.find_free_port() == 9
    assert bound["address"] == ("127.0.0.1", 0)
