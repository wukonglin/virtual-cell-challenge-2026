"""Load the repository's scorer-summary helpers without the GPU model stack.

summarize_public_validation imports three tiny I/O helpers from modules that
also import Torch. EXS-1 is intentionally CPU-only and does not otherwise need
those model modules, so this adapter supplies API-compatible I/O functions only
while the existing summary module is loaded. The temporary modules are removed
immediately; test discovery and later imports are not polluted.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

from exs1_local_io import (
    atomic_write_json,
    describe_file,
    require,
)


def _helper_module(
    name: str, attributes: dict[str, Any]
) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_summary() -> types.ModuleType:
    module_name = "_exs1_summarize_public_validation"
    summary_path = (
        Path(__file__).resolve().with_name(
            "summarize_public_validation.py"
        )
    )
    saved = {
        name: sys.modules.get(name)
        for name in (
            "generate_state_direct_counts",
            "infer_state_effect_prior",
        )
    }
    sys.modules["generate_state_direct_counts"] = _helper_module(
        "generate_state_direct_counts",
        {"atomic_write_json": atomic_write_json},
    )
    sys.modules["infer_state_effect_prior"] = _helper_module(
        "infer_state_effect_prior",
        {"describe_file": describe_file, "require": require},
    )
    try:
        spec = importlib.util.spec_from_file_location(
            module_name, summary_path
        )
        require(
            spec is not None and spec.loader is not None,
            "Cannot load scorer-summary helpers",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


_SUMMARY = _load_summary()
MSE_METRIC = _SUMMARY.MSE_METRIC
NMAE_METRIC = _SUMMARY.NMAE_METRIC
_mean_summary = _SUMMARY._mean_summary
_read_results = _SUMMARY._read_results
_validate_cell_eval_sidecars = (
    _SUMMARY._validate_cell_eval_sidecars
)
