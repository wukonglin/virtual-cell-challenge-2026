#!/usr/bin/env python3
"""Authenticate pinned scDFM source and preflight its runtime imports."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from authenticate_scdfm_source import authenticate_source


SCHEMA = "vcc-scdfm-environment-preflight-v1"

# Import name to distribution name. These packages are imported by official
# training or the Norman/ComboSciPlex preprocessing paths at the pinned commit.
REQUIRED_IMPORTS: Mapping[str, str] = {
    "accelerate": "accelerate",
    "anndata": "anndata",
    "bs4": "beautifulsoup4",
    "cell_eval": "cell-eval",
    "jax": "jax",
    "networkx": "networkx",
    "numpy": "numpy",
    "ot": "POT",
    "pandas": "pandas",
    "pertpy": "pertpy",
    "rdkit": "rdkit",
    "requests": "requests",
    "rich": "rich",
    "scanpy": "scanpy",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "timm": "timm",
    "torch": "torch",
    "torchdiffeq": "torchdiffeq",
    "tqdm": "tqdm",
    "triton": "triton",
    "typing_extensions": "typing-extensions",
    "tyro": "tyro",
}

# These are guarded alternative scGPT attention backends. The official
# PAD-Transformer/origin training path does not select either backend.
OPTIONAL_BACKENDS = frozenset({"fast_transformers", "flash_attn"})
LOCAL_TOP_LEVEL = frozenset({"config", "src"})


class EnvironmentPreflightError(RuntimeError):
    """Raised when static dependency coverage or importability is incomplete."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EnvironmentPreflightError(message)


def discover_external_imports(source: Path) -> dict[str, list[str]]:
    """Statically collect top-level non-stdlib imports and their source files."""

    discovered: dict[str, set[str]] = {}
    for path in sorted(source.rglob("*.py")):
        relative = path.relative_to(source).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeError, SyntaxError) as error:
            raise EnvironmentPreflightError(f"Cannot parse {relative}: {error}") from error
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".", 1)[0]]
            for name in names:
                if name in sys.stdlib_module_names or name in LOCAL_TOP_LEVEL:
                    continue
                discovered.setdefault(name, set()).add(relative)
    return {name: sorted(paths) for name, paths in sorted(discovered.items())}


def preflight_environment(
    source: Path,
    *,
    source_authenticator: Callable[[Path], Mapping[str, Any]] = authenticate_source,
    importer: Callable[[str], Any] = importlib.import_module,
) -> dict[str, Any]:
    """Authenticate source, prove static coverage, and import required packages."""

    source_receipt = source_authenticator(source)
    _require(source_receipt.get("status") == "passed", "Source authentication failed")
    discovered = discover_external_imports(source)
    discovered_names = set(discovered)
    covered_names = set(REQUIRED_IMPORTS) | set(OPTIONAL_BACKENDS)
    unknown = sorted(discovered_names - covered_names)
    _require(not unknown, "Unmapped upstream imports: " + ", ".join(unknown))
    missing_static = sorted(set(REQUIRED_IMPORTS) - discovered_names)
    _require(
        not missing_static,
        "Required import map contains names absent from pinned source: "
        + ", ".join(missing_static),
    )

    imported: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for import_name, distribution in REQUIRED_IMPORTS.items():
        try:
            module = importer(import_name)
        except Exception as error:
            failures.append(f"{import_name}: {type(error).__name__}: {error}")
            continue
        imported[import_name] = {
            "distribution": distribution,
            "module_version": getattr(module, "__version__", None),
            "source_files": discovered[import_name],
        }
    _require(not failures, "Required imports failed: " + "; ".join(failures))

    optional = {}
    for import_name in sorted(OPTIONAL_BACKENDS):
        try:
            importer(import_name)
        except Exception:
            optional[import_name] = False
        else:
            optional[import_name] = True

    return {
        "schema": SCHEMA,
        "status": "passed",
        "checks": {
            "pinned_source_authenticated": True,
            "all_static_external_imports_are_classified": True,
            "all_required_runtime_packages_import": True,
            "no_model_or_data_object_instantiated": True,
        },
        "source": source_receipt,
        "required_imports": imported,
        "optional_attention_backends": optional,
        "notes": {
            "optional_backends_required_for_origin_model": False,
            "network_requests_performed": False,
            "upstream_modules_imported": False,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    receipt = preflight_environment(args.source)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


if __name__ == "__main__":
    main()
