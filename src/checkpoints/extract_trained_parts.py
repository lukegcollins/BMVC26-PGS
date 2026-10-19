"""Extract the trained parts of a headline checkpoint and check that they rebuild the original model exactly.

Each original Lightning checkpoint holds the whole DINOv3 ViT-L/16 backbone, the classification head, the
training objective's own parameters (``loss_fn.*``), optimiser and scheduler state, and callback state. Most of
the backbone is bit-identical to the public base weights (Hugging Face ``facebook/dinov3-vitl16-pretrain-lvd1689m``
at revision ``HF_REVISION``), which are not ours to redistribute. The released file keeps only

- every backbone tensor that differs from the base weights (the tuned LayerNorms of the LN models),
- the LoRA adapter tensors (``lora_A``, ``lora_B``) of the LoRA and GSD-LoRA models,
- the classification head,

under the key names of the original ``state_dict``. ``loss_fn.*`` is dropped: it parameterises the training
objective, and inference (``test_step``) calls only the detector. Optimiser, scheduler, loop and callback state
are dropped too; the callback state records absolute paths of the training machine.

Key layout of the original checkpoints:

- backbone tensors are ``model._orig_mod.backbone.model.<hf_key>`` (``_orig_mod`` comes from ``torch.compile``);
- in LoRA and GSD-LoRA models the backbone is wrapped by PEFT: its keys gain ``base_model.model.`` and each
  adapted linear layer stores its base weight as ``<name>.base_layer.weight|bias``;
- the head is ``model._orig_mod.head.*``.

To rebuild a model, build the detector from the sidecar's ``config``, load the base weights at ``HF_REVISION``,
and load the released tensors over them. ``reassemble`` does the same at the level of state_dict entries, and
``verify`` compares its output with the original checkpoint tensor by tensor.

Usage (training environment, which provides torch and safetensors; run from the repository root)::

    python src/checkpoints/extract_trained_parts.py --hf <model.safetensors> --out data/release \\
        --sidecars checkpoints [--forbid <name> ...] <run>/checkpoints/<file>.ckpt ...

The sidecars are refused if any string in them is an absolute or home-relative path, or contains a ``--forbid``
string. Machine and directory names to refuse are passed at run time rather than listed here, since a list in
this file would publish them.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
from collections.abc import Iterator, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
except ImportError:  # the repository environment has no torch; main() says so
    torch = None

HF_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
HF_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
HF_SHA256 = "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"

BACKBONE_PREFIX = "model._orig_mod.backbone.model."
PEFT_PREFIX = "base_model.model."
HEAD_PREFIX = "model._orig_mod.head."
LOSS_PREFIX = "loss_fn."

# a path that starts at a filesystem root or a home directory, anywhere in a string (not the "//" of a URL)
ABSOLUTE_PATH = re.compile(r"""(?:^|[\s"'=,;:(\[])(?:/(?!/)|~/|[A-Za-z]:[\\/])""")

RUN_RE = re.compile(r"^orthoflow_gtl_(ln|lora|gsd_lora)_s(\d+)$")
CKPT_RE = re.compile(r"^epoch=(\d+)-video_auc=(\d+\.\d+)\.ckpt$")
VARIANTS = {"ln": "LN", "lora": "LoRA", "gsd_lora": "GSD-LoRA"}
SIDECAR_KEYS = (
    "run", "variant", "seed", "epoch", "val_video_auc", "source_file", "source_sha256", "hf_id", "hf_revision",
    "hf_sha256", "key_prefix", "peft_wrapped", "kept_keys", "dropped_groups", "config", "parts_sha256",
)


def hf_key(ckpt_key: str) -> str | None:
    """The base-weight key a checkpoint key corresponds to, or None if it has no base-weight counterpart."""
    if not ckpt_key.startswith(BACKBONE_PREFIX):
        return None
    rest = ckpt_key.removeprefix(BACKBONE_PREFIX).removeprefix(PEFT_PREFIX)
    if any(part.startswith("lora_") for part in rest.split(".")):
        return None
    for leaf in ("weight", "bias"):
        if rest.endswith(f".base_layer.{leaf}"):
            return rest[: -len(f".base_layer.{leaf}")] + f".{leaf}"
    return rest


def _identical(a, b) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(a, b))


def _group(key: str) -> str:
    if key.startswith(LOSS_PREFIX):
        return "loss_fn"
    if key.startswith(HEAD_PREFIX):
        return "head"
    if key.startswith(BACKBONE_PREFIX):
        return "backbone" if hf_key(key) is not None else "lora"
    return "other"


def select_trained(sd: Mapping[str, Any], hf: Mapping[str, Any]) -> dict[str, Any]:
    """Every tensor the base weights cannot supply: changed backbone tensors, LoRA, head; never ``loss_fn.*``."""
    kept = {}
    for key, tensor in sd.items():
        if key.startswith(LOSS_PREFIX):
            continue
        base = hf_key(key)
        if base is None or base not in hf or not _identical(tensor, hf[base]):
            kept[key] = tensor
    return kept


def _resolve(key: str, parts: Mapping[str, Any], hf: Mapping[str, Any]):
    if key in parts:
        return parts[key]
    base = hf_key(key)
    if base is None or base not in hf:
        raise KeyError(f"{key} is neither in the released parts nor in the base weights")
    return hf[base]


def reassemble(parts: Mapping[str, Any], hf: Mapping[str, Any], keys: list[str]) -> dict[str, Any]:
    """The full state_dict for ``keys``: released tensors where present, base weights everywhere else."""
    return {key: _resolve(key, parts, hf) for key in keys}


def _walk_clean(value: Any, where: str, forbid: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        if ABSOLUTE_PATH.search(value):
            raise ValueError(f"config value at {where} holds an absolute path")
        if any(s in value for s in forbid):
            raise ValueError(f"config value at {where} contains a forbidden string")
        return value
    if isinstance(value, Mapping):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(f"config key {k!r} at {where} is not a string")
            _walk_clean(k, f"{where} (key)", forbid)
            out[k] = _walk_clean(v, f"{where}.{k}", forbid)
        return out
    if isinstance(value, (list, tuple)):
        return [_walk_clean(v, f"{where}[{i}]", forbid) for i, v in enumerate(value)]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise TypeError(f"config value at {where} has type {type(value).__name__}")


def sanitise_config(cfg: dict, forbid: tuple[str, ...] = ()) -> dict:
    """A JSON-ready copy of ``cfg``; ValueError if any string is an absolute path or contains a ``forbid`` string."""
    return _walk_clean(cfg, "config", tuple(s for s in forbid if s))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=4)
def _sha256_cached(path: str, size: int, mtime_ns: int) -> str:
    return _sha256_file(Path(path))


def sha256_file(path: Path) -> str:
    st = Path(path).stat()
    return _sha256_cached(str(Path(path).resolve()), st.st_size, st.st_mtime_ns)


class _LazyTensors(Mapping):
    """Read-only mapping over an open safetensors file; each tensor is read when asked for."""

    def __init__(self, handle):
        self._h = handle
        self._keys = list(handle.keys())
        self._set = set(self._keys)

    def __getitem__(self, key: str):
        if key not in self._set:
            raise KeyError(key)
        return self._h.get_tensor(key)

    def __contains__(self, key: object) -> bool:
        return key in self._set

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)


def _load_checkpoint(ckpt: Path) -> dict:
    return torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)


def parse_run(run: str) -> tuple[str, int]:
    m = RUN_RE.match(run)
    if not m:
        raise ValueError(f"run name {run!r} is not orthoflow_gtl_<ln|lora|gsd_lora>_s<seed>")
    return VARIANTS[m.group(1)], int(m.group(2))


def parse_checkpoint_name(name: str) -> tuple[int, float]:
    m = CKPT_RE.match(name)
    if not m:
        raise ValueError(f"checkpoint name {name!r} is not epoch=<n>-video_auc=<auc>.ckpt")
    return int(m.group(1)), float(m.group(2))


def extract(ckpt: Path, hf_file: Path, out_dir: Path, run: str, forbid: tuple[str, ...] = ()) -> dict:
    """Write ``<out_dir>/<run>.safetensors`` with the trained parts of ``ckpt`` and return its sidecar."""
    ckpt, hf_file, out_dir = Path(ckpt), Path(hf_file), Path(out_dir)
    variant, seed = parse_run(run)
    epoch, val_auc = parse_checkpoint_name(ckpt.name)
    if ckpt.parent.name != "checkpoints" or ckpt.parent.parent.name != run:
        raise ValueError(f"expected <run>/checkpoints/<file> for run {run!r}, got {ckpt.name!r} elsewhere")
    source_sha = sha256_file(ckpt)
    hf_sha = sha256_file(hf_file)

    blob = _load_checkpoint(ckpt)
    try:
        if blob.get("epoch") != epoch:
            raise ValueError(f"checkpoint records epoch {blob.get('epoch')}, its file name says {epoch}")
        sd = blob["state_dict"]
        config = sanitise_config(blob["config"], forbid)
        peft = any(k.startswith(BACKBONE_PREFIX + PEFT_PREFIX) for k in sd)
        if peft != (variant != "LN"):
            raise ValueError(f"{variant} run with peft_wrapped={peft}")
        with safe_open(str(hf_file), framework="pt", device="cpu") as handle:
            kept = select_trained(sd, _LazyTensors(handle))
        kept = {k: v.detach().clone().contiguous() for k, v in kept.items()}
        groups = {k: _group(k) for k in sd}
        n_loss = sum(g == "loss_fn" for g in groups.values())
        n_from_base = sum(1 for k, g in groups.items() if g != "loss_fn" and k not in kept)
        entries = sorted(k for k in blob if k not in ("state_dict", "config", "epoch"))
    finally:
        del blob
        gc.collect()

    kept_counts = {"backbone": 0, "lora": 0, "head": 0}
    for k in kept:
        g = groups[k]
        kept_counts[g] = kept_counts.get(g, 0) + 1

    out_dir.mkdir(parents=True, exist_ok=True)
    parts = out_dir / f"{run}.safetensors"
    # one metadata entry only: safetensors writes several in an unstable order, which would change the file's sha256
    save_file(kept, str(parts), metadata={"format": "pt"})
    del kept
    gc.collect()

    sidecar = {
        "run": run,
        "variant": variant,
        "seed": seed,
        "epoch": epoch,
        "val_video_auc": val_auc,
        "source_file": f"{run}/checkpoints/{ckpt.name}",
        "source_sha256": source_sha,
        "hf_id": HF_ID,
        "hf_revision": HF_REVISION,
        "hf_sha256": hf_sha,
        "key_prefix": BACKBONE_PREFIX + (PEFT_PREFIX if peft else ""),
        "peft_wrapped": peft,
        "kept_keys": kept_counts,
        "dropped_groups": {
            "backbone_equal_to_base": n_from_base,
            "loss_fn": n_loss,
            "checkpoint_entries": entries,
        },
        "config": config,
        "parts_sha256": sha256_file(parts),
    }
    assert tuple(sidecar) == SIDECAR_KEYS
    sanitise_config(sidecar, forbid)
    return sidecar


def verify(ckpt: Path, hf_file: Path, parts: Path) -> dict:
    """Rebuild every non-``loss_fn`` tensor of ``ckpt`` from ``parts`` and the base weights and compare."""
    released = load_file(str(parts))
    blob = _load_checkpoint(Path(ckpt))
    try:
        sd = blob["state_dict"]
        keys = [k for k in sd if not k.startswith(LOSS_PREFIX)]
        n_mismatch = sum(1 for k in released if k not in sd or k.startswith(LOSS_PREFIX))
        with safe_open(str(hf_file), framework="pt", device="cpu") as handle:
            hf = _LazyTensors(handle)
            for key in keys:
                try:
                    rebuilt = _resolve(key, released, hf)
                except KeyError:
                    n_mismatch += 1
                    continue
                if not _identical(rebuilt, sd[key]):
                    n_mismatch += 1
                del rebuilt
        n_from_base = sum(1 for k in keys if k not in released)
    finally:
        del blob
        gc.collect()
    return {"n_compared": len(keys), "n_mismatch": n_mismatch, "n_backbone_equal_hf": n_from_base}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hf", type=Path, required=True, help="base weights, model.safetensors at HF_REVISION")
    ap.add_argument("--out", type=Path, required=True, help="directory for <run>.safetensors")
    ap.add_argument("--sidecars", type=Path, required=True, help="directory for <run>.json")
    ap.add_argument("--forbid", action="append", default=[], help="a string no sidecar may contain (repeatable)")
    ap.add_argument("ckpt", type=Path, nargs="+", help="<run>/checkpoints/epoch=<n>-video_auc=<auc>.ckpt")
    args = ap.parse_args(argv)

    if torch is None:
        print("This script needs torch and safetensors, which the repository environment does not provide.",
              file=sys.stderr)
        return 2
    hf_sha = sha256_file(args.hf)
    if hf_sha != HF_SHA256:
        print(f"{args.hf.name}: sha256 {hf_sha}; revision {HF_REVISION[:12]} has {HF_SHA256}", file=sys.stderr)
        return 1

    for ckpt in args.ckpt:
        run = ckpt.parent.parent.name
        sidecar = extract(ckpt, args.hf, args.out, run, tuple(args.forbid))
        gc.collect()
        result = verify(ckpt, args.hf, args.out / f"{run}.safetensors")
        gc.collect()
        expected = sum(sidecar["kept_keys"].values()) + sidecar["dropped_groups"]["backbone_equal_to_base"]
        ok = (
            result["n_mismatch"] == 0
            and result["n_compared"] == expected
            and result["n_backbone_equal_hf"] == sidecar["dropped_groups"]["backbone_equal_to_base"]
        )
        print(json.dumps({"run": run, "kept_keys": sidecar["kept_keys"],
                          "dropped": {k: v for k, v in sidecar["dropped_groups"].items() if k != "checkpoint_entries"},
                          "verify": result, "parts_sha256": sidecar["parts_sha256"], "ok": ok}))
        if not ok:
            print(f"{run}: verification failed; no sidecar written", file=sys.stderr)
            return 1
        args.sidecars.mkdir(parents=True, exist_ok=True)
        text = json.dumps(sidecar, indent=2, allow_nan=False) + "\n"
        (args.sidecars / f"{run}.json").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
