"""Released model parts: key mapping, selection of trained tensors, reassembly, the sidecar and the CLI."""

import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

import pytest

REASON = "{} is not in the repository environment; the extraction runs in the training environment"
torch = pytest.importorskip("torch", reason=REASON.format("torch"))
pytest.importorskip("safetensors", reason=REASON.format("safetensors"))
from safetensors.torch import load_file, save_file  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import extract_trained_parts as etp  # noqa: E402
from extract_trained_parts import (  # noqa: E402
    HF_ID,
    HF_REVISION,
    extract,
    hf_key,
    reassemble,
    sanitise_config,
    select_trained,
    verify,
)

BB = "model._orig_mod.backbone.model."
PEFT = BB + "base_model.model."
HEAD = "model._orig_mod.head."
SIDECAR_KEYS = {
    "run", "variant", "seed", "epoch", "val_video_auc", "source_file", "source_sha256", "hf_id", "hf_revision",
    "hf_sha256", "key_prefix", "peft_wrapped", "kept_keys", "dropped_groups", "config", "parts_sha256",
}
MACHINE = ("buildhost", "LABDRIVE")  # stand-ins for the machine and directory names passed with --forbid


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hf() -> dict:
    g = torch.Generator().manual_seed(0)
    return {
        "embeddings.cls_token": torch.randn(1, 1, 4, generator=g),
        "layer.0.norm1.weight": torch.randn(4, generator=g),
        "layer.0.norm1.bias": torch.randn(4, generator=g),
        "layer.0.attention.q_proj.weight": torch.randn(4, 4, generator=g),
        "layer.0.attention.q_proj.bias": torch.randn(4, generator=g),
    }


def _ln_sd(hf: dict) -> "OrderedDict[str, torch.Tensor]":
    sd = OrderedDict((BB + k, v.clone()) for k, v in hf.items())
    sd[BB + "layer.0.norm1.weight"] = hf["layer.0.norm1.weight"] + 0.5
    sd[BB + "layer.0.norm1.bias"] = hf["layer.0.norm1.bias"] - 0.25
    sd[HEAD + "head.1.weight"] = torch.ones(2, 4)
    sd[HEAD + "head.1.bias"] = torch.zeros(2)
    sd["loss_fn.opa.core.transform_discriminator.weight"] = torch.ones(2, 3)
    return sd


def _lora_sd(hf: dict) -> "OrderedDict[str, torch.Tensor]":
    sd = OrderedDict()
    for k, v in hf.items():
        if k.startswith("layer.0.attention.q_proj."):
            leaf = k.rsplit(".", 1)[1]
            sd[PEFT + f"layer.0.attention.q_proj.base_layer.{leaf}"] = v.clone()
        else:
            sd[PEFT + k] = v.clone()
    sd[PEFT + "layer.0.attention.q_proj.lora_A.default.weight"] = torch.full((2, 4), 0.1)
    sd[PEFT + "layer.0.attention.q_proj.lora_B.default.weight"] = torch.full((4, 2), 0.2)
    sd[HEAD + "head.0.weight"] = torch.ones(3, 4)
    sd[HEAD + "head.0.bias"] = torch.zeros(3)
    sd[HEAD + "head.3.weight"] = torch.ones(2, 3)
    sd[HEAD + "head.3.bias"] = torch.zeros(2)
    sd["loss_fn.opa.core.strategy.weights"] = torch.ones(3)
    return sd


def _without_loss(sd):
    return [k for k in sd if not k.startswith("loss_fn.")]


def test_hf_key_mapping():
    assert hf_key("model._orig_mod.backbone.model.layer.0.norm1.weight") == "layer.0.norm1.weight"
    assert hf_key(
        "model._orig_mod.backbone.model.base_model.model.layer.0.attention.q_proj.base_layer.weight"
    ) == "layer.0.attention.q_proj.weight"
    assert hf_key(
        "model._orig_mod.backbone.model.base_model.model.layer.0.attention.q_proj.base_layer.bias"
    ) == "layer.0.attention.q_proj.bias"
    assert hf_key("model._orig_mod.backbone.model.base_model.model.embeddings.cls_token") == "embeddings.cls_token"
    assert hf_key(
        "model._orig_mod.backbone.model.base_model.model.layer.0.attention.q_proj.lora_A.default.weight"
    ) is None
    assert hf_key(
        "model._orig_mod.backbone.model.base_model.model.layer.0.attention.q_proj.lora_B.default.weight"
    ) is None
    assert hf_key("model._orig_mod.head.head.1.weight") is None
    assert hf_key("loss_fn.opa.core.transform_discriminator.weight") is None


def test_select_trained_keeps_only_differences():
    hf = _hf()
    ln = _ln_sd(hf)
    kept = select_trained(ln, hf)
    assert list(kept) == [
        BB + "layer.0.norm1.weight", BB + "layer.0.norm1.bias", HEAD + "head.1.weight", HEAD + "head.1.bias"
    ]
    for k, v in kept.items():
        assert torch.equal(v, ln[k])

    lora = _lora_sd(hf)
    kept = select_trained(lora, hf)
    assert set(kept) == {
        PEFT + "layer.0.attention.q_proj.lora_A.default.weight",
        PEFT + "layer.0.attention.q_proj.lora_B.default.weight",
        HEAD + "head.0.weight", HEAD + "head.0.bias", HEAD + "head.3.weight", HEAD + "head.3.bias",
    }


def test_select_trained_keeps_dtype_or_shape_changes_and_unmatched_backbone_keys():
    hf = _hf()
    sd = OrderedDict((BB + k, v.clone()) for k, v in hf.items())
    sd[BB + "embeddings.cls_token"] = hf["embeddings.cls_token"].double()
    sd[BB + "layer.0.norm1.weight"] = hf["layer.0.norm1.weight"].reshape(2, 2)
    sd[BB + "layer.1.norm1.weight"] = torch.ones(4)
    assert set(select_trained(sd, hf)) == {
        BB + "embeddings.cls_token", BB + "layer.0.norm1.weight", BB + "layer.1.norm1.weight"
    }


def test_reassemble_round_trip():
    hf = _hf()
    for sd in (_ln_sd(hf), _lora_sd(hf)):
        keys = _without_loss(sd)
        out = reassemble(select_trained(sd, hf), hf, keys)
        assert list(out) == keys
        for k in keys:
            assert out[k].dtype == sd[k].dtype and torch.equal(out[k], sd[k]), k


def test_reassemble_refuses_a_key_it_cannot_source():
    hf = _hf()
    sd = _ln_sd(hf)
    parts = select_trained(sd, hf)
    del parts[HEAD + "head.1.bias"]
    with pytest.raises(KeyError):
        reassemble(parts, hf, _without_loss(sd))


def test_sanitise_rejects_host_paths():
    with pytest.raises(ValueError):
        sanitise_config({"dir": "/srv/x/y"})
    for bad in ("/srv/x", "dir=/opt/runs", "see /data/a", "~/runs", "C:\\runs", "D:/runs"):
        with pytest.raises(ValueError):
            sanitise_config({"a": {"b": ["fine", bad]}})
    for name in MACHINE:
        with pytest.raises(ValueError):
            sanitise_config({"a": ["fine", f"runs-on-{name}"]}, forbid=MACHINE)
    with pytest.raises(ValueError):
        sanitise_config({"/srv/x": 1})
    for fine in ("./training/logs", "configs/datasets/manifests/rgb", "facebook/dinov3", "https://example.org/a"):
        assert sanitise_config({"a": fine}, forbid=MACHINE) == {"a": fine}
    cfg = {"log_dir": "./training/logs", "lora": {"r": 8, "targets": ("q_proj", "v_proj")}, "lr": 1e-4, "x": None}
    out = sanitise_config(cfg)
    assert out == {
        "log_dir": "./training/logs", "lora": {"r": 8, "targets": ["q_proj", "v_proj"]}, "lr": 1e-4, "x": None
    }
    assert json.loads(json.dumps(out)) == out


def _synthetic_release(tmp_path: Path, run: str, name: str, epoch: int, sd) -> tuple[Path, Path]:
    hf_file = tmp_path / "hf" / "model.safetensors"
    hf_file.parent.mkdir(parents=True)
    save_file(_hf(), str(hf_file))
    ckpt = tmp_path / "runs" / run / "checkpoints" / name
    ckpt.parent.mkdir(parents=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": 100,
            "state_dict": sd,
            "callbacks": {"ModelCheckpoint": {"dirpath": "/srv/buildhost/runs/" + run}},
            "optimizer_states": [{"state": {}}],
            "lr_schedulers": [],
            "config": {"log_dir": "./training/logs", "backbone": {"weights_dir": "./pretrained/dinov3"}},
        },
        ckpt,
    )
    return ckpt, hf_file


def test_sidecar_schema(tmp_path):
    run = "orthoflow_gtl_lora_s137"
    sd = _lora_sd(_hf())
    ckpt, hf_file = _synthetic_release(tmp_path, run, "epoch=04-video_auc=0.8433.ckpt", 4, sd)
    out_dir = tmp_path / "release"
    side = extract(ckpt, hf_file, out_dir, run, forbid=MACHINE)

    assert set(side) == SIDECAR_KEYS
    assert (side["run"], side["variant"], side["seed"]) == (run, "LoRA", 137)
    assert (side["epoch"], side["val_video_auc"]) == (4, 0.8433)
    assert side["hf_id"] == HF_ID == "facebook/dinov3-vitl16-pretrain-lvd1689m"
    assert side["hf_revision"] == HF_REVISION == "ea8dc2863c51be0a264bab82070e3e8836b02d51"
    assert side["hf_sha256"] == _sha256(hf_file)
    assert side["source_sha256"] == _sha256(ckpt)
    assert side["source_file"] == f"{run}/checkpoints/epoch=04-video_auc=0.8433.ckpt"
    assert side["key_prefix"] == "model._orig_mod.backbone.model.base_model.model."
    assert side["peft_wrapped"] is True
    assert side["kept_keys"] == {"backbone": 0, "lora": 2, "head": 4}
    assert side["dropped_groups"]["backbone_equal_to_base"] == 5
    assert side["dropped_groups"]["loss_fn"] == 1
    assert "optimizer_states" in side["dropped_groups"]["checkpoint_entries"]
    assert "callbacks" in side["dropped_groups"]["checkpoint_entries"]
    assert side["config"]["backbone"]["weights_dir"] == "./pretrained/dinov3"

    parts = out_dir / f"{run}.safetensors"
    assert side["parts_sha256"] == _sha256(parts)
    assert set(load_file(str(parts))) == set(select_trained(sd, _hf()))
    text = json.dumps(side)
    assert "/srv/" not in text and "buildhost" not in text and str(tmp_path) not in text

    assert verify(ckpt, hf_file, parts) == {"n_compared": len(sd) - 1, "n_mismatch": 0, "n_backbone_equal_hf": 5}


def test_ln_sidecar_has_no_peft_wrapping(tmp_path):
    run = "orthoflow_gtl_ln_s42"
    ckpt, hf_file = _synthetic_release(tmp_path, run, "epoch=17-video_auc=0.8519.ckpt", 17, _ln_sd(_hf()))
    side = extract(ckpt, hf_file, tmp_path / "release", run)
    assert (side["variant"], side["seed"], side["epoch"], side["val_video_auc"]) == ("LN", 42, 17, 0.8519)
    assert side["key_prefix"] == "model._orig_mod.backbone.model."
    assert side["peft_wrapped"] is False
    assert side["kept_keys"] == {"backbone": 2, "lora": 0, "head": 2}


def test_extract_refuses_a_config_that_names_the_machine(tmp_path):
    run = "orthoflow_gtl_ln_s42"
    ckpt, hf_file = _synthetic_release(tmp_path, run, "epoch=17-video_auc=0.8519.ckpt", 17, _ln_sd(_hf()))
    with pytest.raises(ValueError):
        extract(ckpt, hf_file, tmp_path / "release", run, forbid=("training",))


def test_extract_refuses_an_epoch_that_disagrees_with_the_file_name(tmp_path):
    run = "orthoflow_gtl_ln_s42"
    ckpt, hf_file = _synthetic_release(tmp_path, run, "epoch=17-video_auc=0.8519.ckpt", 16, _ln_sd(_hf()))
    with pytest.raises(ValueError):
        extract(ckpt, hf_file, tmp_path / "release", run)


def test_verify_detects_a_changed_tensor(tmp_path):
    run = "orthoflow_gtl_ln_s42"
    ckpt, hf_file = _synthetic_release(tmp_path, run, "epoch=17-video_auc=0.8519.ckpt", 17, _ln_sd(_hf()))
    extract(ckpt, hf_file, tmp_path / "release", run)
    parts = tmp_path / "release" / f"{run}.safetensors"
    tensors = load_file(str(parts))
    tensors[HEAD + "head.1.bias"] = tensors[HEAD + "head.1.bias"] + 1e-7
    save_file(tensors, str(parts))
    assert verify(ckpt, hf_file, parts)["n_mismatch"] == 1


def test_cli_writes_parts_and_sidecar(tmp_path, monkeypatch):
    run = "orthoflow_gtl_gsd_lora_s1729"
    ckpt, hf_file = _synthetic_release(tmp_path, run, "epoch=07-video_auc=0.8651.ckpt", 7, _lora_sd(_hf()))
    monkeypatch.setattr(etp, "HF_SHA256", _sha256(hf_file))
    out, sidecars = tmp_path / "data" / "release", tmp_path / "checkpoints"
    assert etp.main(["--hf", str(hf_file), "--out", str(out), "--sidecars", str(sidecars), str(ckpt)]) == 0
    side = json.loads((sidecars / f"{run}.json").read_text(encoding="utf-8"))
    assert set(side) == SIDECAR_KEYS
    assert side["variant"] == "GSD-LoRA"
    assert side["parts_sha256"] == _sha256(out / f"{run}.safetensors")


def test_cli_refuses_other_base_weights(tmp_path):
    run = "orthoflow_gtl_ln_s42"
    ckpt, hf_file = _synthetic_release(tmp_path, run, "epoch=17-video_auc=0.8519.ckpt", 17, _ln_sd(_hf()))
    sidecars = tmp_path / "checkpoints"
    assert etp.main(["--hf", str(hf_file), "--out", str(tmp_path / "out"), "--sidecars", str(sidecars), str(ckpt)]) != 0
    assert not sidecars.exists() or not any(sidecars.iterdir())


def test_extract_writes_the_same_bytes_every_time(tmp_path):
    run = "orthoflow_gtl_lora_s42"
    ckpt, hf_file = _synthetic_release(tmp_path, run, "epoch=02-video_auc=0.8567.ckpt", 2, _lora_sd(_hf()))
    digests = {extract(ckpt, hf_file, tmp_path / f"release{i}", run)["parts_sha256"] for i in range(8)}
    assert len(digests) == 1
