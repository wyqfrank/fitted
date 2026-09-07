"""Regression checks for the concrete validity failures in docs/ranker-audit.md."""

import asyncio
import io
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark_judges as benchmark
import distil_teacher as distil
from assemble_teacher_dataset import assemble, snapshot_rows
from ranker_artifact import load_scorer
from ranker_evaluation import (
    embedding_metadata,
    file_hash,
    fingerprint,
    group_split,
    load_embeddings,
    outcome,
    paired_summary,
    partition_pairs,
    validate_manifest,
    validate_teacher_cache,
    write_json,
)
from train_ranker import Comparison, fit_projection, load_teacher, project, teacher_split


def comparison(a, b):
    return Comparison(a, b, 1.0, "teacher", "vlm")


def manifest_for(records):
    manifest = {"version": 1, "images": records}
    manifest["fingerprint"] = fingerprint(manifest)
    return manifest


def record(group, split, content):
    return {"group": group, "split": split, "pixelSha256": content}


@pytest.mark.parametrize("key", ["group", "pixelSha256"])
def test_manifest_rejects_cross_split_group_or_identical_pixels(key):
    images = {
        "a": record("person-a", "train", "pixels-a"),
        "b": record("person-b", "val", "pixels-b"),
    }
    images["b"][key] = images["a"][key]
    with pytest.raises(ValueError, match="Cross-split"):
        validate_manifest(manifest_for(images))


def test_manifest_is_frozen():
    manifest = manifest_for({"a": record("a", "train", "a")})
    manifest["images"]["a"]["split"] = "test"
    with pytest.raises(ValueError, match="fingerprint"):
        validate_manifest(manifest)


def test_image_split_removes_shared_endpoints_and_is_order_invariant():
    pairs = [comparison(str(i), str(i + 1)) for i in range(300)]
    train, held = teacher_split(pairs, 0.3)
    reverse_train, reverse_held = teacher_split(list(reversed(pairs)), 0.3)

    def ids(rows):
        return {i for c in rows for i in (c.left, c.right)}

    assert train and held
    assert ids(train).isdisjoint(ids(held))
    assert len(train) + len(held) < len(pairs)
    assert ids(train) == ids(reverse_train)
    assert ids(held) == ids(reverse_held)
    assert group_split("existing") == group_split("existing")


def test_manifest_partition_discards_crossing_pairs_and_rejects_reverse_duplicates():
    manifest = manifest_for(
        {
            "a": record("a", "train", "a"),
            "b": record("b", "train", "b"),
            "c": record("c", "val", "c"),
        }
    )
    splits, dropped = partition_pairs([comparison("a", "b"), comparison("a", "c")], manifest)
    assert len(splits["train"]) == 1
    assert dropped == 1
    with pytest.raises(ValueError, match="Duplicate"):
        partition_pairs([comparison("a", "b"), comparison("b", "a")], manifest)


def test_identity_basis_really_preserves_all_coordinates():
    vectors = np.arange(24, dtype=np.float32).reshape(3, 8)
    _, truncated = fit_projection(vectors, 384)
    centre, identity = fit_projection(vectors, 384, mode="identity")
    assert truncated.shape == (3, 8)
    assert identity.shape == (8, 8)
    np.testing.assert_array_equal(
        project(vectors, centre, identity, normalize=False), vectors - vectors.mean(0)
    )


@pytest.mark.parametrize("normalize", [True, False, None])
def test_projection_matches_live_inference_and_legacy_artifacts(tmp_path, monkeypatch, normalize):
    import torch
    from transformers import AutoModel

    from fitted_inference.ranker import Dinov2FitRanker

    vector = np.array([2.0, 4.0, -3.0], dtype=np.float32)
    centre = np.array([0.5, 1.0, 0.0], dtype=np.float32)
    basis = np.eye(3, dtype=np.float32)
    weights = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    fields = {} if normalize is None else {"normalize": np.array(normalize)}
    np.savez(
        tmp_path / "ranker.npz",
        centre=centre,
        basis=basis,
        weights=weights,
        calibration=np.linspace(-30, 30, 256),
        **fields,
    )

    class Encoder:
        def eval(self):
            return self

        def to(self, device):
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace(last_hidden_state=torch.tensor(vector[None, None, :]))

    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *_a, **_k: Encoder())
    image = io.BytesIO()
    Image.new("RGB", (8, 16), "blue").save(image, format="PNG")
    runtime = Dinov2FitRanker(tmp_path, display_min=55, display_max=85)
    live = runtime.score(image.getvalue()).raw
    offline = load_scorer(tmp_path, {"image": vector})("image")
    training = (
        project(vector[None], centre, basis, normalize=True if normalize is None else normalize)[0]
        @ weights
    )
    assert live == pytest.approx(offline)
    assert live == pytest.approx(float(training))


def test_embedding_cache_refuses_unknown_or_changed_provenance(tmp_path):
    image = tmp_path / "image.bin"
    image.write_bytes(b"original content")
    images = {"a": image}
    cache = tmp_path / "embeddings.npz"
    np.savez(cache, a=np.ones(3))
    with pytest.raises(ValueError, match="Unversioned"):
        load_embeddings(cache, images, "encoder")
    _, provenance = load_embeddings(cache, images, "encoder", allow_legacy=True)
    assert provenance == "legacy-unverified"
    meta = embedding_metadata(images, "encoder")
    meta["cacheSha256"] = file_hash(cache)
    write_json(cache.with_suffix(".metadata.json"), meta)
    assert load_embeddings(cache, images, "encoder")[1] == "verified"
    image.write_bytes(b"replacement content")
    with pytest.raises(ValueError, match="mismatch"):
        load_embeddings(cache, images, "encoder", allow_legacy=True)


def test_teacher_cache_does_not_mix_prompts_or_hide_legacy_rows():
    rows = [{"margin": 3.0}]
    with pytest.raises(ValueError, match="no prompt provenance"):
        validate_teacher_cache(rows, {"prompt": "v1"})
    assert validate_teacher_cache(rows, {}, allow_legacy=True) == "legacy-unverified"
    rows = [{"margin": 3.0, "provenance": {"prompt": "v2"}}]
    with pytest.raises(ValueError, match="differs"):
        validate_teacher_cache(rows, {"prompt": "v1"}, allow_legacy=True)


def test_teacher_loader_rejects_reversed_duplicate_labels(tmp_path):
    path = tmp_path / "teacher.jsonl"
    rows = [{"leftId": a, "rightId": b, "margin": 3} for a, b in [("a", "b"), ("b", "a")]]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="Duplicate"):
        load_teacher(path, 2, allow_legacy=True)


def test_bootstrap_keeps_repeated_human_votes_together():
    pairs = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "a")]
    values = [1, 0, 1, 0, 1]
    single = paired_summary(values, pairs, draws=300)
    repeated = paired_summary(np.repeat(values, 2), [p for p in pairs for _ in range(2)], draws=300)
    assert single["ci95"] == repeated["ci95"]
    assert single["estimate"] == repeated["estimate"]
    delta = paired_summary(np.zeros(5), pairs, draws=300)
    assert delta["ci95"] == [0, 0]


def test_production_draws_are_not_forced_into_teacher_fidelity():
    assert outcome(1.999) == 0
    assert outcome(-1.999) == 0
    assert outcome(2) == 1
    assert outcome(-2) == -1
    by_pair = {("a", "b"): {"AC": True, "DP": True}, ("c", "d"): {"AC": True, "DP": True}}
    report = benchmark.summarize_judges(
        by_pair, {"a|b": 1, "c|d": 3}, lambda i: 1 if i in {"a", "c"} else 0
    )
    assert report["teacherDraws"] == 1
    assert report["teacherDecisiveCoverage"] == 0.5
    assert report["studentTeacherDecisiveAgreement"]["estimate"] == 1
    assert report["studentTeacherOutcomeAgreement"]["estimate"] == 0.5


def test_cached_only_benchmark_cannot_create_a_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "GeminiVlmProvider",
        lambda **kwargs: pytest.fail("cached-only evaluation attempted a VLM call"),
    )
    cache = tmp_path / "absent.jsonl"
    result = asyncio.run(benchmark.run_gemini([("a", "b")], 1, cache))
    assert result == {}
    assert not cache.exists()


def test_benchmark_cannot_silently_score_only_successful_requests():
    pairs = [("a", "b"), ("c", "d")]
    with pytest.raises(ValueError, match="missing 1"):
        benchmark.require_complete_judgments(pairs, {"a|b": 3.0})
    benchmark.require_complete_judgments(pairs, {"a|b": 3.0, "c|d": 0.0})


def test_model_comparison_reports_paired_deltas_and_separates_teacher_draws():
    pairs = {("a", "b"): {"AC": True, "DP": True}, ("c", "d"): {"AC": True, "DP": False}}
    report = benchmark.compare_students(
        pairs,
        {"a|b": 3, "c|d": 0},
        lambda i: float(i in {"a", "c"}),
        lambda i: float(i in {"b", "d"}),
    )
    assert report["teacherDecisiveAgreement"]["estimate"] == 1
    assert report["teacherOutcomeAgreement"]["estimate"] == 0.5
    assert report["humanAgreement"]["estimate"] == 0.5
    assert report["humanAgreement"]["rows"] == 4


def test_dotenv_defaults_are_loaded_before_argument_parsing(tmp_path, monkeypatch):
    for name in ("FITTED_VLM_MODEL", "FITTED_VLM_PROMPT_VERSION", "FITTED_VLM_MEDIA_RESOLUTION"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / "settings.env"
    env.write_text("FITTED_VLM_PROMPT_VERSION=v2\nFITTED_VLM_PROMPT_VERSION=v1\n")
    args = distil.parse_args(["--manifest", "manifest.json"], env_path=env)
    assert args.prompt_version == "v1"
    override = distil.parse_args(
        ["--manifest", "manifest.json", "--prompt-version", "v2"], env_path=env
    )
    assert override.prompt_version == "v2"


def test_pair_sampling_covers_images_before_reusing_the_prefix():
    ids = [str(i) for i in range(100)]
    pairs = distil.sample_pairs(ids, 50, 1)
    appearances = Counter(i for pair in pairs for i in pair)
    assert len(appearances) == 100
    assert set(appearances.values()) == {1}
    assert pairs == distil.sample_pairs(list(reversed(ids)), 50, 1)
    for size in (4, 5):
        complete = distil.sample_pairs(ids[:size], 100, 1)
        assert len(set(complete)) == size * (size - 1) // 2


def test_snapshot_excludes_a_partially_written_judgment(tmp_path):
    path = tmp_path / "collect.jsonl"
    path.write_bytes(b'{"margin": 3}\n{"margin":')
    rows, raw = snapshot_rows(path)
    assert rows == [{"margin": 3}]
    assert raw == b'{"margin": 3}\n'
    path.write_bytes(b'{"margin":')
    assert snapshot_rows(path) == ([], b"")


@pytest.mark.parametrize("fault", [None, "split", "hash", "judge", "incomplete"])
def test_new_teacher_dataset_enforces_matched_judge_and_frozen_splits(tmp_path, fault):
    images = {}
    for i, split in (("a", "train"), ("b", "train"), ("c", "val"), ("d", "val")):
        path = tmp_path / f"{i}.bin"
        path.write_bytes(i.encode())
        images[i] = {
            **record(i, split, i),
            "path": str(path),
            "sha256": file_hash(path),
        }
    manifest = {"version": 1, "images": images, "inputs": {}}
    manifest["fingerprint"] = fingerprint(manifest)
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    paths = []
    for split, a, b in (("train", "a", "b"), ("val", "c", "d")):
        row = {
            "leftId": a,
            "rightId": b,
            "margin": 3,
            "provenance": {"prompt": "v1"},
            "manifestFingerprint": manifest["fingerprint"],
            "imageHashes": {i: images[i]["sha256"] for i in (a, b)},
        }
        if split == "val":
            if fault == "split":
                row["leftId"] = "a"
            elif fault == "hash":
                row["imageHashes"]["c"] = "different"
            elif fault == "judge":
                row["provenance"]["prompt"] = "v2"
        path = tmp_path / f"{split}.jsonl"
        path.write_text("" if fault == "incomplete" and split == "val" else json.dumps(row) + "\n")
        paths.append(path)
    out = tmp_path / "frozen"
    if fault:
        with pytest.raises(ValueError):
            assemble(manifest_path, *paths, out, expected_val=1)
        assert not out.exists()
    else:
        report = assemble(manifest_path, *paths, out, expected_val=1)
        assert report["trainPairs"] == report["valPairs"] == 1
        assert report["testPairs"] == 0
        frozen = json.loads((out / "manifest.json").read_text())
        validate_manifest(frozen, verify_files=True)
        paths[0].write_text("collection changed after the snapshot\n")
        validate_manifest(frozen, verify_files=True)
        with pytest.raises(ValueError, match="exists"):
            assemble(manifest_path, *paths, out, expected_val=1)
