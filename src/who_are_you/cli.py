from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from who_are_you.bigrams import PhonemeSpan, build_in_word_bigrams
from who_are_you.config import ReproductionConfig
from who_are_you.corpus import discover_speakers
from who_are_you.detector import DetectorModel, build_detector, evaluate_detector
from who_are_you.runtime import runtime_status
from who_are_you.transfer_function import recover_cross_sectional_areas, tube_length_cm


DEFAULT_ORGANIC_ROOT = Path("datasets/TIMIT")
DEFAULT_GENERATED_ROOT = Path("datasets/generated_TIMIT")
DEFAULT_MODEL_PATH = Path("artifacts/detector_model.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="who-are-you",
        description="Paper-faithful reproduction scaffold for vocal tract deepfake detection.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("show-config", help="Print the default reproduction configuration.")
    subparsers.add_parser("demo-transfer", help="Show the tract areas recovered from a constant-diameter tube.")
    subparsers.add_parser("demo-bigrams", help="Show how in-word phoneme bigrams are constructed.")
    runtime = subparsers.add_parser(
        "runtime-info",
        help="Show whether the runtime resolves to torch on CPU, CUDA, or MPS.",
    )
    runtime.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")

    summary = subparsers.add_parser("dataset-summary", help="Summarize the prepared organic and generated corpora.")
    add_dataset_args(summary)

    build = subparsers.add_parser("build-detector", help="Build and save a detector model from the local corpora.")
    add_dataset_args(build)
    add_common_build_args(build)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a saved detector model on held-out speakers.")
    add_dataset_args(evaluate)
    evaluate.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=None)
    evaluate.add_argument("--mode", choices=("ideal", "range"), default="ideal")
    evaluate.add_argument("--max-sentence-pairs", type=int, default=None)

    return parser


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--organic-root", type=Path, default=DEFAULT_ORGANIC_ROOT)
    parser.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT)


def add_common_build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--feature-speakers", type=int, default=None)
    parser.add_argument("--min-precision", type=float, default=0.9)
    parser.add_argument("--min-recall", type=float, default=0.9)
    parser.add_argument("--fft-bin-stride", type=int, default=None)
    parser.add_argument("--coordinate-search-step", type=float, default=None)
    parser.add_argument("--coordinate-search-max-iterations", type=int, default=None)
    parser.add_argument("--max-sentence-pairs", type=int, default=None)


def run_show_config() -> int:
    config = ReproductionConfig()
    print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
    return 0


def run_runtime_info(device: str) -> int:
    print(json.dumps(runtime_status(device), indent=2, sort_keys=True))
    return 0


def run_demo_transfer() -> int:
    config = ReproductionConfig()
    zero_reflections = [0.0] * (config.num_tract_segments - 1)
    areas = recover_cross_sectional_areas(
        reflection_coefficients=zero_reflections,
        initial_area_cm2=config.initial_glottis_area_cm2,
        device=config.device,
    )
    payload = {
        "num_segments": config.num_tract_segments,
        "device": config.device,
        "tube_length_cm": tube_length_cm(
            sample_rate_hz=config.sample_rate_hz,
            speed_of_sound_cm_per_s=config.speed_of_sound_cm_per_s,
        ),
        "tract_areas_cm2": areas,
    }
    print(json.dumps(payload, indent=2))
    return 0


def run_demo_bigrams() -> int:
    phonemes = [
        PhonemeSpan(symbol="k", start_sample=0, end_sample=800, word="cat"),
        PhonemeSpan(symbol="ae", start_sample=800, end_sample=2240, word="cat"),
        PhonemeSpan(symbol="t", start_sample=2240, end_sample=3040, word="cat"),
        PhonemeSpan(symbol="s", start_sample=3200, end_sample=3840, word="sat"),
        PhonemeSpan(symbol="ae", start_sample=3840, end_sample=5120, word="sat"),
        PhonemeSpan(symbol="t", start_sample=5120, end_sample=5920, word="sat"),
    ]
    bigrams = [bigram.to_dict() for bigram in build_in_word_bigrams(phonemes)]
    print(json.dumps(bigrams, indent=2))
    return 0


def run_dataset_summary(organic_root: Path, generated_root: Path) -> int:
    bundles = discover_speakers(organic_root=organic_root, generated_root=generated_root)
    by_dialect: dict[str, int] = {}
    for bundle in bundles:
        by_dialect[bundle.dialect] = by_dialect.get(bundle.dialect, 0) + 1
    payload = {
        "organic_root": str(organic_root),
        "generated_root": str(generated_root),
        "paired_speakers": len(bundles),
        "dialects": by_dialect,
        "sample_speakers": [bundle.speaker_id for bundle in bundles[:10]],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_build_detector(args: argparse.Namespace) -> int:
    config = ReproductionConfig()
    if args.device is not None:
        config.device = args.device
    if args.fft_bin_stride is not None:
        config.fft_bin_stride = args.fft_bin_stride
    if args.coordinate_search_step is not None:
        config.coordinate_search_step = args.coordinate_search_step
    if args.coordinate_search_max_iterations is not None:
        config.coordinate_search_max_iterations = args.coordinate_search_max_iterations
    if args.max_sentence_pairs is not None:
        config.max_sentence_pairs = args.max_sentence_pairs
    model = build_detector(
        organic_root=args.organic_root,
        generated_root=args.generated_root,
        config=config,
        seed=args.seed,
        sample_size=args.sample_size,
        feature_extraction_speakers=args.feature_speakers,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
    )
    model.save(args.model_path)
    payload = {
        "model_path": str(args.model_path),
        "resolved_runtime": runtime_status(config.device),
        "sampled_speakers": len(model.sampled_speakers),
        "feature_speakers": len(model.feature_speakers),
        "evaluation_speakers": len(model.evaluation_speakers),
        "organic_ranges": len(model.organic_ranges),
        "ideal_features": len(model.ideal_features),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    model = DetectorModel.load(args.model_path)
    if args.device is not None:
        model.config.device = args.device
    if args.max_sentence_pairs is not None:
        model.config.max_sentence_pairs = args.max_sentence_pairs
    result = evaluate_detector(
        organic_root=args.organic_root,
        generated_root=args.generated_root,
        model=model,
        mode=args.mode,
    )
    result["resolved_runtime"] = runtime_status(model.config.device)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command in (None, "show-config"):
        return run_show_config()
    if args.command == "demo-transfer":
        return run_demo_transfer()
    if args.command == "demo-bigrams":
        return run_demo_bigrams()
    if args.command == "runtime-info":
        return run_runtime_info(args.device)
    if args.command == "dataset-summary":
        return run_dataset_summary(args.organic_root, args.generated_root)
    if args.command == "build-detector":
        return run_build_detector(args)
    if args.command == "evaluate":
        return run_evaluate(args)

    parser.error(f"unknown command: {args.command}")
    return 2
