#!/usr/bin/env python3
"""
Reproduction diagnostics for Who Are You pipeline outputs.

Reads files under an output directory (default: ./outputs) and writes a
diagnostic report under <output_dir>/diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate diagnostics for reproduction outputs")
    parser.add_argument("--output_dir", default="./outputs", help="Directory with pipeline outputs")
    parser.add_argument(
        "--report_dir",
        default=None,
        help="Directory to write diagnostic artifacts (default: <output_dir>/diagnostics)",
    )
    parser.add_argument("--edge_min", type=float, default=1e-4, help="Lower edge bound for diagnostics")
    parser.add_argument("--edge_max", type=float, default=1e4, help="Upper edge bound for diagnostics")
    parser.add_argument("--chunksize", type=int, default=50000, help="CSV chunksize")
    return parser.parse_args()


def _safe_quantiles(values: List[float], probs: List[float]) -> Dict[str, float]:
    if not values:
        return {str(p): float("nan") for p in probs}
    arr = np.asarray(values, dtype=float)
    return {str(p): float(np.quantile(arr, p)) for p in probs}


def analyze_features_agg(
    features_agg_path: Path,
    edge_min: float,
    edge_max: float,
    chunksize: int,
) -> Dict:
    summary = {
        "rows": 0,
        "parse_errors": 0,
        "nonfinite_windows": 0,
        "nonpositive_windows": 0,
        "windows_with_any_edge": 0,
        "windows_edge_ge_0_2": 0,
        "windows_edge_ge_0_4": 0,
        "windows_edge_ge_0_6": 0,
        "windows_edge_eq_1_0": 0,
        "edge_ratio_quantiles": {},
    }
    edge_ratios: List[float] = []

    real_speakers = set()
    fake_speakers = set()
    real_files = set()
    fake_files = set()
    windows_real = 0
    windows_fake = 0

    usecols = ["file_id", "speaker_id", "is_fake", "vt_features"]
    for chunk in pd.read_csv(features_agg_path, usecols=usecols, chunksize=chunksize):
        summary["rows"] += int(len(chunk))

        for _, row in chunk.iterrows():
            is_fake = bool(row["is_fake"])
            speaker_id = str(row["speaker_id"])
            file_id = str(row["file_id"])
            if is_fake:
                fake_speakers.add(speaker_id)
                fake_files.add(file_id)
                windows_fake += 1
            else:
                real_speakers.add(speaker_id)
                real_files.add(file_id)
                windows_real += 1

            try:
                values = json.loads(row["vt_features"])
                arr = np.asarray(values, dtype=float)
            except Exception:
                summary["parse_errors"] += 1
                continue

            if arr.ndim != 1 or arr.size == 0:
                summary["parse_errors"] += 1
                continue
            if not np.all(np.isfinite(arr)):
                summary["nonfinite_windows"] += 1
                continue

            if np.any(arr <= 0):
                summary["nonpositive_windows"] += 1

            edge_ratio = float(np.mean((arr <= edge_min) | (arr >= edge_max)))
            edge_ratios.append(edge_ratio)

            if edge_ratio > 0:
                summary["windows_with_any_edge"] += 1
            if edge_ratio >= 0.2:
                summary["windows_edge_ge_0_2"] += 1
            if edge_ratio >= 0.4:
                summary["windows_edge_ge_0_4"] += 1
            if edge_ratio >= 0.6:
                summary["windows_edge_ge_0_6"] += 1
            if edge_ratio == 1.0:
                summary["windows_edge_eq_1_0"] += 1

    if edge_ratios:
        summary["edge_ratio_quantiles"] = _safe_quantiles(edge_ratios, [0.5, 0.9, 0.95, 0.99])

    speaker_coverage = {
        "real_speakers": len(real_speakers),
        "fake_speakers": len(fake_speakers),
        "speaker_overlap": len(real_speakers & fake_speakers),
        "real_files": len(real_files),
        "fake_files": len(fake_files),
        "windows_real": windows_real,
        "windows_fake": windows_fake,
    }
    return {"feature_summary": summary, "speaker_coverage": speaker_coverage}


def analyze_ideal_hits(
    exploded_path: Path,
    ideal_path: Path,
    chunksize: int,
) -> Dict:
    if not exploded_path.exists() or not ideal_path.exists():
        return {
            "status": "skipped",
            "reason": "missing features_exploded.csv or ideal_features.pkl",
            "per_file": None,
            "per_feature": None,
            "summary": {},
        }

    ideal_df = pd.read_pickle(ideal_path)
    if ideal_df is None or ideal_df.empty:
        return {
            "status": "skipped",
            "reason": "ideal_features.pkl is empty",
            "per_file": None,
            "per_feature": None,
            "summary": {},
        }

    keys = ["bigram_label", "window_index", "tube_idx"]
    keep_cols = keys + ["threshold"]
    if "direction" in ideal_df.columns:
        keep_cols.append("direction")
    ideal_df = ideal_df[keep_cols].copy()
    if "direction" not in ideal_df.columns:
        ideal_df["direction"] = "gt"

    file_parts = []
    feature_parts = []

    usecols = ["file_id", "is_fake", "bigram_label", "window_index", "tube_idx", "value"]
    for chunk in pd.read_csv(exploded_path, usecols=usecols, chunksize=chunksize):
        merged = chunk.merge(ideal_df, on=keys, how="inner")
        if merged.empty:
            continue

        direction = merged["direction"].fillna("gt")
        merged["hit"] = np.where(
            direction == "gt",
            merged["value"] > merged["threshold"],
            merged["value"] < merged["threshold"],
        ).astype(np.int8)

        file_part = (
            merged.groupby(["file_id", "is_fake"], as_index=False)
            .agg(hit_sum=("hit", "sum"), hit_count=("hit", "count"))
        )
        feature_part = (
            merged.groupby(keys + ["is_fake"], as_index=False)
            .agg(hit_sum=("hit", "sum"), hit_count=("hit", "count"))
        )
        file_parts.append(file_part)
        feature_parts.append(feature_part)

    if not file_parts:
        return {
            "status": "skipped",
            "reason": "no overlap between ideal features and exploded features",
            "per_file": None,
            "per_feature": None,
            "summary": {},
        }

    per_file = (
        pd.concat(file_parts, ignore_index=True)
        .groupby(["file_id", "is_fake"], as_index=False)[["hit_sum", "hit_count"]]
        .sum()
    )
    per_file["hit_ratio"] = per_file["hit_sum"] / per_file["hit_count"]

    per_feature = (
        pd.concat(feature_parts, ignore_index=True)
        .groupby(keys + ["is_fake"], as_index=False)[["hit_sum", "hit_count"]]
        .sum()
    )
    per_feature["hit_rate"] = per_feature["hit_sum"] / per_feature["hit_count"]

    real_ratios = per_file.loc[per_file["is_fake"] == False, "hit_ratio"].tolist()
    fake_ratios = per_file.loc[per_file["is_fake"] == True, "hit_ratio"].tolist()

    real_feature = per_feature[per_feature["is_fake"] == False][keys + ["hit_rate"]].rename(
        columns={"hit_rate": "real_hit_rate"}
    )
    fake_feature = per_feature[per_feature["is_fake"] == True][keys + ["hit_rate"]].rename(
        columns={"hit_rate": "fake_hit_rate"}
    )
    feature_effect = real_feature.merge(fake_feature, on=keys, how="outer").fillna(0.0)
    feature_effect["gap_fake_minus_real"] = feature_effect["fake_hit_rate"] - feature_effect["real_hit_rate"]
    feature_effect = feature_effect.sort_values("gap_fake_minus_real", ascending=False).reset_index(drop=True)

    summary = {
        "status": "ok",
        "per_file_count": int(len(per_file)),
        "per_feature_count": int(len(feature_effect)),
        "real_hit_ratio_quantiles": _safe_quantiles(real_ratios, [0.5, 0.9, 0.95, 0.99]),
        "fake_hit_ratio_quantiles": _safe_quantiles(fake_ratios, [0.5, 0.9, 0.95, 0.99]),
    }

    return {
        "status": "ok",
        "reason": "",
        "per_file": per_file,
        "per_feature": feature_effect,
        "summary": summary,
    }


def parse_run_log(run_log_path: Path) -> Dict:
    info = {
        "missing_phn_wrd_count": 0,
        "missing_examples": [],
    }
    if not run_log_path.exists():
        return info

    for line in run_log_path.read_text(errors="ignore").splitlines():
        if "Missing PHN/WRD for" in line:
            info["missing_phn_wrd_count"] += 1
            if len(info["missing_examples"]) < 20:
                info["missing_examples"].append(line.strip())
    return info


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    report_dir = Path(args.report_dir).resolve() if args.report_dir else (output_dir / "diagnostics")
    report_dir.mkdir(parents=True, exist_ok=True)

    features_agg_path = output_dir / "features_agg.csv"
    exploded_path = output_dir / "features_exploded.csv"
    ideal_path = output_dir / "ideal_features.pkl"
    results_path = output_dir / "results.json"
    run_log_path = output_dir / "run.log"

    report = {
        "inputs": {
            "output_dir": str(output_dir),
            "edge_min": args.edge_min,
            "edge_max": args.edge_max,
            "chunksize": args.chunksize,
        },
        "results_json": {},
        "feature_summary": {},
        "speaker_coverage": {},
        "ideal_hit_summary": {},
        "run_log": {},
        "warnings": [],
    }

    if results_path.exists():
        report["results_json"] = json.loads(results_path.read_text())
    else:
        report["warnings"].append("results.json not found")

    if features_agg_path.exists():
        feature_res = analyze_features_agg(
            features_agg_path=features_agg_path,
            edge_min=args.edge_min,
            edge_max=args.edge_max,
            chunksize=args.chunksize,
        )
        report["feature_summary"] = feature_res["feature_summary"]
        report["speaker_coverage"] = feature_res["speaker_coverage"]
    else:
        report["warnings"].append("features_agg.csv not found")

    ideal_res = analyze_ideal_hits(
        exploded_path=exploded_path,
        ideal_path=ideal_path,
        chunksize=max(100000, args.chunksize),
    )
    report["ideal_hit_summary"] = ideal_res["summary"] if ideal_res["summary"] else {
        "status": ideal_res["status"],
        "reason": ideal_res["reason"],
    }

    if isinstance(ideal_res.get("per_file"), pd.DataFrame):
        ideal_res["per_file"].to_csv(report_dir / "ideal_hit_by_file.csv", index=False)
    if isinstance(ideal_res.get("per_feature"), pd.DataFrame):
        ideal_res["per_feature"].to_csv(report_dir / "ideal_feature_effect.csv", index=False)

    report["run_log"] = parse_run_log(run_log_path)

    report_path = report_dir / "diagnostics_summary.json"
    report_path.write_text(json.dumps(report, indent=2))

    speaker_df = pd.DataFrame([report["speaker_coverage"]]) if report["speaker_coverage"] else pd.DataFrame()
    if not speaker_df.empty:
        speaker_df.to_csv(report_dir / "speaker_coverage.csv", index=False)

    print(f"Diagnostics written to: {report_dir}")
    print(f"Summary: {report_path}")


if __name__ == "__main__":
    main()

