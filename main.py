#!/usr/bin/env python3
"""
Who Are You (USENIX Security 2022) reproduction pipeline.

This script runs the paper-style pipeline directly on:
  - datasets/TIMIT (real)
  - datasets/generated_TIMIT (fake)

It extracts vocal-tract cross-sectional area features from bigram windows,
builds organic ranges, selects ideal feature sets, evaluates, and emits plots.

Outputs (under --output_dir, default: ./outputs):
  - run.log
  - features/real/**/*.csv, features/fake/**/*.csv
  - features_agg.csv, features_exploded.csv
  - organic_ranges.csv, ideal_features.pkl
  - results.json
  - plots/*.png
"""

import argparse
import json
import logging
import os
from pathlib import Path
import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent / "core"))

CORE_IMPORT_ERROR = None
core_main = None


def load_core(logger: logging.Logger):
    global CORE_IMPORT_ERROR
    try:
        from core import core_main as _core_main
        return _core_main
    except Exception as exc:
        CORE_IMPORT_ERROR = exc
        msg = [f"core import failed: {exc}"]
        msg.append("This often means NVVM mismatch. See CUDA env setup tips.")
        logger.error("\n".join(msg))
        raise


WINDOW_SIZE = 565
OVERLAP = 115
VT_MIN_VALUE = 1e-4
VT_MAX_VALUE = 1e4

SILENCE_TOKENS = {"h#", "pau", "epi", "sp", "spn", "sil"}

DEFAULT_CONFIG = {
    "data": {
        "real_dir": "./datasets/TIMIT",
        "fake_dir": "./datasets/generated_TIMIT",
        "max_files": None,
    },
    "processing": {
        "window_size": WINDOW_SIZE,
        "overlap": OVERLAP,
        "min_windows": 1,
        "expected_sample_rate": 16000,
    },
    "evaluation": {
        "eval_speakers": 250,
        "random_seed": 42,
        "precision_target": 0.9,
        "recall_target": 0.9,
        "threshold_steps": 25,
    },
    "output": {
        "output_dir": "./outputs",
        "plots": True,
    },
}


class _ConsoleFilter(logging.Filter):
    def __init__(self, allow_prefixes: Optional[List[str]] = None):
        super().__init__()
        self.allow_prefixes = allow_prefixes or []

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if record.levelno >= logging.WARNING:
            return True
        return any(msg.startswith(prefix) for prefix in self.allow_prefixes)


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "run.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[],
        force=True,
    )

    logger = logging.getLogger("who_are_you")

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    console_handler.addFilter(
        _ConsoleFilter(
            allow_prefixes=[
                "Starting reproduction pipeline",
                "[real] audio files",
                "[fake] audio files",
                "Done",
                "RESULTS",
            ]
        )
    )
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


def iter_audio_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        name = path.name.lower()
        if name.endswith(".wav") or name.endswith(".sph"):
            yield path


def strip_audio_suffix(name: str) -> str:
    lowered = name.lower()
    for suffix in [".wav", ".sph"]:
        if lowered.endswith(suffix):
            return name[:-len(suffix)]
    return name


def find_annotation_files(audio_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    base = strip_audio_suffix(audio_path.name)
    phn = audio_path.with_name(base + ".PHN")
    wrd = audio_path.with_name(base + ".WRD")
    if not phn.exists() or not wrd.exists():
        return None, None
    return phn, wrd


def parse_phn(phn_path: Path) -> List[Dict]:
    phonemes = []
    with open(phn_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            start, end, token = int(parts[0]), int(parts[1]), parts[2].lower()
            phonemes.append({
                "start": start,
                "end": end,
                "token": token,
            })
    return phonemes


def parse_wrd(wrd_path: Path) -> List[Dict]:
    words = []
    with open(wrd_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            start, end = int(parts[0]), int(parts[1])
            word = parts[2]
            words.append({
                "start": start,
                "end": end,
                "word": word,
            })
    return words


def load_audio(audio_path: Path) -> Tuple[np.ndarray, int]:
    suffix = audio_path.suffix.lower()
    if suffix == ".sph":
        try:
            from sphfile import SPHFile
        except Exception as exc:
            raise RuntimeError("sphfile is required to read .sph files") from exc
        sph = SPHFile(str(audio_path))
        return sph.content.astype(np.float64), int(sph.format["sample_rate"])

    try:
        import soundfile as sf
        audio, sr = sf.read(str(audio_path), dtype="int16")
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.int16)
        return audio.astype(np.float64), int(sr)
    except Exception:
        try:
            from scipy.io import wavfile
            sr, audio = wavfile.read(str(audio_path))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio.astype(np.float64), int(sr)
        except Exception as exc:
            raise RuntimeError(f"Failed to load audio: {audio_path}") from exc


def construct_uniform_windows_ph(start: int, div: int, end: int, window_size: int, overlap: int) -> List[Tuple[int, int]]:
    step = window_size - overlap
    seg_len = end - start
    if seg_len <= window_size:
        return []

    center = div - window_size // 2
    center = max(center, start)
    center = min(center, end - window_size)
    windows = [(int(center), int(center + window_size))]

    left = center - step
    while left >= start:
        windows.insert(0, (int(left), int(left + window_size)))
        left -= step

    right = center + step
    while right + window_size <= end:
        windows.append((int(right), int(right + window_size)))
        right += step

    return windows


def assign_phonemes_to_words(phonemes: List[Dict], words: List[Dict]) -> List[Dict]:
    word_blocks = []
    for w in words:
        p_list = [
            p for p in phonemes
            if p["start"] >= w["start"] and p["end"] <= w["end"]
            and p["token"] not in SILENCE_TOKENS
        ]
        p_list.sort(key=lambda x: x["start"])
        if p_list:
            word_blocks.append({
                "word": w["word"],
                "start": w["start"],
                "end": w["end"],
                "phonemes": p_list,
            })
    return word_blocks


def extract_features_for_file(
    audio_path: Path,
    is_fake: bool,
    rel_path: Path,
    output_dir: Path,
    config: Dict,
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    phn_path, wrd_path = find_annotation_files(audio_path)
    if not phn_path or not wrd_path:
        logger.warning(f"Missing PHN/WRD for {audio_path}")
        return None

    audio, sr = load_audio(audio_path)
    if sr != config["processing"]["expected_sample_rate"]:
        logger.warning(f"Unexpected sample rate {sr} for {audio_path}")

    phonemes = parse_phn(phn_path)
    words = parse_wrd(wrd_path)
    word_blocks = assign_phonemes_to_words(phonemes, words)

    speaker_id = audio_path.parent.name
    sample_id = strip_audio_suffix(audio_path.name)
    sex = "m" if speaker_id.upper().startswith("M") else "f"

    rows = []
    skipped_short = 0

    for block in word_blocks:
        phs = block["phonemes"]
        for i in range(len(phs) - 1):
            first = phs[i]
            second = phs[i + 1]

            bigram_label = f"{first['token']} -- {second['token']}"
            windows = construct_uniform_windows_ph(
                start=first["start"],
                div=first["end"],
                end=second["end"],
                window_size=config["processing"]["window_size"],
                overlap=config["processing"]["overlap"],
            )
            if not windows:
                skipped_short += 1
                continue

            for win_idx, (ws, we) in enumerate(windows):
                if ws < 0 or we > len(audio):
                    continue
                core_meta = {"oper": "ext", "ph_type": "vt", "FS": sr, "sex": sex}
                try:
                    acoustic_data, _ = core_main(audio[ws:we], bigram_label, core_meta)
                except Exception:
                    logger.exception(f"Core failed: {audio_path} win[{ws}:{we}]")
                    continue

                if not acoustic_data or "cross_sect_est" not in acoustic_data:
                    continue
                vt_features = _sanitize_vt_features(acoustic_data["cross_sect_est"])
                if vt_features is None:
                    continue

                rows.append({
                    "file_id": str(rel_path),
                    "speaker_id": speaker_id,
                    "sample_id": sample_id,
                    "is_fake": is_fake,
                    "bigram_label": bigram_label,
                    "window_index": win_idx,
                    "vt_features": vt_features,
                })

    if skipped_short > 0:
        logger.debug(f"Skipped {skipped_short} short bigrams in {audio_path}")

    if not rows:
        return None

    return pd.DataFrame(rows)


def feature_cache_path(output_dir: Path, split: str, rel_path: Path) -> Path:
    return output_dir / "features" / split / rel_path.with_suffix(".csv")


def _serialize_vt_features(value) -> str:
    arr = np.asarray(value, dtype=float).tolist()
    return json.dumps(arr)


def _deserialize_vt_features(value: str) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    if pd.isna(value):
        return np.asarray([], dtype=float)
    return np.asarray(json.loads(value), dtype=float)


def _sanitize_vt_features(value) -> Optional[np.ndarray]:
    arr = _deserialize_vt_features(value)
    if arr.ndim != 1 or arr.size == 0:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    arr = np.clip(arr, VT_MIN_VALUE, VT_MAX_VALUE)
    return arr.astype(float)


def extract_features_dataset(
    root_dir: Path,
    split: str,
    is_fake: bool,
    output_dir: Path,
    config: Dict,
    logger: logging.Logger,
    force_reextract: bool = False,
) -> List[Path]:
    cached = []
    files = list(iter_audio_files(root_dir))
    if config["data"]["max_files"]:
        files = files[: config["data"]["max_files"]]
    logger.info(f"[{split}] audio files: {len(files)}")

    for audio_path in tqdm(files, desc=f"extract {split}"):
        rel_path = audio_path.relative_to(root_dir)
        cache_path = feature_cache_path(output_dir, split, rel_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists() and not force_reextract:
            cached.append(cache_path)
            continue

        df = extract_features_for_file(
            audio_path=audio_path,
            is_fake=is_fake,
            rel_path=rel_path,
            output_dir=output_dir,
            config=config,
            logger=logger,
        )
        if df is not None:
            to_save = df.copy()
            to_save["vt_features"] = to_save["vt_features"].apply(_serialize_vt_features)
            to_save.to_csv(cache_path, index=False)
            cached.append(cache_path)
        elif force_reextract and cache_path.exists():
            cache_path.unlink()

    return cached


def collect_cached_features(output_dir: Path) -> pd.DataFrame:
    feature_files = list((output_dir / "features").rglob("*.csv"))
    dfs = []
    for p in tqdm(feature_files, desc="load features"):
        df = pd.read_csv(p)
        if "vt_features" in df.columns:
            df["vt_features"] = df["vt_features"].apply(_sanitize_vt_features)
            df = df[df["vt_features"].notna()]
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def explode_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="explode features"):
        vt = _sanitize_vt_features(row["vt_features"])
        if vt is None:
            continue
        for tube_idx, value in enumerate(vt):
            rows.append({
                "file_id": row["file_id"],
                "speaker_id": row["speaker_id"],
                "is_fake": row["is_fake"],
                "bigram_label": row["bigram_label"],
                "window_index": row["window_index"],
                "tube_idx": tube_idx,
                "value": float(value),
            })
    return pd.DataFrame(rows)


def compute_organic_ranges(exploded_df: pd.DataFrame) -> pd.DataFrame:
    real_df = exploded_df[exploded_df["is_fake"] == False]
    ranges = (
        real_df
        .groupby(["bigram_label", "window_index", "tube_idx"])["value"]
        .agg(["min", "max", "mean", "std", "count"])
        .reset_index()
    )
    return ranges


def split_by_speaker(df: pd.DataFrame, eval_speakers: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    speakers = sorted(df["speaker_id"].unique().tolist())
    random.seed(seed)
    eval_count = min(eval_speakers, len(speakers))
    eval_set = set(random.sample(speakers, eval_count))
    df_eval = df[df["speaker_id"].isin(eval_set)]
    df_train = df[~df["speaker_id"].isin(eval_set)]
    return df_train, df_eval


def scan_threshold(y_true: np.ndarray, scores: np.ndarray, precision_target: float, recall_target: float, steps: int) -> float:
    if scores.size == 0:
        return 0.0
    best_threshold = None
    best_f1 = -1.0
    thresholds = np.linspace(scores.min(), scores.max(), steps)
    for t in thresholds:
        y_pred = scores > t
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision >= precision_target and recall >= recall_target:
            return t
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    return float(best_threshold) if best_threshold is not None else float(scores.max())


def compute_violation_ratio(exploded_df: pd.DataFrame, ranges_df: pd.DataFrame) -> pd.DataFrame:
    merged = exploded_df.merge(
        ranges_df,
        on=["bigram_label", "window_index", "tube_idx"],
        how="inner",
    )
    merged["out_of_range"] = (merged["value"] < merged["min"]) | (merged["value"] > merged["max"])
    ratios = (
        merged.groupby(["file_id", "speaker_id", "is_fake"])["out_of_range"]
        .mean()
        .reset_index()
        .rename(columns={"out_of_range": "violation_ratio"})
    )
    return ratios


def build_ideal_features(
    exploded_df: pd.DataFrame,
    precision_target: float,
    recall_target: float,
    steps: int,
) -> pd.DataFrame:
    rows = []
    grouped = exploded_df.groupby(["bigram_label", "window_index", "tube_idx"])
    for key, group in tqdm(grouped, desc="ideal feature search"):
        real_vals = group[group["is_fake"] == False]["value"].values
        fake_vals = group[group["is_fake"] == True]["value"].values
        if len(real_vals) < 5 or len(fake_vals) < 5:
            continue

        values = np.concatenate([real_vals, fake_vals])
        thresholds = np.linspace(values.min(), values.max(), steps)
        best = None

        for direction in ("gt", "lt"):
            for t in thresholds:
                if direction == "gt":
                    y_pred = group["value"].values > t
                else:
                    y_pred = group["value"].values < t
                y_true = group["is_fake"].values.astype(int)
                tp = np.sum((y_true == 1) & (y_pred == 1))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                fn = np.sum((y_true == 1) & (y_pred == 0))
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                if precision >= precision_target and recall >= recall_target:
                    best = {
                        "bigram_label": key[0],
                        "window_index": key[1],
                        "tube_idx": key[2],
                        "threshold": float(t),
                        "direction": direction,
                        "precision": float(precision),
                        "recall": float(recall),
                        "support": int(len(group)),
                    }
                    break
            if best is not None:
                break

        if best is not None:
            rows.append(best)

    return pd.DataFrame(rows)


def evaluate_from_votes(votes_df: pd.DataFrame) -> Dict:
    tp = np.sum((votes_df["true_label"] == 1) & (votes_df["pred_label"] == 1))
    tn = np.sum((votes_df["true_label"] == 0) & (votes_df["pred_label"] == 0))
    fp = np.sum((votes_df["true_label"] == 0) & (votes_df["pred_label"] == 1))
    fn = np.sum((votes_df["true_label"] == 1) & (votes_df["pred_label"] == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(votes_df) if len(votes_df) > 0 else 0.0
    return {
        "total": int(len(votes_df)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def evaluate_non_optimized(
    exploded_df: pd.DataFrame,
    ranges_df: pd.DataFrame,
    config: Dict,
) -> Dict:
    ratios = compute_violation_ratio(exploded_df, ranges_df)
    if ratios.empty:
        return {"error": "no overlapping ranges for evaluation"}, ratios
    ratios_train, ratios_eval = split_by_speaker(
        ratios,
        eval_speakers=config["evaluation"]["eval_speakers"],
        seed=config["evaluation"]["random_seed"],
    )
    if ratios_train.empty:
        ratios_train = ratios
        ratios_eval = ratios
    y_true = ratios_train["is_fake"].values.astype(int)
    scores = ratios_train["violation_ratio"].values
    threshold = scan_threshold(
        y_true,
        scores,
        precision_target=config["evaluation"]["precision_target"],
        recall_target=config["evaluation"]["recall_target"],
        steps=config["evaluation"]["threshold_steps"],
    )

    ratios_eval["pred_label"] = (ratios_eval["violation_ratio"] > threshold).astype(int)
    ratios_eval["true_label"] = ratios_eval["is_fake"].astype(int)
    metrics = evaluate_from_votes(ratios_eval[["true_label", "pred_label"]])
    metrics["threshold"] = float(threshold)
    return metrics, ratios_eval


def evaluate_ideal_features(
    exploded_df: pd.DataFrame,
    ideal_df: pd.DataFrame,
    config: Dict,
) -> Dict:
    if ideal_df is None or ideal_df.empty or "bigram_label" not in ideal_df.columns:
        return {"error": "no ideal features found"}, None
    merged = exploded_df.merge(
        ideal_df,
        on=["bigram_label", "window_index", "tube_idx"],
        how="inner",
    )
    if merged.empty:
        return {"error": "no ideal features found"}, None

    def classify_row(row):
        if row["direction"] == "gt":
            return int(row["value"] > row["threshold"])
        return int(row["value"] < row["threshold"])

    merged["pred"] = merged.apply(classify_row, axis=1)
    votes = (
        merged.groupby(["file_id", "speaker_id", "is_fake"])["pred"]
        .mean()
        .reset_index()
    )
    votes["pred_label"] = (votes["pred"] >= 0.5).astype(int)
    votes["true_label"] = votes["is_fake"].astype(int)

    votes_train, votes_eval = split_by_speaker(
        votes,
        eval_speakers=config["evaluation"]["eval_speakers"],
        seed=config["evaluation"]["random_seed"],
    )
    metrics = evaluate_from_votes(votes_eval[["true_label", "pred_label"]])
    return metrics, votes_eval


def plot_artifacts(
    output_dir: Path,
    exploded_df: pd.DataFrame,
    ratios_eval: Optional[pd.DataFrame],
    ideal_df: pd.DataFrame,
    top_k: int = 4,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) Window count distribution
    window_counts = (
        exploded_df.groupby(["bigram_label", "window_index"])["value"]
        .count()
        .reset_index()
    )
    plt.figure(figsize=(8, 4))
    plt.hist(window_counts["window_index"].values, bins=20, color="#4c78a8")
    plt.title("Window index distribution")
    plt.xlabel("window_index")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(plots_dir / "window_index_distribution.png")
    plt.close()

    # 2) Violation ratio histogram (real vs fake)
    if ratios_eval is not None and not ratios_eval.empty:
        plt.figure(figsize=(8, 4))
        real_vals = ratios_eval[ratios_eval["is_fake"] == False]["violation_ratio"].values
        fake_vals = ratios_eval[ratios_eval["is_fake"] == True]["violation_ratio"].values
        plt.hist(real_vals, bins=40, alpha=0.6, label="real")
        plt.hist(fake_vals, bins=40, alpha=0.6, label="fake")
        plt.title("Violation ratio distribution (eval)")
        plt.xlabel("violation_ratio")
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "violation_ratio_hist.png")
        plt.close()

    # 3) Top-K ideal features distribution plots
    if ideal_df is not None and not ideal_df.empty:
        top = ideal_df.sort_values(["precision", "recall"], ascending=False).head(top_k)
        for _, row in top.iterrows():
            subset = exploded_df[
                (exploded_df["bigram_label"] == row["bigram_label"]) &
                (exploded_df["window_index"] == row["window_index"]) &
                (exploded_df["tube_idx"] == row["tube_idx"])
            ]
            real_vals = subset[subset["is_fake"] == False]["value"].values
            fake_vals = subset[subset["is_fake"] == True]["value"].values
            if len(real_vals) == 0 or len(fake_vals) == 0:
                continue
            plt.figure(figsize=(8, 4))
            plt.hist(real_vals, bins=40, alpha=0.6, label="real")
            plt.hist(fake_vals, bins=40, alpha=0.6, label="fake")
            title = f"{row['bigram_label']} | win={row['window_index']} | tube={row['tube_idx']}"
            plt.title(title)
            plt.xlabel("cross-sectional area estimate")
            plt.ylabel("count")
            plt.legend()
            plt.tight_layout()
            out_name = f"dist_{row['bigram_label'].replace(' ', '_')}_w{row['window_index']}_t{row['tube_idx']}.png"
            plt.savefig(plots_dir / out_name)
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Who Are You reproduction pipeline")
    parser.add_argument("--real_dir", default=DEFAULT_CONFIG["data"]["real_dir"])
    parser.add_argument("--fake_dir", default=DEFAULT_CONFIG["data"]["fake_dir"])
    parser.add_argument("--output_dir", default=DEFAULT_CONFIG["output"]["output_dir"])
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--eval_speakers", type=int, default=DEFAULT_CONFIG["evaluation"]["eval_speakers"])
    parser.add_argument("--precision_target", type=float, default=DEFAULT_CONFIG["evaluation"]["precision_target"])
    parser.add_argument("--recall_target", type=float, default=DEFAULT_CONFIG["evaluation"]["recall_target"])
    parser.add_argument("--threshold_steps", type=int, default=DEFAULT_CONFIG["evaluation"]["threshold_steps"])
    parser.add_argument("--force_reextract", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    logger = setup_logger(output_dir)
    logger.info("Starting reproduction pipeline")

    try:
        global core_main
        core_main = load_core(logger)
    except Exception:
        msg = [f"core import failed: {CORE_IMPORT_ERROR}"]
        if "No module named" in str(CORE_IMPORT_ERROR):
            msg.append("Install missing dependencies (e.g., scikit-learn) or re-run with `uv run --refresh`.")
        msg.extend([
            "If this is a CUDA/numba/llvmlite mismatch:",
            "Recommended: Python 3.12 + numba 0.59 + CUDA toolkit 11.8+.",
            "Make sure NUMBA_CUDA_NVVM points to CUDA 12.x (e.g., /usr/local/cuda-12.4/nvvm/lib64/libnvvm.so).",
        ])
        raise RuntimeError("\n".join(msg))

    config = DEFAULT_CONFIG.copy()
    config["data"]["real_dir"] = args.real_dir
    config["data"]["fake_dir"] = args.fake_dir
    config["data"]["max_files"] = args.max_files
    config["evaluation"]["eval_speakers"] = args.eval_speakers
    config["evaluation"]["precision_target"] = args.precision_target
    config["evaluation"]["recall_target"] = args.recall_target
    config["evaluation"]["threshold_steps"] = args.threshold_steps

    logger.info(json.dumps(config, indent=2))

    real_dir = Path(args.real_dir)
    fake_dir = Path(args.fake_dir)
    if not real_dir.exists() or not fake_dir.exists():
        raise RuntimeError("Dataset dirs not found")

    extract_features_dataset(
        real_dir, "real", False, output_dir, config, logger, force_reextract=args.force_reextract
    )
    extract_features_dataset(
        fake_dir, "fake", True, output_dir, config, logger, force_reextract=args.force_reextract
    )

    features_df = collect_cached_features(output_dir)
    if features_df.empty:
        raise RuntimeError("No features extracted")
    features_df_to_save = features_df.copy()
    if "vt_features" in features_df_to_save.columns:
        features_df_to_save["vt_features"] = features_df_to_save["vt_features"].apply(_serialize_vt_features)
    features_df_to_save.to_csv(output_dir / "features_agg.csv", index=False)

    exploded_df = explode_features(features_df)
    exploded_df.to_csv(output_dir / "features_exploded.csv", index=False)

    ranges_df = compute_organic_ranges(exploded_df)
    ranges_df.to_csv(output_dir / "organic_ranges.csv", index=False)

    nonopt_metrics, ratios_eval = evaluate_non_optimized(exploded_df, ranges_df, config)

    ideal_df = build_ideal_features(
        exploded_df,
        precision_target=config["evaluation"]["precision_target"],
        recall_target=config["evaluation"]["recall_target"],
        steps=config["evaluation"]["threshold_steps"],
    )
    ideal_df.to_pickle(output_dir / "ideal_features.pkl")
    ideal_metrics, ideal_votes = evaluate_ideal_features(exploded_df, ideal_df, config)

    results = {
        "non_optimized": nonopt_metrics,
        "ideal_features": ideal_metrics,
        "counts": {
            "features_rows": int(len(features_df)),
            "exploded_rows": int(len(exploded_df)),
            "ideal_feature_count": int(len(ideal_df)),
        },
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    if config["output"]["plots"]:
        plot_artifacts(output_dir, exploded_df, ratios_eval, ideal_df)

    logger.info("Done")
    logger.info("RESULTS\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
