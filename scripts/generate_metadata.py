#!/usr/bin/env python3
"""
Generate metadata CSVs for TIMIT and generated_TIMIT datasets.
This script reads .wav, .PHN/.phn, and .WRD/.wrd files to create
per-phoneme timing metadata required by the handler.
"""

from pathlib import Path
import csv


ROOT = Path(".").resolve()


def read_segments(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        s, e = int(parts[0]), int(parts[1])
        label = " ".join(parts[2:])
        rows.append((s, e, label))
    return rows


def build_meta(dataset_root: Path, out_csv: Path):
    wavs = sorted(dataset_root.rglob("*.wav"))
    fieldnames = [
        "start_word", "end_word", "word",
        "sample_id", "speaker_id",
        "start_phoneme", "end_phoneme",
        "sex", "arpabet", "ipa",
        "filepath", "filename", "index_phoneme",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for wav in wavs:
            phn = wav.with_suffix(".PHN")
            wrd = wav.with_suffix(".WRD")
            if not phn.exists():
                phn = wav.with_suffix(".phn")
            if not wrd.exists():
                wrd = wav.with_suffix(".wrd")
            if not (phn.exists() and wrd.exists()):
                continue
            phns = read_segments(phn)
            wrds = read_segments(wrd)
            speaker_id = wav.parent.name
            sex = "f" if speaker_id.startswith("F") else "m"
            sample_id = f"{speaker_id}_{wav.stem}"
            filepath = str(wav.resolve())
            for ws, we, word in wrds:
                in_word = [(ps, pe, p) for (ps, pe, p) in phns if ps >= ws and pe <= we]
                for idx, (ps, pe, p) in enumerate(in_word):
                    writer.writerow({
                        "start_word": ws,
                        "end_word": we,
                        "word": word,
                        "sample_id": sample_id,
                        "speaker_id": speaker_id,
                        "start_phoneme": ps,
                        "end_phoneme": pe,
                        "sex": sex,
                        "arpabet": p,
                        "ipa": p,
                        "filepath": filepath,
                        "filename": filepath,
                        "index_phoneme": idx,
                    })


def main():
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    timit_dir = ROOT / "datasets" / "TIMIT"
    generated_dir = ROOT / "datasets" / "generated_TIMIT"
    
    print("Generating metadata for TIMIT...")
    build_meta(timit_dir, data_dir / "timit_metadata.csv")
    
    print("Generating metadata for generated_TIMIT...")
    build_meta(generated_dir, data_dir / "generated_timit_metadata.csv")
    
    print("Done!")
    print(f"  - {data_dir / 'timit_metadata.csv'}")
    print(f"  - {data_dir / 'generated_timit_metadata.csv'}")


if __name__ == "__main__":
    main()
