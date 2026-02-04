#!/usr/bin/env python3
"""
Convert TIMIT and generated deepfake audio dataset to Hugging Face Dataset format.

This script converts the "Who Are You" audio deepfake detection dataset to the
Hugging Face datasets format for easy use in machine learning projects.

Features:
    - Speaker-aware train/test split (prevents speaker leakage)
    - Pair ID for matching real and fake samples
    - Dialect region (DR) information for all samples
    - Complete phoneme and bigram annotations

Usage:
    python convert_to_huggingface.py \
        --real_dir ./datasets/TIMIT \
        --fake_dir ./datasets/generated_TIMIT \
        --output_dir ./hf_who_are_you \
        --push_to_hub \
        --repo_id your_username/who_are_you

Requirements:
    pip install datasets pandas numpy tqdm huggingface-hub

Author: Assistant
Date: 2026-02-04
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, DatasetDict, Features, Audio, Value, Sequence
from huggingface_hub import login


# ARPABET to IPA conversion mapping
ARPA_TO_IPA = {
    'aa': 'ɑ', 'ae': 'æ', 'ah': 'ʌ', 'ao': 'ɔ', 'aw': 'aʊ', 'ax': 'ə', 
    'axr': 'ɚ', 'ay': 'aɪ', 'eh': 'ɛ', 'er': 'ɝ', 'ey': 'eɪ', 'ih': 'ɪ',
    'ix': 'ɨ', 'iy': 'i', 'ow': 'oʊ', 'oy': 'ɔɪ', 'uh': 'ʊ', 'uw': 'u',
    'ux': 'ʉ', 'b': 'b', 'ch': 'tʃ', 'd': 'd', 'dh': 'ð', 'dx': 'ɾ',
    'el': 'l̩', 'em': 'm̩', 'en': 'n̩', 'f': 'f', 'g': 'ɡ', 'h': 'h',
    'hh': 'h', 'jh': 'dʒ', 'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n',
    'ng': 'ŋ', 'nx': 'ɾ̃', 'p': 'p', 'q': 'ʔ', 'r': 'ɹ', 's': 's',
    'sh': 'ʃ', 't': 't', 'th': 'θ', 'v': 'v', 'w': 'w', 'wh': 'ʍ',
    'y': 'j', 'z': 'z', 'zh': 'ʒ', 'ax-h': 'ə̥', 'bcl': 'b̚', 'dcl': 'd̚',
    'eng': 'ŋ̍', 'gcl': 'ɡ̚', 'hv': 'ɦ', 'kcl': 'k̚', 'pcl': 'p̚',
    'tcl': 't̚', 'pau': 'N/A', 'epi': 'N/A', 'h#': 'N/A'
}


def build_speaker_dialect_mapping(fake_dir: str) -> Dict[str, str]:
    """
    Build a mapping from speaker_id to dialect_region (DR) using fake data structure.
    
    Args:
        fake_dir: Path to generated TIMIT directory
        
    Returns:
        Dictionary mapping speaker_id -> dialect_region (e.g., "DR1")
    """
    speaker_to_dr = {}
    fake_path = Path(fake_dir)
    
    print("Building speaker to dialect region mapping...")
    for dr_dir in sorted(fake_path.glob('DR*')):
        if dr_dir.is_dir():
            dr_name = dr_dir.name  # "DR1", "DR2", etc.
            for speaker_dir in sorted(dr_dir.glob('*')):
                if speaker_dir.is_dir():
                    speaker_id = speaker_dir.name
                    speaker_to_dr[speaker_id] = dr_name
    
    print(f"  Mapped {len(speaker_to_dr)} speakers to dialect regions")
    return speaker_to_dr


def parse_phn_file(phn_path: str) -> List[Dict[str, any]]:
    """
    Parse a PHN (phoneme) file.
    
    Returns list of dicts with keys: start, end, phoneme, ipa
    """
    phonemes = []
    with open(phn_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                start = int(parts[0])
                end = int(parts[1])
                phoneme = parts[2]
                ipa = ARPA_TO_IPA.get(phoneme, phoneme)
                phonemes.append({
                    'start': start,
                    'end': end,
                    'phoneme': phoneme,
                    'ipa': ipa
                })
    return phonemes


def parse_wrd_file(wrd_path: str) -> List[Dict[str, any]]:
    """
    Parse a WRD (word) file.
    
    Returns list of dicts with keys: start, end, word
    """
    words = []
    with open(wrd_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                start = int(parts[0])
                end = int(parts[1])
                word = parts[2]
                words.append({
                    'start': start,
                    'end': end,
                    'word': word
                })
    return words


def parse_txt_file(txt_path: str) -> Tuple[int, int, str]:
    """
    Parse a TXT (text) file.
    
    Returns tuple of (start, end, text)
    """
    with open(txt_path, 'r') as f:
        line = f.readline().strip()
        parts = line.split(' ', 2)
        if len(parts) >= 3:
            return int(parts[0]), int(parts[1]), parts[2]
    return 0, 0, ""


def get_speaker_info(speaker_id: str, speaker_to_dr: Dict[str, str]) -> Dict[str, str]:
    """
    Extract speaker information from speaker ID.
    
    Args:
        speaker_id: TIMIT speaker ID
        speaker_to_dr: Mapping from speaker to dialect region
        
    Returns:
        Dictionary with speaker info
    """
    gender = 'male' if speaker_id[0].upper() == 'M' else 'female'
    dialect_region = speaker_to_dr.get(speaker_id, None)
    
    return {
        'speaker_id': speaker_id,
        'gender': gender,
        'dialect_region': dialect_region
    }


def process_timit_dataset(
    timit_dir: str, 
    is_fake: bool, 
    speaker_to_dr: Dict[str, str]
) -> List[Dict]:
    """
    Process TIMIT dataset directory structure.
    
    Args:
        timit_dir: Path to TIMIT directory
        is_fake: Whether this is generated (fake) audio
        speaker_to_dr: Mapping from speaker to dialect region
        
    Returns:
        List of record dictionaries
    """
    records = []
    timit_path = Path(timit_dir)
    
    # Determine structure: TIMIT uses flat speaker dirs, generated uses DR dirs
    if is_fake:
        # generated_TIMIT/DR1/FAEM0/...
        speaker_dirs = []
        for dr_dir in sorted(timit_path.glob('DR*')):
            if dr_dir.is_dir():
                for speaker_dir in sorted(dr_dir.glob('*')):
                    if speaker_dir.is_dir():
                        speaker_dirs.append(speaker_dir)
    else:
        # TIMIT/FAEM0/...
        speaker_dirs = [d for d in sorted(timit_path.glob('*')) if d.is_dir()]
    
    desc = f"Processing {'fake' if is_fake else 'real'} speakers"
    for speaker_dir in tqdm(speaker_dirs, desc=desc):
        speaker_id = speaker_dir.name
        speaker_info = get_speaker_info(speaker_id, speaker_to_dr)
        
        # Skip if no dialect region info available (shouldn't happen for fake)
        if is_fake and speaker_info['dialect_region'] is None:
            print(f"Warning: No DR info for speaker {speaker_id}")
        
        # Find all .wav files for this speaker
        for wav_file in sorted(speaker_dir.glob('*.wav')):
            utterance_id = wav_file.stem
            
            # Find corresponding annotation files
            phn_file = wav_file.with_suffix('.PHN')
            wrd_file = wav_file.with_suffix('.WRD')
            txt_file = wav_file.with_suffix('.TXT')
            
            if not all(f.exists() for f in [phn_file, wrd_file, txt_file]):
                continue
            
            # Parse annotations
            phonemes = parse_phn_file(str(phn_file))
            words = parse_wrd_file(str(wrd_file))
            start_sample, end_sample, text = parse_txt_file(str(txt_file))
            
            # Create bigrams (adjacent phoneme pairs)
            bigrams = []
            for i in range(len(phonemes) - 1):
                bigrams.append({
                    'phoneme1': phonemes[i]['phoneme'],
                    'phoneme2': phonemes[i + 1]['phoneme'],
                    'ipa1': phonemes[i]['ipa'],
                    'ipa2': phonemes[i + 1]['ipa'],
                    'boundary': phonemes[i]['end']
                })
            
            # Create unique pair ID for matching real and fake samples
            pair_id = f"{speaker_id}_{utterance_id}"
            
            record = {
                'audio': str(wav_file),
                'pair_id': pair_id,
                'utterance_id': utterance_id,
                'speaker_id': speaker_id,
                'gender': speaker_info['gender'],
                'dialect_region': speaker_info['dialect_region'],
                'text': text,
                'start_sample': start_sample,
                'end_sample': end_sample,
                'sample_rate': 16000,
                'is_fake': is_fake,
                'phonemes': phonemes,
                'words': words,
                'bigrams': bigrams,
                'num_phonemes': len(phonemes),
                'num_words': len(words),
            }
            records.append(record)
    
    return records


def speaker_aware_split(
    records: List[Dict], 
    test_ratio: float = 0.2, 
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split records into train/test sets based on speakers.
    
    This ensures that all samples from the same speaker are in the same split,
    preventing speaker leakage between train and test sets.
    
    Args:
        records: List of all record dictionaries
        test_ratio: Proportion of speakers to use for test set
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_records, test_records)
    """
    # Group records by speaker
    speaker_to_records = defaultdict(list)
    for record in records:
        speaker_to_records[record['speaker_id']].append(record)
    
    speakers = list(speaker_to_records.keys())
    np.random.seed(seed)
    np.random.shuffle(speakers)
    
    n_test = max(1, int(len(speakers) * test_ratio))
    test_speakers = set(speakers[:n_test])
    train_speakers = set(speakers[n_test:])
    
    # Collect records
    train_records = []
    test_records = []
    
    for speaker in speakers:
        if speaker in test_speakers:
            test_records.extend(speaker_to_records[speaker])
        else:
            train_records.extend(speaker_to_records[speaker])
    
    # Verify no overlap
    train_pairs = set(r['pair_id'] for r in train_records)
    test_pairs = set(r['pair_id'] for r in test_records)
    overlap = train_pairs & test_pairs
    
    print(f"\nSpeaker-aware split results:")
    print(f"  Total speakers: {len(speakers)}")
    print(f"  Train speakers: {len(train_speakers)}")
    print(f"  Test speakers: {len(test_speakers)}")
    print(f"  Train samples: {len(train_records)}")
    print(f"  Test samples: {len(test_records)}")
    print(f"  Pair ID overlap: {len(overlap)} (should be 0)")
    
    return train_records, test_records


def create_dataset(
    real_dir: str, 
    fake_dir: str,
    test_ratio: float = 0.2,
    seed: int = 42
) -> DatasetDict:
    """
    Create a Hugging Face Dataset from real and fake audio directories.
    
    Args:
        real_dir: Path to real TIMIT directory
        fake_dir: Path to generated fake TIMIT directory
        test_ratio: Proportion of speakers for test set
        seed: Random seed for reproducibility
        
    Returns:
        DatasetDict with train/test splits
    """
    # First, build speaker to dialect region mapping from fake data
    speaker_to_dr = build_speaker_dialect_mapping(fake_dir)
    
    print("\nProcessing real TIMIT audio...")
    real_records = process_timit_dataset(real_dir, is_fake=False, speaker_to_dr=speaker_to_dr)
    
    print("Processing generated fake audio...")
    fake_records = process_timit_dataset(fake_dir, is_fake=True, speaker_to_dr=speaker_to_dr)
    
    # Combine all records
    all_records = real_records + fake_records
    
    print(f"\nTotal samples: {len(all_records)}")
    print(f"  Real: {len(real_records)}")
    print(f"  Fake: {len(fake_records)}")
    
    # Speaker-aware split
    train_records, test_records = speaker_aware_split(all_records, test_ratio, seed)
    
    # Create datasets without specifying features (auto-infer)
    # This avoids issues with Audio feature encoding
    train_dataset = Dataset.from_list(train_records)
    test_dataset = Dataset.from_list(test_records)
    
    # Cast audio column to Audio type after creation
    train_dataset = train_dataset.cast_column('audio', Audio(sampling_rate=16000))
    test_dataset = test_dataset.cast_column('audio', Audio(sampling_rate=16000))
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset,
    })
    
    return dataset_dict


def main():
    parser = argparse.ArgumentParser(
        description='Convert TIMIT deepfake dataset to Hugging Face format'
    )
    parser.add_argument(
        '--real_dir',
        type=str,
        default='./datasets/TIMIT',
        help='Path to real TIMIT audio directory'
    )
    parser.add_argument(
        '--fake_dir',
        type=str,
        default='./datasets/generated_TIMIT',
        help='Path to generated fake audio directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./hf_who_are_you',
        help='Output directory for the Hugging Face dataset'
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.2,
        help='Proportion of speakers to use for test set (default: 0.2)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--push_to_hub',
        action='store_true',
        help='Push the dataset to Hugging Face Hub'
    )
    parser.add_argument(
        '--repo_id',
        type=str,
        default=None,
        help='Hugging Face Hub repository ID (e.g., username/dataset-name)'
    )
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='Hugging Face API token (or set HF_TOKEN env var)'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Make the repository private'
    )
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.exists(args.real_dir):
        raise ValueError(f"Real directory not found: {args.real_dir}")
    if not os.path.exists(args.fake_dir):
        raise ValueError(f"Fake directory not found: {args.fake_dir}")
    
    # Create dataset
    print("Creating Hugging Face dataset...")
    print(f"  Test ratio: {args.test_ratio}")
    print(f"  Random seed: {args.seed}")
    dataset = create_dataset(args.real_dir, args.fake_dir, args.test_ratio, args.seed)
    
    # Save locally
    print(f"\nSaving dataset to {args.output_dir}...")
    dataset.save_to_disk(args.output_dir)
    print(f"Dataset saved successfully!")
    
    # Print statistics
    print("\nFinal Dataset Statistics:")
    for split in dataset.keys():
        split_dataset = dataset[split]
        print(f"\n{split.upper()}:")
        print(f"  Samples: {len(split_dataset)}")
        
        # Count real vs fake
        df = split_dataset.to_pandas()
        real_count = len(df[df['is_fake'] == False])
        fake_count = len(df[df['is_fake'] == True])
        print(f"  Real: {real_count}, Fake: {fake_count}")
        
        # Count unique speakers
        unique_speakers = df['speaker_id'].nunique()
        print(f"  Unique speakers: {unique_speakers}")
        
        # Count pairs
        unique_pairs = df['pair_id'].nunique()
        print(f"  Unique pairs: {unique_pairs}")
        
        # Dialect regions
        dr_counts = df['dialect_region'].value_counts().to_dict()
        print(f"  Dialect regions: {dr_counts}")
    
    # Push to Hub if requested
    if args.push_to_hub:
        if args.repo_id is None:
            raise ValueError("--repo_id is required when using --push_to_hub")
        
        token = args.token or os.environ.get('HF_TOKEN')
        if token is None:
            raise ValueError(
                "Hugging Face token required. Provide via --token or HF_TOKEN env var."
            )
        
        print(f"\nPushing dataset to Hugging Face Hub: {args.repo_id}...")
        
        login(token=token)
        
        dataset.push_to_hub(
            args.repo_id,
            private=args.private,
            token=token
        )
        
        print("Dataset pushed successfully!")
        print(f"View at: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == '__main__':
    main()
