#!/usr/bin/env python3
"""
Usage examples for the "Who Are You" Hugging Face Dataset.

This script demonstrates common use cases for the dataset.
"""

from datasets import load_from_disk, load_dataset
import numpy as np
from collections import defaultdict


def example_1_basic_loading():
    """Example 1: Load and inspect the dataset."""
    print("=" * 60)
    print("Example 1: Basic Loading")
    print("=" * 60)
    
    # Load from local directory
    dataset = load_from_disk('../hf_who_are_you')
    
    print(f"\nDataset splits: {list(dataset.keys())}")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    # Show a sample
    sample = dataset['train'][0]
    print(f"\nSample keys: {list(sample.keys())}")
    print(f"  pair_id: {sample['pair_id']}")
    print(f"  speaker_id: {sample['speaker_id']}")
    print(f"  utterance_id: {sample['utterance_id']}")
    print(f"  is_fake: {sample['is_fake']}")
    print(f"  gender: {sample['gender']}")
    print(f"  dialect_region: {sample['dialect_region']}")
    print(f"  text: {sample['text'][:50]}...")
    print(f"  num_phonemes: {sample['num_phonemes']}")
    print(f"  num_words: {sample['num_words']}")


def example_2_pair_matching():
    """Example 2: Find matching real and fake pairs."""
    print("\n" + "=" * 60)
    print("Example 2: Pair Matching")
    print("=" * 60)
    
    dataset = load_from_disk('../hf_who_are_you')
    
    # Group by pair_id
    pairs = defaultdict(lambda: {'real': None, 'fake': None})
    
    for sample in dataset['train']:
        pid = sample['pair_id']
        if sample['is_fake']:
            pairs[pid]['fake'] = sample
        else:
            pairs[pid]['real'] = sample
    
    # Find complete pairs
    complete_pairs = [pid for pid, data in pairs.items() 
                      if data['real'] is not None and data['fake'] is not None]
    
    print(f"\nTotal unique pair_ids: {len(pairs)}")
    print(f"Complete pairs (both real & fake): {len(complete_pairs)}")
    
    # Show an example pair
    example_pid = complete_pairs[0]
    print(f"\nExample pair: {example_pid}")
    print(f"  Real audio shape: {pairs[example_pid]['real']['audio']['array'].shape}")
    print(f"  Fake audio shape: {pairs[example_pid]['fake']['audio']['array'].shape}")
    print(f"  Text: {pairs[example_pid]['real']['text'][:50]}...")


def example_3_speaker_analysis():
    """Example 3: Analyze speakers and dialect regions."""
    print("\n" + "=" * 60)
    print("Example 3: Speaker Analysis")
    print("=" * 60)
    
    dataset = load_from_disk('../hf_who_are_you')
    
    # Convert to pandas for easier analysis
    df = dataset['train'].to_pandas()
    
    # Gender distribution
    print("\nGender distribution:")
    gender_counts = df['gender'].value_counts()
    print(gender_counts)
    
    # Dialect region distribution
    print("\nDialect region distribution:")
    dr_counts = df['dialect_region'].value_counts().sort_index()
    print(dr_counts)
    
    # Real vs Fake by dialect region
    print("\nReal vs Fake by dialect region:")
    cross_tab = pd.crosstab(df['dialect_region'], df['is_fake'], 
                            rownames=['DR'], colnames=['Is Fake'])
    print(cross_tab)


def example_4_filter_by_dialect():
    """Example 4: Filter samples by dialect region."""
    print("\n" + "=" * 60)
    print("Example 4: Filter by Dialect Region")
    print("=" * 60)
    
    dataset = load_from_disk('../hf_who_are_you')
    
    # Filter to only DR1 (New England)
    dr1_samples = dataset['train'].filter(lambda x: x['dialect_region'] == 'DR1')
    print(f"\nDR1 samples: {len(dr1_samples)}")
    
    # Filter to only female speakers in test set
    female_test = dataset['test'].filter(
        lambda x: x['gender'] == 'female' and not x['is_fake']
    )
    print(f"Female real speakers in test: {len(female_test)}")


def example_5_phoneme_bigram_analysis():
    """Example 5: Analyze phonemes and bigrams (for detector input)."""
    print("\n" + "=" * 60)
    print("Example 5: Phoneme and Bigram Analysis")
    print("=" * 60)
    
    dataset = load_from_disk('../hf_who_are_you')
    sample = dataset['train'][0]
    
    print(f"\nSample: {sample['pair_id']}")
    print(f"Text: {sample['text']}")
    
    # Show phonemes
    print(f"\nPhonemes ({len(sample['phonemes'])} total):")
    for ph in sample['phonemes'][:5]:
        print(f"  {ph['phoneme']} ({ph['ipa']}): {ph['start']} - {ph['end']}")
    print("  ...")
    
    # Show bigrams
    print(f"\nBigrams ({len(sample['bigrams'])} total):")
    for bg in sample['bigrams'][:5]:
        label = f"{bg['ipa1']} -- {bg['ipa2']}"
        print(f"  {label}: boundary at {bg['boundary']}")
    print("  ...")


def example_6_extract_for_detector():
    """Example 6: Extract audio windows for the original detector."""
    print("\n" + "=" * 60)
    print("Example 6: Extract for Original Detector")
    print("=" * 60)
    
    dataset = load_from_disk('../hf_who_are_you')
    
    WINDOW_SIZE = 565
    
    def extract_bigram_windows(sample):
        """Extract audio windows around bigram boundaries."""
        audio = sample['audio']['array']
        bigrams = sample['bigrams']
        
        windows = []
        labels = []
        
        for bigram in bigrams:
            boundary = bigram['boundary']
            label = f"{bigram['ipa1']} -- {bigram['ipa2']}"
            
            start = boundary - WINDOW_SIZE // 2
            end = start + WINDOW_SIZE
            
            if start >= 0 and end <= len(audio):
                windows.append(audio[start:end])
                labels.append(label)
        
        return windows, labels
    
    # Process a sample
    sample = dataset['train'][0]
    windows, labels = extract_bigram_windows(sample)
    
    print(f"\nSample: {sample['pair_id']}")
    print(f"Extracted {len(windows)} windows")
    print(f"First 5 bigram labels: {labels[:5]}")
    print(f"Window shape: {windows[0].shape if windows else None}")


def example_7_verify_split():
    """Example 7: Verify no speaker leakage between splits."""
    print("\n" + "=" * 60)
    print("Example 7: Verify No Speaker Leakage")
    print("=" * 60)
    
    dataset = load_from_disk('../hf_who_are_you')
    
    train_speakers = set(dataset['train']['speaker_id'])
    test_speakers = set(dataset['test']['speaker_id'])
    
    overlap = train_speakers & test_speakers
    
    print(f"\nTrain speakers: {len(train_speakers)}")
    print(f"Test speakers: {len(test_speakers)}")
    print(f"Overlap: {len(overlap)} (should be 0)")
    
    if len(overlap) == 0:
        print("✓ No speaker leakage detected!")
    else:
        print(f"✗ Warning: {len(overlap)} speakers in both splits!")


def main():
    """Run all examples."""
    import pandas as pd  # needed for example 3
    
    print("\n" + "=" * 60)
    print("Who Are You Dataset - Usage Examples")
    print("=" * 60)
    
    examples = [
        example_1_basic_loading,
        example_2_pair_matching,
        example_3_speaker_analysis,
        example_4_filter_by_dialect,
        example_5_phoneme_bigram_analysis,
        example_6_extract_for_detector,
        example_7_verify_split,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
