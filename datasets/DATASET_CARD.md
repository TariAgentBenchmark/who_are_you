# Who Are You: Audio Deepfake Detection Dataset

This dataset accompanies the paper **"Who Are You (I Really Wanna Know)? Detecting Audio DeepFakes Through Vocal Tract Reconstruction"** presented at USENIX Security 2022.

## Dataset Description

This dataset contains real and synthetic audio samples used for researching audio deepfake detection. The dataset is based on the TIMIT Acoustic-Phonetic Continuous Speech Corpus with corresponding synthetic audio generated using deepfake models.

### Key Features

- **Audio**: 16kHz, 16-bit PCM WAV files
- **Phoneme annotations**: Time-aligned phonetic transcriptions in ARPABET and IPA
- **Word annotations**: Time-aligned word boundaries
- **Bigram annotations**: Adjacent phoneme pairs for articulatory analysis
- **Speaker information**: Gender, dialect region, unique speaker ID
- **Labels**: Binary `is_fake` label indicating synthetic vs. real audio
- **Pair matching**: `pair_id` for matching real and fake samples

### Dataset Statistics

| Split | Samples | Real | Fake | Speakers | Pairs |
|-------|---------|------|------|----------|-------|
| Train | 7,372 | 3,700 | 3,672 | 370 | 3,700 |
| Test | 1,832 | 920 | 912 | 92 | 920 |
| **Total** | **9,204** | **4,620** | **4,584** | **462** | **3,700** |

**Note**: The dataset uses speaker-aware split (no speaker overlap between train/test).

## Original Dataset Structure

### Real Audio (TIMIT)

The original TIMIT dataset uses a flat structure organized by speakers:

```
datasets/TIMIT/
├── FAEM0/                    # Speaker ID
│   ├── SA1.wav              # Audio file (16kHz, 16-bit, mono)
│   ├── SA1.PHN              # Phoneme annotation
│   ├── SA1.WRD              # Word annotation
│   ├── SA1.TXT              # Text and sample range
│   ├── SA2.wav
│   ├── SI1392.wav           # SI type (specific sentences)
│   ├── SX132.wav            # SX type (dialect sentences)
│   └── ...                  # ~10 sentences per speaker
├── FAJW0/
└── ... (462 speakers total)
```

**File Types:**
- `.wav`: Audio files, 16kHz sample rate, 16-bit PCM, mono
- `.PHN`: Phoneme annotations, format: `start_sample end_sample phoneme`
- `.WRD`: Word annotations, format: `start_sample end_sample word`
- `.TXT`: Full text, format: `start_sample end_sample text`

**Sentence Types:**
- `SA1/SA2`: Standard sentences (same for all speakers)
- `SI*`: Specific sentences (different per speaker)
- `SX*`: Dialect sentences (covering dialect characteristics)

### Fake Audio (generated_TIMIT)

The fake audio uses a hierarchical structure organized by dialect region (DR):

```
datasets/generated_TIMIT/
├── DR1/                     # Dialect Region 1 (New England)
│   ├── FAEM0/              # Speaker ID (matches real data)
│   │   ├── SA1.wav         # Fake version of real audio
│   │   └── ...             # Same structure as real data
├── DR2/                     # Dialect Region 2 (Northern)
├── DR3/                     # Dialect Region 3 (North Midland)
├── DR4/                     # Dialect Region 4 (South Midland)
├── DR5/                     # Dialect Region 5 (Southern)
├── DR6/                     # Dialect Region 6 (New York City)
├── DR7/                     # Dialect Region 7 (Western)
└── DR8/                     # Dialect Region 8 (Army Brat)
```

**Correspondence:**
- Same speaker ID = same speaker
- Same utterance ID (e.g., `SA1`) = same sentence
- Same text annotations, only audio waveform differs

```
Real: datasets/TIMIT/FAEM0/SA1.wav
Fake: datasets/generated_TIMIT/DR2/FAEM0/SA1.wav
      └── Same speaker (FAEM0)
      └── Same sentence (SA1)
      └── Same text annotations
      └── Only audio waveform differs
```

## Dataset Features

```python
{
    'audio': Audio(16000 Hz),
    'pair_id': 'FAEM0_SA1',              # Match real & fake samples
    'utterance_id': 'SA1',
    'speaker_id': 'FAEM0',
    'is_fake': False,                     # True/False
    'gender': 'female',                   # male/female
    'dialect_region': 'DR2',              # DR1-DR8
    'text': 'She had your dark suit...',
    'phonemes': [
        {'start': 0, 'end': 1600, 'phoneme': 'h#', 'ipa': 'N/A'},
        {'start': 1600, 'end': 4160, 'phoneme': 'sh', 'ipa': 'ʃ'},
        ...
    ],
    'words': [
        {'start': 1600, 'end': 5280, 'word': 'she'},
        ...
    ],
    'bigrams': [                          # Bigram pairs (core for detector)
        {'phoneme1': 'h#', 'phoneme2': 'sh', 'ipa1': 'N/A', 'ipa2': 'ʃ', 'boundary': 1600},
        ...
    ],
    'num_phonemes': 42,
    'num_words': 12,
}
```

## Usage

### Loading the Dataset

```python
from datasets import load_dataset

# Load from Hugging Face Hub
dataset = load_dataset("tari-tech/13832472466-who_are_you")

# Access splits
train_dataset = dataset['train']
test_dataset = dataset['test']

# Access a sample
sample = dataset['train'][0]
print(sample['audio'])      # Audio array and path
print(sample['is_fake'])    # True or False
print(sample['phonemes'])   # List of phoneme annotations
```

### Pairing Real and Fake Samples

```python
from collections import defaultdict

pairs = defaultdict(lambda: {'real': None, 'fake': None})

for sample in dataset['train']:
    pid = sample['pair_id']
    if sample['is_fake']:
        pairs[pid]['fake'] = sample
    else:
        pairs[pid]['real'] = sample

# Get a pair
real_sample = pairs['FAEM0_SA1']['real']
fake_sample = pairs['FAEM0_SA1']['fake']
```

### Extracting Bigram Windows (for Original Detector)

```python
WINDOW_SIZE = 565

def extract_windows(sample):
    audio = sample['audio']['array']
    windows = []
    
    for bigram in sample['bigrams']:
        boundary = bigram['boundary']
        start = boundary - WINDOW_SIZE // 2
        end = start + WINDOW_SIZE
        
        if start >= 0 and end <= len(audio):
            windows.append({
                'audio': audio[start:end],
                'label': f"{bigram['ipa1']} -- {bigram['ipa2']}",
                'boundary': boundary
            })
    
    return windows

windows = extract_windows(dataset['train'][0])
```

### Filtering by Dialect Region

```python
# Keep only DR1 (New England) samples
dr1_samples = dataset['train'].filter(lambda x: x['dialect_region'] == 'DR1')

# Keep only female real samples
female_real = dataset['test'].filter(
    lambda x: x['gender'] == 'female' and not x['is_fake']
)
```

## Split Strategy

The dataset uses **Speaker-aware Split**:

- All samples from the same speaker are in the same split
- Training and testing speakers do not overlap
- This prevents speaker leakage for vocal tract features

```
Train speakers: ~80% (369 speakers)
Test speakers: ~20% (93 speakers)
```

## Dialect Regions

From TIMIT documentation:

| DR | Region | Description |
|----|--------|-------------|
| DR1 | New England | New England |
| DR2 | Northern | Northern |
| DR3 | North Midland | North Midland |
| DR4 | South Midland | South Midland |
| DR5 | Southern | Southern |
| DR6 | New York City | New York City |
| DR7 | Western | Western |
| DR8 | Army Brat | Army Brat (frequent moves) |

## Dataset Creation

### Curation Rationale

This dataset was created to support research in audio deepfake detection using articulatory phonetics. Unlike traditional detection methods that rely on statistical patterns, this approach uses fluid dynamics to estimate vocal tract configurations during speech generation.

### Source Data

- **Real Audio**: TIMIT Acoustic-Phonetic Continuous Speech Corpus
- **Synthetic Audio**: Generated using neural speech synthesis models

### Annotations

- Phoneme annotations follow the TIMIT standard (ARPABET encoding)
- IPA conversions provided for international phonetic representation
- Word boundaries manually aligned
- Bigram pairs automatically generated from adjacent phonemes

## Considerations for Using the Data

### Social Impact

This dataset is intended for research in detecting non-consensual voice synthesis (deepfakes). The goal is to develop tools that can protect individuals from voice impersonation attacks while preserving legitimate uses of voice synthesis technology.

### Discussion of Biases

- **Gender**: TIMIT contains both male and female speakers, but may not represent all gender identities
- **Dialect**: TIMIT covers 8 major dialect regions of American English
- **Language**: English only
- **Synthetic models**: Represents the state of deepfake technology at the time of dataset creation

### Other Known Limitations

- Audio quality limited to 16kHz (telephone bandwidth)
- Synthetic audio may not represent latest deepfake technologies
- Dataset size is relatively small compared to modern deep learning datasets

## Citation Information

```bibtex
@inproceedings{blue2022who,
  title={Who Are You (I Really Wanna Know)? Detecting Audio DeepFakes Through Vocal Tract Reconstruction},
  author={Blue, Logan and Givehchian, Hadi and others},
  booktitle={USENIX Security Symposium},
  year={2022}
}
```

## Additional Information

### Dataset Curators

- Logan Blue
- Hadi Givehchian
- Other authors from the USENIX Security 2022 paper

### Licensing Information

- Real audio: Subject to TIMIT corpus license (LDC User Agreement)
- Synthetic audio: Generated for research purposes
- Code and annotations: See original repository license

### Contributions

Thanks to the original authors for providing the dataset and code for vocal tract reconstruction-based deepfake detection.
