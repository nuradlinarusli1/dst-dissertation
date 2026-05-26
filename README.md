# COM 3610 Final Project - Prompt-Based Dialogue State Tracking using LLMs

## Project Overview

This project investigates the effectiveness of prompt-based Large Language Models (LLMs) for Dialogue State Tracking (DST) in task-oriented dialogue (TOD) systems without task-specific fine-tuning

The system evaluates zero-shot and few-shot prompting approaches on the MultiWOZ 2.2 dataset. Three approaches are evaluated:

1. A rule-based baseline
2. A LLM-based DST system using OpenRouter API
3. A speech-based pipeline using OpenAI Whisper ASR

The evaluated domains are:

- Restaurant
- Hotel
- Train

---

## Project Structure

```
src/
│
├── prompting/
│   ├── prompt_builder.py
│   └── few_shot_examples.py
│
├── models/
│   ├── rule_based.py
|   ├── run_wer_only.py
│   └── run_whisper.py
│
├── helper/
│   └── api_helper.py
│
├── schema.py
├── parser.py
└── run_evaluation.py

data/
└── processed/
    └── filtered
        ├── dev_filtered.json
        ├── test_filtered.json
        └── train_filtered.json
```

# Important Files

1. run_evaluation.py: Main evaluation script
2. rule_based.py: Rule-based baseline evaluation script
3. run_wer_only.py: Transcription evaluation script to get WER percentage
4. run_whisper.py: Speech transcription using Whisper
5. prompt_builder.py: Prompt generation logic
6. few_shot_examples.py: Few-shot examples
7. parser.py: JSON extraction and output parsing

## Setup

### Requirements

Python 3.10+ (recommended).

### Install dependencies

```bash
pip install openai-whisper
pip install faster-whisper
pip install jiwer
pip install python-dotenv
pip install requests
```

### Configure API key

The LLM experiments require an OpenRouter API key.

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_api_key_here
```

Without a valid API key, LLM-based evaluations cannot be executed.

### Download MultiWOZ 2.2 Dataset

The dataset is not included in this submission due to its size.

Download it from: https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2

Place the downloaded folder at:
data/MultiWOZ_2.2/

The preprocessed `test_filtered.json` file is included in `data/processed/filtered/` and is what the evaluation scripts use directly.

---

### Running Evaluations

All scripts should be run from the `src/` directory:

```bash
cd src
```

#### Rule-Based Baseline

Configure:

```python
USE_RULE_BASED = True
```

Then run:

```bash
python run_evaluation.py
```

#### Zero-Shot Prompting

Configure:

```python
USE_RULE_BASED = False
USE_FEW_SHOT = False
```

Run:

```bash
python run_evaluation.py
```
#### Few-Shot Prompting

Configure:

```python
USE_RULE_BASED = False
USE_FEW_SHOT = True
```

Run:

```bash
python run_evaluation.py
```

---

### Speech-Based Evaluation

Enable speech mode:

```python
USE_SPEECH = True
```

The system will:

1. Transcribe audio using Whisper.
2. Replace user utterances with ASR transcriptions.
3. Perform DST evaluation.
4. Compute Word Error Rate (WER).

To run WER evaluation independently:

```bash
python run_wer_only.py
```

---

## Evaluation Metrics

The following metrics are reported:

### Joint Goal Accuracy (JGA)

Measures whether the entire dialogue state is predicted correctly for a turn.

### Slot Accuracy (SA)

Measures accuracy at the individual slot level.

### Word Error Rate (WER)

Measures transcription quality for speech-based experiments.

---

## Dataset

This project uses the MultiWOZ 2.2 dataset.

Reference:

```
Zang et al. (2020). MultiWOZ 2.2: A Dialogue Dataset with Additional Annotation Corrections and State Tracking Baselines.
```

## Author

Nuradlina NB Rusli

BSc Computer Science (Artificial Intelligence)

University of Sheffield
