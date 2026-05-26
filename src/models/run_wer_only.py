# src/models/run_wer_only.py
# Script is used to calculate Word Error Rate (WER) on the first 50 dialogues on the filtered test set, using transcriptions from Whisper.
# Usage: python src/models/run_wer_only.py from the src/ directory.

import json
from models.run_whisper import calculate_wer

# Load filtered test dataset
with open("../data/processed/filtered/test_filtered.json", "r", encoding="utf-8") as f:
    dialogues = json.load(f)

# Evaluate WER on the first 50 dialogues using transcriptions from Whisper.
calculate_wer(dialogues, len(dialogues[:50]), "../data/audio/converted_audio")