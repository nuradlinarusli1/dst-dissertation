# record_audio.py
import json
import os

with open("../data/processed/filtered/train_multi-domain.json", "r", encoding="utf-8") as f:
    dialogues = json.load(f)

subset = dialogues[:50]
audio_dir = "../data/audio"
os.makedirs(audio_dir, exist_ok=True)

print("AUDIO RECORDING HELPER")
print("======================")
print("For each utterance:")
print("1. Read the utterance out loud and record it")
print("2. Save the file with the exact filename shown")
print("3. Press Enter to continue to the next utterance")
print("======================\n")

for dialogue_idx, dialogue in enumerate(subset):
    turn_idx = 0
    
    for turn in dialogue["turns"]:
        if turn.get("speaker") != "USER":
            continue

        utterance = turn.get("utterance", "")
        filename = f"dialogue_{dialogue_idx+1:02d}_turn_{turn_idx+1:02d}.wav"
        filepath = os.path.join(audio_dir, filename)

        # Skip if already recorded
        if os.path.exists(filepath):
            print(f"ALREADY RECORDED: {filename} — skipping")
            turn_idx += 1
            continue

        print(f"\nDialogue {dialogue_idx+1}/50, Turn {turn_idx+1}")
        print(f"Save as: {filename}")
        print(f"Say: \"{utterance}\"")
        input("Press Enter when recorded...")

        turn_idx += 1

print("\nAll utterances done!")