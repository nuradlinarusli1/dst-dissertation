import os
import whisper

audio_path = os.path.abspath("../data/audio/sample.wav")

model = whisper.load_model("medium")

def transcribe(audio_path: str) -> str:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    result = model.transcribe(audio_path)
    return result["text"].strip()

def calculate_wer(dialogues: list, subset_size: int, audio_dir: str) -> dict:
    try:
        from jiwer import wer, cer
    except ImportError:
        raise ImportError("Please install jiwer to calculate WER/CER: pip install jiwer")
    
    subset = dialogues[:subset_size]
    references = []
    hypotheses = []
    missing_files = 0

    for dialogue_idx, dialogue in enumerate(subset):
        turn_idx = 0
        for turn in dialogue["turns"]:
            if turn.get("speaker") != "USER":
                continue
            
            utterance = turn.get("utterance", "").strip().lower()
            audio_path = os.path.join(audio_dir, "sample.wav")
            #audio_path = os.path.join(audio_dir, f"dialogue_{dialogue_idx+1:02d}_turn_{turn_idx+1:02d}.wav")

            if not os.path.exists(audio_path):
                print(f"WARNING: Missing audio file {os.path.basename(audio_path)} — skipping")
                missing_files += 1
                turn_idx += 1
                continue

            transcription = transcribe(audio_path).lower()

            references.append(utterance)
            hypotheses.append(transcription)

            print(f"Dialogue {dialogue_idx+1}, Turn {turn_idx+1}")
            print(f"  Reference: \"{utterance}\"")
            print(f"  Hypothesis: \"{transcription}\"")
            turn_idx += 1

    results = {
        "total_turns": len(references),
        "missing_files": missing_files,
    }

    if references:
        results["wer"] = wer(references, hypotheses)
        results["cer"] = cer(references, hypotheses)
        print("\n================")
        print("WER Results:")
        print(f"\nTotal Turns Evaluated: {results['total_turns']}")
        print(f"Missing Audio Files: {results['missing_files']}")
        print(f"Word Error Rate (WER): {results['wer']:.3f} ({results['wer']*100:.1f}%)")
        print(f"Character Error Rate (CER): {results['cer']:.3f} ({results['cer']*100:.1f}%)")
    else:
        print("No valid audio files found for evaluation.")

    return results