import json
from pathlib import Path
from typing import Any

ALLOWED_DOMAINS = {"restaurant", "hotel", "train"}

INPUT_PATH = Path("data/processed/dev.json")
OUTPUT_PATH = Path("data/processed/dev_multi-domain.json")
# IDS_OUTPUT_PATH = Path("data/processed/test_filtered_ids.txt")

def extract_domains_from_turn(turn: dict[str, Any]) -> set[str]:
    domains = set()

    for frame in turn.get("frames", []):
        service = frame.get("service")
        if not isinstance(service, str) or not service.strip():
            continue

        state = frame.get("state", {})
        active_intent = state.get("active_intent", "NONE")
        requested_slots = state.get("requested_slots", [])
        slot_values = state.get("slot_values", {})

        actions = frame.get("actions", [])
        slots = frame.get("slots", [])

        is_meaningful = (
            active_intent != "NONE" or
            bool(requested_slots) or
            bool(slot_values) or
            bool(actions) or
            bool(slots)
        )

        if is_meaningful:
            domains.add(service.strip().lower())

    return domains

def extract_domains_from_dialogue(dialogue: dict[str, Any]) -> set[str]:
    dialogue_domains = set()
    for turn in dialogue.get("turns", []):
        dialogue_domains.update(extract_domains_from_turn(turn))
    return dialogue_domains

def get_dialogue_id(dialogue : dict[str, Any], fallback_index: int) -> str:
    for key in ["dialogue_id", "id"]:
        value = dialogue.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"dialogue_{fallback_index}"

def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")
    
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

        print(type(data))
        print(type(data[0]))
        print(json.dumps(data[0], indent=2)[:2000])

    if not isinstance(data, list):
        raise ValueError("Expected input JSON to be a list of dialogues.")
    
    filtered_dialogues: list[dict[str, Any]] = []
    kept_ids: list[str] = []

    for i, dialogue in enumerate(data):
        if not isinstance(dialogue, dict):
            continue

        dialogue_id = get_dialogue_id(dialogue, i)
        domains = extract_domains_from_dialogue(dialogue)

        if i < 10:
            print(f"Dialogue ID: {dialogue_id}, Domains: {domains}, Kept: {domains and domains.issubset(ALLOWED_DOMAINS)}")

        if domains and domains.issubset(ALLOWED_DOMAINS):
            filtered_dialogues.append(dialogue)
            kept_ids.append(dialogue_id)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(filtered_dialogues, f, indent=2, ensure_ascii=False)

    #with IDS_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        #for dialogue_id in kept_ids:
            #f.write(dialogue_id + "\n")

    print(f"Total dialogues in input: {len(data)}")
    print(f"Filtered dialogues kept: {len(filtered_dialogues)}")
    print(f"Saved filtered dialogues to: {OUTPUT_PATH}")
    #print(f"Saved dialogue IDs to: {IDS_OUTPUT_PATH}")

if __name__ == "__main__":
    main()