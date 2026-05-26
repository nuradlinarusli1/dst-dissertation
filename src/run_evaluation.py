# src/run_evaluation.py
# Main evaluation script for Dialogue State Tracking (DST) on the MultiWOZ 2.2 dataset.
# Evaluates both LLM-based and rule-based DST approaches.
# For LLM-based DST, constructs prompts with optional few-shot examples, calls the model API, and parses the output.
# For rule-based DST, applies rules to update the belief state based on user utterances
# Calculates slot accuracy and joint goal accuracy

import time
import json
import re
import numpy as np

from schema import get_initial_belief_state, get_allowed_values
from prompting.prompt_builder import build_prompt
from parser import extract_json
from helper.api_helper import call_model_api
from models.rule_based import rule_based_predict
from models.run_whisper import transcribe

# Evaluation configuration toggles
MODEL_ID = "qwen/qwen-2.5-7b-instruct"
USE_FEW_SHOT =  True # "True" or "False"
PRINT_PROMPTS = False # "True" or "False"
USE_RULE_BASED = False
USE_SPEECH = False
N_SHOTS = 5 # 0, 1, 3 or 5

print("\n==============================")
print(f"Running mode: {N_SHOTS}-shot" if USE_FEW_SHOT else "zero-shot")
print("\n==============================")

ALLOWED_VALUES = get_allowed_values()
INITIAL_BELIEF = get_initial_belief_state()
ALLOWED_DOMAINS = set(ALLOWED_VALUES.keys())

def extract_gold_belief_from_turn(turn: dict) -> dict | None:
    # Extract the gold belief state from the turn's frames in the dialogue data.
    # Returns None if the turn has no relevant state information
    if turn.get("speaker") != "USER":
        return None
    
    belief = get_initial_belief_state()
    found_relevant_info = False
    
    for frame in turn.get("frames", []):
        service = frame.get("service")
        if service not in ALLOWED_DOMAINS:
            continue

        if service not in belief:
            continue

        state = frame.get("state", {})
        if not isinstance(state, dict):
            continue

        slot_values = state.get("slot_values", {})
        requested_slots = state.get("requested_slots", [])

        for full_slot_name, values in slot_values.items():
            if "-" not in full_slot_name:
                continue
            if not isinstance(values, list) or not values:
                continue

            domain, slot = full_slot_name.split("-", 1)

            if domain != service:
                continue
            if domain not in belief or slot not in belief[domain]:
                continue
            if slot == "request":
                continue

            belief[domain][slot] = values[0]
            found_relevant_info = True

        for req in requested_slots:
            if "-" not in req:
                continue

            domain, slot = req.split("-", 1)

            if domain != service:
                continue
            if domain not in belief:
                continue
            if "request" not in belief[domain]:
                continue

            if slot in ALLOWED_VALUES[domain]["request"] and slot not in belief[domain]["request"]:
                belief[domain]["request"].append(slot)
                found_relevant_info = True
    return belief if found_relevant_info else None

def update_slot_metrics(predicted: dict, gold: dict, metrics: dict):
    # Compare predicted and gold belief states slot by slot and update metric counter
    # Skips slots where the gold value is "none" (unfilled) or empty request lists
    for domain in gold:
        if domain not in predicted:
            continue

        for slot in gold[domain]:
            if slot not in predicted[domain]:
                continue

            predicted_value = predicted[domain][slot]
            gold_value = gold[domain][slot]

            if slot == "request":
                predicted_value = sorted(predicted_value) if isinstance(predicted_value, list) else []
                gold_value = sorted(gold_value) if isinstance(gold_value, list) else []

                if not gold_value:
                    continue
            else:
                if gold_value == "none":
                    continue
            
            metrics["total"][domain][slot] += 1

            if predicted_value == gold_value:
                metrics["correct"][domain][slot] += 1

def normalise_text(value: str) -> str:
    # Lowercase, strip and map common variants of words to slot values
    if not isinstance(value, str):
        return value
    
    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)

    value_map = {
        "city centre": "centre",
        "city center": "centre",
        "center": "centre",
        "guesthouses": "guesthouse",
        "guest house": "guesthouse",
        "classic british": "british",
        "modern european": "european",
        "free internet": "free",
        "free wifi": "free",
        "free parking": "free",
    }

    return value_map.get(value, value)

def normalise_number_string(value:str) -> str:
    # Convert word numbers and digit strings to a consistent integer string format
    number_map = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
        "fourteen": "14",
        "fifteen": "15",
        "zero": "0"
    }

    value = normalise_text(value)
    if value.isdigit():
        return str(int(value))
    return number_map.get(value, value)

def normalise_time_string(value: str) -> str:
    # Normalise time strings to HH:MM format, handling am/pm and common prefixes
    if not isinstance(value, str):
        return value
            
    value = value.lower().strip()
    value = value.replace(".", ":")
    value = re.sub(r"\s*:\s*", ":", value)

    value = re.sub(r"\b(after|before|around|at|about|by|from)\b", "", value).strip()

    match = re.search(r"(\d{1,2}):(\d{2})", value)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        return f"{hour:02d}:{minute}"
    
    match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", value)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        period = match.group(3)
    
        if period == "pm" and hour != 12:
            hour += 12
        if period == "am" and hour == 12:
            hour = 0

        return f"{hour:02d}:{minute:02d}"

    return value

def normalise_belief(belief: dict) -> dict:
    # Apply all normalisation functions to every slot in the belief state
    # Routes each slot to the appropriate normaliser based on slot type
    normalised = get_initial_belief_state()

    for domain, slots in normalised.items():
        domain_data = belief.get(domain, {}) if isinstance(belief, dict) else {}

        for slot in slots:
            if slot == "request":
                request_values = domain_data.get("request", [])
                if isinstance(request_values, list):
                    cleaned = []
                    for v in request_values:
                        item = normalise_text(str(v))
                        if item in ALLOWED_VALUES[domain]["request"]:
                            cleaned.append(item)
                    normalised[domain]["request"] = sorted(set(cleaned))
                else:
                    normalised[domain]["request"] = []
            else:
                value = domain_data.get(slot, slots[slot])
                if slot in {"bookpeople", "bookstay", "stars"}:
                    value = normalise_number_string(value)
                elif slot in {"booktime", "leaveat", "arriveby"}:
                    value = normalise_time_string(value)
                else:
                    value = normalise_text(value)

                normalised[domain][slot] = value

    return normalised

def remove_fulfilled_requests(pred_belief: dict, system_utterance: str) -> dict:
    # Remove request slots from the predicted belief that the system has already fulfilled
    # Checks the system utterance for keywords associated with each request slot
    system_lower = system_utterance.lower()

    slot_markers = {
        "phone": ["phone", "number", "phone number"],
        "address": ["address", "location"],
        "postcode": ["postcode", "postal code", "post code"],
        "ref": ["reference", "ref", "booking reference"],
        "id": ["id", "train id", "train number"],
        "price": ["price", "cost"],
        "leaveat": ["leave", "leave at", "departure time", "departure"],
        "arriveby": ["arrive", "arrive by", "arrival time", "arrival"],
        "bookday": ["book day", "booking day", "booked for", "booking date", "book day"],
        "bookpeople": ["book people", "booking people", "booked for", "number of people", "book people"],
        "bookstay": ["book stay", "booking stay", "booked for", "number of nights", "book stay"],
        "stars": ["star", "stars", "rating"],
        "internet": ["internet", "wifi", "wi-fi"],
        "parking": ["parking"],
        "area": ["area", "location"],
        "pricerange": ["price range", "pricerange", "cost"],
        "food": ["food", "cuisine"],
        "type": ["type", "kind"],
        "name": ["name"]
    }

    for domain, domain_state in pred_belief.items():
        requests = domain_state.get("request", [])
        if not isinstance(requests, list):
            continue

        remaining = []
        for req in requests:
            markers = slot_markers.get(req, [req])
            fulfilled = any(marker in system_lower for marker in markers)
            
            if not fulfilled:
                remaining.append(req)
        pred_belief[domain]["request"] = remaining
    return pred_belief


# Load dataset
with open("../data/filtered/test_filtered.json", "r", encoding="utf-8") as f:
    dialogues = json.load(f)

# If running text-based evaluation, run full filtered test dataset
subset = dialogues
# If running speech-based evalution, run only the first 50 dialogues
# subset = dialogues[:50] 

print("\n==============================")
print(f"Total dialogues: {len(dialogues)}")
print(f"Subset size: {len(subset)}")
print("API Loading...")
print("\n==============================")

# Initialise metrics and counters for evaluation
metrics = {
    "correct": {
        domain: {slot: 0 for slot in slots}
        for domain, slots in INITIAL_BELIEF.items()
    },
    "total": {
        domain: {slot: 0 for slot in slots}
        for domain, slots in INITIAL_BELIEF.items()
    },
    "joint_correct": 0,
    "joint_total": 0,
}

# Counters for evaluation summary
evaluated_turns = 0
skipped_no_state_turns = 0
parse_failures = 0
api_failures = 0

def print_slot_differences(predicted: dict, gold: dict):
    # Print comparison of mismatched slots between predicted and gold belief states for error analysis
    print("\nMISMATCHED SLOTS:")

    for domain in gold:
        if domain not in predicted:
            print(f"{domain}: missing in prediction")
            continue

        for slot in gold[domain]:
            pred_val = predicted[domain].get(slot)
            gold_val = gold[domain].get(slot)

            # Handle request list sorting
            if slot == "request":
                pred_val = sorted(pred_val) if isinstance(pred_val, list) else []
                gold_val = sorted(gold_val) if isinstance(gold_val, list) else []

            if pred_val != gold_val:
                print(f"  [{domain}] {slot}:")
                print(f"    predicted = {pred_val}")
                print(f"    gold      = {gold_val}")

# Main evaluation loop
dialogue_jga_scores = []

for dialogue_idx, dialogue in enumerate(subset):
    belief = get_initial_belief_state()
    dialogue_history = []
    system_utterance = ""
    turn_idx = 0
    dialogue_joint_correct = 0
    dialogue_joint_total =  0

    for turn in dialogue["turns"]:
        speaker = turn.get("speaker")
        utterance = turn.get("utterance", "")

        if speaker == "SYSTEM":
            system_utterance = utterance
            dialogue_history.append(f"SYSTEM: {utterance}")
            continue
        
        if speaker != "USER":
            continue

        print("\n==============================")
        print(f"Dialogue ID: {dialogue.get('dialogue_id', 'N/A')}")

        # Limit dialogue history to avoid exceeding model's context window
        MAX_HISTORY_TURNS = 10
        recent_history = dialogue_history[-MAX_HISTORY_TURNS:]
        dialogue_history_str = "\n".join(recent_history)

        print("\n==============================")
        print(f"Dialogue History: ")
        print(f"{dialogue_history_str if dialogue_history_str else '[EMPTY]'}")

        if USE_SPEECH:
            audio_path = f"../data/audio/dialogue_{dialogue_idx+1:02d}_turn_{turn_idx+1:02d}.wav"
            print(f"AUDIO PATH: {audio_path}")

            try:
                user_utterance = transcribe(audio_path)
            except Exception as e:
                api_failures += 1
                print("TRANSCRIPTION FAILED:", e)
                dialogue_history.append(f"USER: {utterance}")
                turn_idx += 1
                continue

            print(f"TRANSCRIBED UTTERANCE:")
            print(user_utterance)
        else:
            user_utterance = utterance
            print("\n==============================")
            print("USER UTTERANCE:")
            print(user_utterance)

        turn_idx += 1
    
        gold_belief = extract_gold_belief_from_turn(turn)

        if gold_belief is None:
            skipped_no_state_turns += 1
            dialogue_history.append(f"USER: {user_utterance}")
            continue

        if USE_RULE_BASED:
            predicted_belief = rule_based_predict(user_utterance, belief)
            predicted_belief = normalise_belief(predicted_belief)
        else:
            fixed_prefix, variable_suffix = build_prompt(dialogue_history_str, belief, user_utterance, system_utterance, USE_FEW_SHOT, N_SHOTS)
            prompt = fixed_prefix + "\n\n" + variable_suffix

            if PRINT_PROMPTS:
                print("\n==============================")
                print("PROMPT SENT TO MODEL:")
                print(prompt)

            try:
                t0 = time.time()
                decoded = call_model_api(prompt, MODEL_ID)
                print("Turn seconds:", round(time.time() - t0, 2))
            except Exception as e:
                api_failures += 1
                print("API CALL FAILED:", e)
                dialogue_history.append(f"USER: {user_utterance}")
                continue

            try:
                extracted_json = extract_json(decoded)
                predicted_belief = normalise_belief(extracted_json)
                predicted_belief = remove_fulfilled_requests(predicted_belief, system_utterance)
            except ValueError as e:
                parse_failures += 1
                print("PARSE FAILED:", e)
                print("RAW THAT FAILED:")
                print(decoded)
                print("RAW OUTPUT LENGTH:", len(decoded))
                dialogue_history.append(f"USER: {user_utterance}")
                continue

        gold_belief = normalise_belief(gold_belief)

        print("\n==============================")
        print(f"NORMALISED PREDICTED:", predicted_belief)
        print(f"\nNORMALISED GOLD:", gold_belief)

        if predicted_belief == gold_belief:
            print("\nPREDICTION MATCHES GOLD")
        else:
            print("\nPREDICTION DOES NOT MATCH GOLD")
            print_slot_differences(predicted_belief, gold_belief)

        update_slot_metrics(predicted_belief, gold_belief, metrics)

        metrics["joint_total"] += 1
        dialogue_joint_total += 1
        if predicted_belief == gold_belief:
            metrics["joint_correct"] += 1
            dialogue_joint_correct += 1

        evaluated_turns += 1
        belief = predicted_belief

        print("\n==============================")
        print("UPDATED BELIEF STATE FOR NEXT TURN:")
        print(json.dumps(belief, indent=2))

        dialogue_history.append(f"USER: {user_utterance}")
    
    if dialogue_joint_total >0:
            dialogue_jga = dialogue_joint_correct / dialogue_joint_total
            dialogue_jga_scores.append(dialogue_jga)

# Print evaluation summary
print("\n==============================")
print("\nEvaluation Summary")
print("==============================")

print(f"Evaluated Turns: {evaluated_turns}")
print(f"Skipped Turns (No State): {skipped_no_state_turns}")
print(f"Parse Failures: {parse_failures}")
print(f"API Failures: {api_failures}")

print("\nSlot Accuracy")
for domain in metrics["correct"]:
    print(f"\n{domain.upper()}")
    for slot in metrics["correct"][domain]:
        total = metrics["total"][domain][slot]
        correct = metrics["correct"][domain][slot]
        acc = correct / total if total else 0.0
        print(f"  {slot}: {acc:.3f} ({correct}/{total})")

joint_acc = metrics["joint_correct"] / metrics["joint_total"] if metrics["joint_total"] else 0.0
print(f"\nJoint Goal Accuracy: {joint_acc:.3f} ({metrics['joint_correct']}/{metrics['joint_total']})")
if dialogue_jga_scores:
    mean_jga = np.mean(dialogue_jga_scores)
    std_jga = np.std(dialogue_jga_scores)
    max_jga = np.max(dialogue_jga_scores)
    min_jga = np.min(dialogue_jga_scores)

    print("\nDialogue-level JGA Statistics:")
    print(f"Mean JGA: {mean_jga:.3f}")
    print(f"Std Dev: {std_jga:.3f}")
    print(f"Max JGA: {max_jga:.3f}")
    print(f"Min JGA: {min_jga:.3f}")
