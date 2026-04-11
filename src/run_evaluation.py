import time
import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from schema import get_initial_belief_state, get_allowed_values
from prompt_builder import build_prompt, build_safe_prompt
from parser import extract_json

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
USE_FEW_SHOT = False

# "oracle" evaluation mode: Use the gold belief state for the next turn
# "rollout" evaluation mode: Use the predicted belief state for the next turn
EVAL_MODE = "oracle"

print(f"\nRunning mode: {'few-shot' if USE_FEW_SHOT else 'zero-shot'}\n")
print(f"\nEvaluation mode: {EVAL_MODE}\n")

ALLOWED_VALUES = get_allowed_values()
INITIAL_BELIEF = get_initial_belief_state()
ALLOWED_DOMAINS = set(ALLOWED_VALUES.keys())

def make_model_input(tokeniser, prompt: str) -> str:
    if hasattr(tokeniser, "apply_chat_template") and getattr(tokeniser, "chat_template", None):
        messages = [
            {"role": "system", "content": "You are a dialogue state tracker (DST) for MultiWOZ 2.2. Output ONLY valid JSON. No extra text."},
            {"role": "user", "content": prompt},
        ]
        return tokeniser.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def extract_gold_belief_from_turn(turn: dict) -> dict | None:
    if turn.get("speaker") != "USER":
        return None
    
    print("\n--- EXTRACTING GOLD ---")
    
    belief = get_initial_belief_state()
    found_relevant_info = False
    
    for frame in turn.get("frames", []):
        print("\nSERVICE:", frame.get("service"))
        print("STATE:", json.dumps(frame.get("state", {}), indent=2))
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
    for domain in gold:
        if domain not in predicted:
            continue

        for slot in gold[domain]:
            if slot not in predicted[domain]:
                continue

            metrics["total"][domain][slot] += 1

            predicted_value = predicted[domain][slot]
            gold_value = gold[domain][slot]

            if slot == "request":
                predicted_value = sorted(predicted_value) if isinstance(predicted_value, list) else []
                gold_value = sorted(gold_value) if isinstance(gold_value, list) else []

            if predicted_value == gold_value:
                metrics["correct"][domain][slot] += 1

def normalise_text(value: str) -> str:
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
    if not isinstance(value, str):
        return value
            
    value = value.lower().strip()
    value = value.replace(".", ":")
    value = re.sub(r"\s*:\s*", ":", value)

    if re.match(r"^\d{1,2}:\d{2}$", value):
        hour, minute = value.split(":")
        return f"{int(hour):02d}:{minute}"
    
    match = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", value)
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

with open("../data/processed/train_restaurant.json", "r", encoding="utf-8") as f:
    dialogues = json.load(f)

#restaurant_dialogues = [d for d in dialogues if has_restaurant_turn(d)]
subset = [dialogues[7]]

print(f"Total dialogues: {len(dialogues)}")
print(f"Subset size: {len(subset)}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading model...")
tokeniser = AutoTokenizer.from_pretrained(MODEL_ID)
if tokeniser.pad_token is None:
    tokeniser.pad_token = tokeniser.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    quantization_config=bnb_config
)

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

evaluated_turns = 0
skipped_no_state_turns = 0
parse_failures = 0

for dialogue in subset:
    belief = get_initial_belief_state()
    dialogue_history = []

    for turn in dialogue["turns"]:
        speaker = turn.get("speaker")
        utterance = turn.get("utterance", "")

        if speaker == "SYSTEM":
            dialogue_history.append(f"SYSTEM (context only, do not copy values): {utterance}")
            continue
        
        if turn.get("speaker") != "USER":
            continue

        MAX_HISTORY_TURNS = 4
        recent_history = dialogue_history[-MAX_HISTORY_TURNS:]
        dialogue_history_str = "\n".join(recent_history)

        print("\n==============================")
        print("USER UTTERANCE:", utterance)
        print("RAW FRAMES FROM DATASET:")
        print(json.dumps(turn.get("frames", []), indent=2))

        gold_belief = extract_gold_belief_from_turn(turn)
        if gold_belief is None:
            skipped_no_state_turns += 1
            dialogue_history.append(f"USER: {utterance}")
            continue

        user_utterance = utterance

        fixed_prefix, variable_suffix = build_prompt(dialogue_history_str, belief, user_utterance, USE_FEW_SHOT)
        prompt = build_safe_prompt(tokeniser, fixed_prefix, variable_suffix, max_length=2048)
        model_input = make_model_input(tokeniser, prompt)
        inputs = tokeniser(
            model_input, 
            return_tensors="pt",
            truncation=False
            ).to(model.device)
        
        with torch.no_grad():
            t0 = time.time()

            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tokeniser.eos_token_id,
                pad_token_id=tokeniser.eos_token_id,
                use_cache=True
            )

            print("Turn seconds:", round(time.time() - t0, 2))

        prompt_len = inputs["input_ids"].shape[1]
        generated_only = outputs[0][prompt_len:]
        decoded = tokeniser.decode(generated_only, skip_special_tokens=True)

        print("\nRAW MODEL OUTPUT:")
        print(decoded)

        try:
            predicted_belief = extract_json(decoded)
            predicted_belief = normalise_belief(predicted_belief)
        except ValueError as e:
            parse_failures += 1
            print("PARSE FAILED:", e)
            print("RAW THAT FAILED:")
            print(decoded)
            print("RAW OUTPUT LENGTH:", len(decoded))
            dialogue_history.append(f"USER: {utterance}")
            continue

        gold_belief = normalise_belief(gold_belief)

        print(f"PREDICTED:", predicted_belief)
        print(f"GOLD:", gold_belief)

        update_slot_metrics(predicted_belief, gold_belief, metrics)

        metrics["joint_total"] += 1
        if predicted_belief == gold_belief:
            metrics["joint_correct"] += 1

        evaluated_turns += 1

        if EVAL_MODE == "oracle":
            belief = gold_belief
        elif EVAL_MODE == "rollout":
            belief = predicted_belief
        else:
            raise ValueError(f"Invalid EVAL_MODE: {EVAL_MODE}. Must be 'oracle' or 'rollout'.")

        dialogue_history.append(f"USER: {utterance}")

print("\nEvaluation Summary")
print(f"Evaluated Turns: {evaluated_turns}")
print(f"Skipped Turns (No State): {skipped_no_state_turns}")
print(f"Parse Failures: {parse_failures}")

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
