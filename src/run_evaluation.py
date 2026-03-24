import time
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from schema import get_initial_belief_state
from prompt_builder import build_prompt, build_safe_prompt
from parser import extract_json

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
USE_FEW_SHOT = False

print(f"\nRunning mode: {'few-shot' if USE_FEW_SHOT else 'zero-shot'}\n")

def make_model_input(tokeniser, prompt: str) -> str:
    if hasattr(tokeniser, "apply_chat_template") and getattr(tokeniser, "chat_template", None):
        messages = [
            {"role": "system", "content": "You are a dialogue state tracker (DST) for MultiWOZ 2.2. Output ONLY valid JSON. No extra text."},
            {"role": "user", "content": prompt},
        ]
        return tokeniser.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def extract_restaurant_belief_from_turn(turn: dict) -> dict | None:
    belief = get_initial_belief_state()

    if turn.get("speaker") != "USER":
        return None
    
    found_restaurant = False
    
    for frame in turn.get("frames", []):
        if frame.get("service") != "restaurant":
            continue

        state = frame.get("state", {})
        if not state:
            continue

        found_restaurant = True

        slot_values = state.get("slot_values", {})
        requested_slots = state.get("requested_slots", [])

        if "restaurant-area" in slot_values and slot_values["restaurant-area"]: 
            belief["restaurant"]["area"] = slot_values["restaurant-area"][0]

        if "restaurant-pricerange" in slot_values and slot_values["restaurant-pricerange"]:
            belief["restaurant"]["pricerange"] = slot_values["restaurant-pricerange"][0]

        if "restaurant-food" in slot_values and slot_values["restaurant-food"]:
            belief["restaurant"]["food"] = slot_values["restaurant-food"][0]

        for req in requested_slots:
            if req.startswith("restaurant-"):
                clean_request = req.replace("restaurant-", "")
                if clean_request not in belief["restaurant"]["request"]:
                    belief["restaurant"]["request"].append(clean_request)

    if not found_restaurant:
        return None

    return belief

def update_slot_metrics(predicted: dict, gold: dict, metrics: dict):
    for slot in ["area", "pricerange", "food", "request"]:
        metrics["total"][slot] += 1

        predicted_value = predicted["restaurant"][slot]
        gold_value = gold["restaurant"][slot]

        if slot == "request":
            predicted_value = sorted(predicted_value)
            gold_value = sorted(gold_value)
        
        if predicted_value == gold_value:
            metrics["correct"][slot] += 1

with open("../data/processed/dev.json", "r", encoding="utf-8") as f:
    dialogues = json.load(f)

subset = dialogues[:2]

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
    "correct": {"area": 0, "pricerange": 0, "food": 0, "request": 0},
    "total": {"area": 0, "pricerange": 0, "food": 0, "request": 0},
    "joint_correct": 0,
    "joint_total": 0,
}

evaluated_turns = 0
non_restaurant_skips = 0
parse_failures = 0

for dialogue in subset:
    belief = get_initial_belief_state()
    dialogue_history = ""

    for turn in dialogue["turns"]:
        speaker = turn.get("speaker")
        utterance = turn.get("utterance", "")

        if speaker == "SYSTEM":
            dialogue_history += f"\nSYSTEM: {utterance}\n"
            continue
        
        if turn.get("speaker") != "USER":
            continue

        gold_belief = extract_restaurant_belief_from_turn(turn)
        if gold_belief is None:
            non_restaurant_skips += 1
            dialogue_history += f"\nUSER: {utterance}\n"
            continue

        user_utterance = utterance

        fixed_prefix, variable_suffix = build_prompt(dialogue_history, belief, user_utterance, USE_FEW_SHOT)
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
                max_new_tokens=80,
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

        print(decoded)

        try:
            predicted_belief = extract_json(decoded)
        except ValueError:
            parse_failures += 1
            dialogue_history += f"\nUSER: {user_utterance}\n"
            continue

        update_slot_metrics(predicted_belief, gold_belief, metrics)

        metrics["joint_total"] += 1

        predicted_restaurant = predicted_belief["restaurant"].copy()
        gold_restaurant = gold_belief["restaurant"].copy()

        predicted_restaurant["request"] = sorted(predicted_restaurant["request"])
        gold_restaurant["request"] = sorted(gold_restaurant["request"])

        if predicted_restaurant == gold_restaurant:
            metrics["joint_correct"] += 1

        evaluated_turns += 1
        belief = predicted_belief
        #belief = gold_belief //Optional Additional Evaluation to compare if each turn is initialised with the correct state (to test evaluation of one slot)

        dialogue_history += f"\nUSER: {user_utterance}\n"

print("\nEvaluation Summary")
print(f"Evaluated Turns: {evaluated_turns}")
print(f"Non-evaluable Turns: {non_restaurant_skips}")
print(f"Parse Failures: {parse_failures}")

print("\nSlot Accuracy")
for slot in metrics["correct"]:
    total = metrics["total"][slot]
    correct = metrics["correct"][slot]
    acc = correct / total if total else 0.0
    print(f"{slot}: {acc:.3f} ({correct}/{total})")

joint_acc = metrics["joint_correct"] / metrics["joint_total"] if metrics["joint_total"] else 0.0
print(f"\nJoint Goal Accuracy: {joint_acc:.3f} ({metrics['joint_correct']}/{metrics['joint_total']})")
