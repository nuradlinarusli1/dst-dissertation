import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from schema import get_initial_belief_state
from prompt_builder import build_prompt, build_safe_prompt
from parser import extract_json

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
USE_FEW_SHOT = False

def make_model_input(tokeniser, prompt: str) -> str:
    if hasattr(tokeniser, "apply_chat_template") and getattr(tokeniser, "chat_template", None):
        messages = [
            {"role": "system", "content": "You are a dialogue state tracker (DST) for MultiWOZ 2.2. Output ONLY valid JSON. No extra text."},
            {"role": "user", "content": prompt},
        ]
        return tokeniser.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

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

belief = get_initial_belief_state()
dialogue_history = ""

turns = [
    "I'm looking for something affordable in the centre.",
    "Actually make it moderate price.",
    "Can I get the address too?"
]

print(f"\nRunning mode: {'few-shot' if USE_FEW_SHOT else 'zero-shot'}\n")

for turn in turns:
    fixed_prefix, variable_suffix = build_prompt(dialogue_history, belief, turn, USE_FEW_SHOT)
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

    print("\nRaw model output:")
    print(decoded)
    
    try:
        belief = extract_json(decoded)
    except ValueError as e:
        print("\nParse error:", e)
        print("Keeping previous belief state.")
        continue

    dialogue_history += f"\nUSER: {turn}\n"
    print("\nUpdated belief state:")
    print(belief)
