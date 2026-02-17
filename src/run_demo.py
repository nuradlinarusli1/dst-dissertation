import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from schema import get_initial_belief_state
from prompt_builder import build_prompt
from parser import extract_json

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading model...")
tokeniser = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretained(
    MODEL_ID,
    device_map="auto",
    quantisation_config=bnb_config
)

belief = get_initial_belief_state()
dialogue_history = ""

turns = [
    "Hi, I'm looking for a cheap restaurant in the centre.",
    "Actually make it moderate price.",
    "Can I get the address too?"
]

for turn in turns:
    prompt = build_prompt(dialogue_history, belief, turn)
    inputs = tokeniser(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=0.0
        )

    decoded = tokeniser.decode(outputs[0], skip_special_tokens=True)

    belief = extract_json(decoded)
    dialogue_history += f"\nUSER: {turn}\n"
    print("\nUpdated belief state:")
    print(belief)