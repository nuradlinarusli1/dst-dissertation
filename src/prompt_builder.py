import json
from schema import get_allowed_values, get_initial_belief_state
from few_shot_examples import FEW_SHOT_EXAMPLES


def build_prompt(dialogue_history: str, prev_belief: dict, user_utterance: str, use_few_shot: bool = False) -> tuple[str, str]  :
    allowed = get_allowed_values()
    required_structure = get_initial_belief_state()

    example_block = ""

    if use_few_shot:
        example_text = []

        for example in FEW_SHOT_EXAMPLES:
            example_text.append(f"""
Example

Dialogue history:
{example["dialogue_history"]}

Previous belief state:
{json.dumps(example["prev_belief"], indent=2)}

User utterance:
"{example["user"]}"

Updated belief state:
{json.dumps(example["belief"], indent=2)}
""". strip())

        example_block = "\n".join(example_text)

    fixed_prefix = f"""
You are a dialogue state tracker (DST) for the MultiWOZ 2.2 dataset.

Task:
Given the dialogue history and the latest user utterance, update the belief state.

STRICT OUTPUT RULES (VERY IMPORTANT):
- Output ONLY valid JSON. NO extra text.
- Output must match this exact JSON structure (same keys, same nesting):
{json.dumps(required_structure, indent=2)}

STATE UPDATE RULES:
- Start from the previous belief state
- ONLY update based on the latest USER utterance
- Keep all previous values unless the user explicitly changes them.
- Do NOT add or change constraint slot values based on SYSTEM responses.
- SYSTEM utterances are context only
- If the system has provided the requested information for a slot, remove it from the request list in the belief state.
- Track ONLY these domains: "restaurant", "hotel", "train". Do NOT include any other domains such as "attraction", "taxi", "museum" etc.
- Use "none" ONLY for slots that are still unknown and have never been filled.
- Use "dontcare" if the user has explicitly indicated no preference for a slot, for example "any", "no preference" or similar.

- Allowed values:
{json.dumps(allowed, indent=2)}
""".strip()
    
    if example_block:
        fixed_prefix += f"\n\n{example_block}"

    variable_suffix = f"""
Now update the belief state for the following turn.

Dialogue history:
{dialogue_history}

Previous belief state:
{json.dumps(prev_belief, indent=2)}

Latest user utterance:
"{user_utterance}"

Return ONLY the updated belief state JSON:
""".strip()

    return fixed_prefix, variable_suffix


def build_safe_prompt(tokeniser, fixed_prefix: str, variable_suffix: str, max_length: int = 2048) -> str:
    fixed_ids = tokeniser(fixed_prefix, add_special_tokens=False)["input_ids"]
    suffix_ids = tokeniser(variable_suffix, add_special_tokens=False)["input_ids"]

    allowed_suffix_len = max_length - len(fixed_ids) - 50

    if allowed_suffix_len <= 0:
        raise ValueError("Fixed prefix is too long. Shorten instructions or few-shot examples")
    
    if len(suffix_ids) > allowed_suffix_len:
        suffix_ids = suffix_ids[-allowed_suffix_len:]

    trimmed_suffix = tokeniser.decode(suffix_ids, skip_special_tokens=True)
    return fixed_prefix + "\n\n" + trimmed_suffix
