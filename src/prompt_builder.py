import json
from schema import get_allowed_values, get_initial_belief_state
from few_shot_examples import FEW_SHOT_EXAMPLES


def build_prompt(dialogue_history: str, prev_belief: dict, user_utterance: str, use_few_shot: bool = False) -> str:
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
""")

        example_block = "\n".join(example_text)

    prompt = f"""
You are a dialogue state tracker (DST) for the MultiWOZ 2.2 dataset.

Task:
Given the dialogue history and the latest user utterance, update the belief state.

STRICT OUTPUT RULES (VERY IMPORTANT):
- Output ONLY valid JSON
- Do NOT include explanations, markdown, or code fences
- Output must match this exact JSON structure (same keys, same nesting):
{json.dumps(required_structure, indent=2)}
- Do NOT add new keys or new domains
- Keep previous values unless the user explicitly changes them
- Use "none" if a slot is not specified
- Use "dontcare" if the user has no preference
- If the user specifies a cuisine type, copy it exactly into the "food" slot
- For categorical slots, choose ONLY from the allowed values below
- If the user requests information (e.g. address/phone/postcode), add it to the "request" list

Allowed values:
{json.dumps(allowed, indent=2)}

{example_block}

Now update the belief state for the following turn.

Dialogue history:
{dialogue_history}

Previous belief state:
{json.dumps(prev_belief, indent=2)}

Latest user utterance:
"{user_utterance}"

Return ONLY the updated belief state JSON:
""".strip()

    return prompt