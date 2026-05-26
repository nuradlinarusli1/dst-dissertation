# src/prompting/prompt_builder.py
# Constructs prompts for the LLM-based DST system.
# Splits the prompt into a fixed prefix (instructions + examples) and a variable suffix (specific dialogue history, previous belief state, and user utterance for the current turn).

import json
from schema import get_allowed_values, get_initial_belief_state
from prompting.few_shot_examples import FEW_SHOT_EXAMPLES

def build_prompt(dialogue_history: str, prev_belief: dict, user_utterance: str, system_utterance: str, use_few_shot: bool = False, n_shots: int = 0) -> tuple[str, str]  :
    
    # Load allowed slot values and the required JSON structure for the belief state.
    allowed = get_allowed_values()
    required_structure = get_initial_belief_state()

    example_block = ""

    if use_few_shot:
        example_text = []

        # Select the first n_shots examples from the predefined FEW_SHOT_EXAMPLES list to include in the prompt.
        selected_examples = FEW_SHOT_EXAMPLES[:n_shots]

        for i, example in enumerate(selected_examples, start=1):
            history = example.get("dialogue_history", "").strip()
            if not history:
                history = "No dialogue history"

            example_text.append(f"""
                              
-------------
Example {i} to reference for how to update the belief state based on the user utterance:
-------------
Dialogue history:
{history}

Previous belief state:
{json.dumps(example["prev_belief"], indent=2)}

User utterance:
"{example["user"]}"

Updated belief state:
{json.dumps(example["belief"], indent=2)}
            """.strip())

        example_block = "\n".join(example_text)

    # Fixed prefix contains the task instructions, strict output rules, state update rules, allowed values for each slot and few-shot examples. 
    # This part does not change across turns.
    fixed_prefix = f"""
You are a dialogue state tracker (DST) for the MultiWOZ 2.2 dataset.

Task:
Given the previous belief state and ONLY latest user utterance, update the belief state.

STRICT OUTPUT RULES (VERY IMPORTANT):
- Output ONLY valid JSON. NO extra text.
- Output must match this exact JSON structure (same keys, same nesting):
{json.dumps(required_structure, indent=2)}

STATE UPDATE RULES (VERY IMPORTANT):
- Start from the previous belief state.
- Update the belief state using ONLY the latest USER utterance.
- Use the dialogue history ONLY for reference resolution, such as understanding what "it", "that one", "there", or "the hotel" refers to in the latest user utterance.
- Do NOT use the dialogue history to introduce new slot values unless the USER explicitly refers back to something earlier.
- Do NOT add or change constraint slot values based only on SYSTEM suggestions, recommendations, or offered options.
- Do NOT replace a USER's existing constraints with specific slot values mentioned by the SYSTEM unless the USER explicitly confirms, selects, or changes them.
- A SYSTEM suggestion, recommendation, or offered option does NOT mean the USER has accepted all of its attributes.
- If the USER refers back to an offered entity (for example "book it", "that one", or "yes"), use dialogue history to identify which entity/domain they are referring to, but do NOT add or change any slot values based on SYSTEM suggestions, recommendations, or offered options.
- The "request" list contains ONLY slots that the USER has explicitly requested information for.
- Do not fill "request" slots with slots the USER did not explicitly ask for.
- Track ONLY these domains: "restaurant", "hotel", and "train". Do NOT include any other domains such as "attraction", "taxi", or "museum".
- Use "none" ONLY for slots that are still unknown and have never been filled.
- Use "dontcare" if the USER has explicitly indicated no preference for a slot, for example "any" or "no preference".
- If the latest USER utterance gives booking details using pronouns such as "book it", "book that", "try for 2 people", "starting Saturday", or "for 2 nights", apply those updates only to the single resolved active domain.
- Slots with the same name across domains, such as area, bookday, or bookpeople, must not be updated in multiple domains.

DO NOT:
- Add any new values that are not in the allowed list for each slot.
- Copy booking details into multiple domain just because slot names overlap. For example, if the USER says "book it for 2 people", only update the single resolved active domain with "bookpeople": "2" and do NOT copy "bookpeople": "2" into multiple domains just because they all have a "bookpeople" slot.

- Allowed values:
{json.dumps(allowed, indent=2)}
- Slots are only updated based on the limited set of allowed values for each slot. Do NOT add any new values that are not in the allowed list for each slot.
    """.strip()

    # Append few-shot examples to the fixed prefix if use_few_shot is True.
    if example_block:
        fixed_prefix += "\n\nBelow are some examples to reference for how to update the belief state based on the user utterance. Use these examples as a guide for how to interpret the dialogue history and user utterance when updating the belief state."
        fixed_prefix += f"\n\n{example_block}"

    # Variable suffix contains the dialogue history, previous belief state, and latest user utterance for the current turn. 
    # This part changes for each turn.
    variable_suffix = f"""
------------------------------
Now update the belief state for the following turn.

Dialogue history:
{dialogue_history}
Use the dialogue history ONLY for reference resolution. Do NOT add or change slot values based only on SYSTEM suggestions, recommendations, or offered options.

Previous belief state:
{json.dumps(prev_belief, indent=2)}

Latest user utterance (ONLY update based on this):
"{user_utterance}"

Return ONLY the updated belief state JSON:
    """.strip()

    return fixed_prefix, variable_suffix

