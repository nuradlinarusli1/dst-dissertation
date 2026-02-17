from schema import get_initial_belief_state
from prompt_builder import build_prompt

history = "USER: hi\nSYSTEM: hello\n"
belief = get_initial_belief_state()
utt = "I'm looking for a cheap restaurant in the centre."

prompt = build_prompt(history, belief, utt)
print(prompt)
