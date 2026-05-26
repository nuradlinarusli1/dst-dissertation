# src/models/rule_based.py
# Rule-based DST baseline implementation for the MultiWOZ 2.2 dataset.
# Uses simple keyword matching and pattern recognition to update the belief state based on the user utterance and current belief state.
# Baseline comparison against LLM-based DST

import re
import json
from schema import get_allowed_values

ALLOWED_VALUES = get_allowed_values()

# Keywords used to detect request slots from user utterances.
REQUEST_KEYWORDS = {
    "address": ["address", "location", "where is", "located"],
    "phone": ["phone", "number", "contact", "phone number", "contact number"],
    "postcode": ["postcode", "postal code", "post code"],
    "ref": ["reference", "ref", "ref number", "booking reference"],
    "trainid": ["train id", "train number", "id number"],
    "price": ["price", "cost", "how much"],
    "duration": ["duration", "how long", "journey time"],
}

# Trigger words used to identigy which domain is active in a user utterance
ACTIVE_DOMAIN = {
    "restaurant": ["restaurant", "food", "eat", "dining", "cuisine", "meal", "lunch", "dinner"],
    "hotel": ["hotel", "guesthouse", "guest house", "accommodation", "stay", "room", "night"],
    "train": ["train", "station", "rail", "railway", "depart", "arrive", "leave", "travel", "journey", "ticket"],
}


# Convert numbers to digit strings for slot value normalisation
WORD_TO_NUM = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
}

def word_boundary_match(value: str, text: str) -> bool:
    # Check if value appears as a whole word in text (avoid partial matches)
    pattern = rf"\b{re.escape(value.lower())}\b"
    return re.search(pattern, text) is not None

def detect_active_domains(text: str, current_belief: dict) -> set:
    # Identify active domains from trigger words in the user utterance
    # Falls back to domains with non-empty belief state slots if no triggers found
    active = set()
    for domain, triggers in ACTIVE_DOMAIN.items():
        if any(trigger in text for trigger in triggers):
            active.add(domain)
    
    if active:
        return active
    
    # If no trigger words found, consider domains with any non-empty slot (including requests) as active.
    for domain, slots in current_belief.items():
        for slot, value in slots.items():
            if slot == "request" and value:
                active.add(domain)
                break
            if slot != "request" and value != "none":
                active.add(domain)
                break

    return active

def extract_time(text: str) -> str | None:
    # Extract time in HH:MM format from text using regex
    # Returns None if no valid time found.
    match = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        return f"{hour:02d}:{minute:02d}"
    return None

def extract_number(text: str) -> str | None:
    # Extract a number from text
    # Returns None if no valid number found.
    match = re.search(r"\b(\d+)\b", text)
    if match:
        return match.group(1)
    for word, num in WORD_TO_NUM.items():
        if re.search(rf"\b{word}\b", text):
            return num
    return None

def extract_train_station(text: str, slot: str) -> str | None:
    # Extract train station names from text based on allowed values for departure/destination slots.
    # Returns None if no valid station name found.
    text = text.lower()

    for location in ALLOWED_VALUES["train"][slot]:
        if location in ["none", "dontcare"]:
            continue
        if word_boundary_match(location, text):
            return location

    return None

def rule_based_predict(utterance: str, current_belief: dict) -> dict:
    # Update the belief state using rule-based slot filling
    # Processes each active domain and slot type
    # Request slots are detected based on keywords via REQUEST_KEYWORDS
    # List slots are matched against allowed values using word boundary matching
    # Free-text slots are updated using simple regex patterns (time, station, number, name, food)
    belief = json.loads(json.dumps(current_belief))
    text = utterance.lower()

    active_domains = detect_active_domains(text, current_belief)

    for domain, slots in ALLOWED_VALUES.items():
        if domain not in active_domains:
            continue

        for slot, values in slots.items():

            if slot == "request":
                for req_slot, keywords in REQUEST_KEYWORDS.items():
                    if req_slot in ALLOWED_VALUES[domain]["request"]:
                        if any(keyword in text for keyword in keywords):
                            if req_slot not in belief[domain]["request"]:
                                belief[domain]["request"].append(req_slot)

            elif isinstance(values, list):
                # Skip already filled slots
                if belief[domain][slot] != "none":
                    continue

                for value in values:
                    if value in ["none", "dontcare"]:
                        continue
                    if word_boundary_match(value, text):
                        belief[domain][slot] = value
                        break

            elif values == "free-text":
                # Skip already filled slots
                if belief[domain][slot] != "none":
                    continue

                if slot in {"leaveat", "arriveby", "booktime"}:
                    time_value = extract_time(text)
                    if time_value:
                        belief[domain][slot] = time_value

                elif slot in {"departure", "destination"}:
                    station_value = extract_train_station(text, slot)
                    if station_value:
                        belief[domain][slot] = station_value

                elif slot in {"bookpeople", "bookstay", "stars"}:
                    number_value = extract_number(text)
                    if number_value:
                        belief[domain][slot] = number_value

                elif slot == "name":
                    match = re.search(
                        r'\bcalled\s+"?([a-z0-9\s&\'-]+?)"?(?=[,.!?]|$)', text
                    )
                    if match:
                        belief[domain][slot] = match.group(1).strip()

                elif slot == "food":
                    match = re.search(
                        r"(?:serves?|serving|eat|cuisine|food)\s+(?:is\s+)?(\w+)", text
                    )
                    if match:
                        belief[domain][slot] = match.group(1).strip()

    return belief