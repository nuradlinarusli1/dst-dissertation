#src/schema.py

def get_initial_belief_state():
    return {
        "restaurant": {
            "area": "none",
            "pricerange": "none",
            "food": "none",
            "name": "none",
            "bookday": "none",
            "bookpeople": "none",
            "booktime": "none",
            "request": []
        },
        "hotel": {
            "pricerange": "none",
            "type": "none",
            "area": "none",
            "name": "none",
            "bookday": "none",
            "bookpeople": "none",
            "bookstay": "none",
            "internet": "none",
            "parking": "none",
            "stars": "none",
            "request": []
        },
        "train": {
            "departure": "none",
            "destination": "none",
            "day": "none",
            "leaveat": "none",
            "arriveby": "none",
            "bookpeople": "none",
            "request": []
        }
    }

def get_allowed_values():
    return {
        "restaurant": {
            "area": ["centre", "north", "south", "east", "west", "none", "dontcare"],
            "pricerange": ["cheap", "moderate", "expensive", "none", "dontcare"],
            "food": "free-text",
            "name": "free-text",
            "bookday": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "none", "dontcare"],
            "bookpeople": ["1", "2", "3", "4", "5", "6", "7", "8", "none", "dontcare"],
            "booktime": "free-text",
            "request": ["address", "phone", "postcode", "ref"]
        },
        "hotel": {
            "pricerange": ["cheap", "moderate", "expensive", "none", "dontcare"],
            "type": ["hotel", "guesthouse", "none", "dontcare"],
            "area": ["centre", "north", "south", "east", "west", "none", "dontcare"],
            "name": "free-text",
            "bookday": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "none", "dontcare"],
            "bookpeople": ["1", "2", "3", "4", "5", "6", "7", "8", "none", "dontcare"],
            "bookstay": ["1", "2", "3", "4", "5", "6", "7", "8", "none", "dontcare"],
            "internet": ["yes", "no", "free", "none", "dontcare"],
            "parking": ["yes", "no", "free", "none", "dontcare"],
            "stars": ["0","1", "2", "3", "4", "5", "none", "dontcare"],
            "request": ["address", "phone", "postcode", "ref"]
        },
        "train": {
            "departure": ["birmingham new street", "bishops stortford", "broxbourne", "cambridge", "ely", "kings lynn", "leicester", "london kings cross", "london liverpool street", "norwich", "peterborough", "stansted airport", "stevenage", "none", "dontcare"],
            "destination": ["birmingham new street", "bishops stortford", "broxbourne", "cambridge", "ely", "kings lynn", "leicester", "london kings cross", "london liverpool street", "norwich", "peterborough", "stansted airport", "stevenage", "none", "dontcare"],
            "day": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "none", "dontcare"],
            "leaveat": "free-text",
            "arriveby": "free-text",
            "bookpeople": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "none", "dontcare"],
            "request": ["trainid", "ref", "price", "duration"]
        }
    }