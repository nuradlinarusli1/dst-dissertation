#src/schema.py

def get_initial_belief_state():
    return {
        "restaurant": {
            "area": "none",
            "pricerange": "none",
            "food": "none",
            "request": []
        }
    }

def get_allowed_values():
    return {
        "restaurant": {
            "area": ["centre", "north", "south", "east", "west", "none", "dontcare"],
            "pricerange": ["cheap", "moderate", "expensive", "none", "dontcare"],
            "food": ["free-text", "none", "dontcare"],
            "request": ["address", "phone", "postcode", "area", "pricerange", "food"]
        }
    }