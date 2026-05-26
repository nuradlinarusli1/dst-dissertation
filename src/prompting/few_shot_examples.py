# src/few_shot_examples.py
# This file contains a predefined set of 5 few-shot examples for the DST task,
# Examples can be included in the prompt to provide the LLM with reference examples of how to update the belief state based on user utterances, previous belief state, dialogue history and the expected updated belief state. 

FEW_SHOT_EXAMPLES = [

    # Example 1: Slot Filling (Extract Slot Values from a User Request)
    {
        "dialogue_history": "",
        "prev_belief":{
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
        },
        "user": "I need a place to dine in the centre that is expensive.",
        "belief": {
            "restaurant": {
                "area": "centre",
                "pricerange": "expensive",
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
    },

    # Example 2: Request Slot (User Explicitly Requests Information for a Slot)
    {
        "dialogue_history": "",
        "prev_belief": {
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
        },
        "user": "I need a train from cambridge to london on friday arriving by 18:00.",
        "belief": {
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
                "departure": "cambridge",
                "destination": "london",
                "day": "friday",
                "leaveat": "none",
                "arriveby": "18:00",
                "bookpeople": "none",
                "request": []
            }
        }
    },

    # Example 3: Reference Resolution (User Refers Back to an Entity Mentioned Earlier in the Dialogue History)
    {
        "dialogue_history": "SYSTEM: Frankie and Bennys is an expensive Italian eatery in the south part of town. Would you like any additional information about this restaurant?",
        "prev_belief":{
            "restaurant": {
                "area": "south",
                "pricerange": "expensive",
                "food": "italian",
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
        },
        "user": "address and phone number",
        "belief": {
            "restaurant": {
                "area": "south",
                "pricerange": "expensive",
                "food": "italian",
                "name": "none",
                "bookday": "none",
                "bookpeople": "none",
                "booktime": "none",
                "request": ["phone", "address"]
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
    },

    # Example 4: Booking Details with Reference Resolution (User Provides Booking Details Using Pronouns Referring Back to an Entity Mentioned Earlier in the Dialogue History)
    {
        "dialogue_history": "SYSTEM: The address is Cambridge Leisure Park Clifton Way Cherry Hinton. The phone number is 01223 412430.",
        "prev_belief":{
            "restaurant": {
                "area": "south",
                "pricerange": "expensive",
                "food": "italian",
                "name": "none",
                "bookday": "none",
                "bookpeople": "none",
                "booktime": "none",
                "request": ["phone", "address"]
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
        },
        "user": "Thank you, goodbye.",
        "belief": {
            "restaurant": {
                "area": "south",
                "pricerange": "expensive",
                "food": "italian",
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
    },

    # Example 5: Booking Details with Reference Resolution and Multiple Slot Updates (User Provides Booking Details Using Pronouns Referring Back to an Entity Mentioned Earlier in the Dialogue History, and Multiple Slots are Updated Based on the User Utterance)
    {
        "dialogue_history": "USER: Sounds good, could I get that phone number? Also, could you recommend me an expensive hotel?\nSYSTEM: Bedouin's phone is 01223367660. As far as hotels go, I recommend the University Arms Hotel in the center of town.\n USER:Yes. Can you book it for me?\nSYSTEM: Sure, when would you like that reservation?",
        "prev_belief": {
            "restaurant": {
                "area": "centre",
                "pricerange": "expensive",
                "food": "none",
                "name": "bedouin",
                "bookday": "none",
                "bookpeople": "none",
                "booktime": "none",
                "request": []
            },
            "hotel": {
                "pricerange": "expensive",
                "type": "hotel",
                "area": "none",
                "name": "university arms hotel",
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
        },
        "user": "i want to book it for 2 people and 2 nights starting from saturday.",
        "belief": {
            "restaurant": {
                "area": "centre",
                "pricerange": "expensive",
                "food": "none",
                "name": "bedouin",
                "bookday": "none",
                "bookpeople": "none",
                "booktime": "none",
                "request": []
            },
            "hotel": {
                "pricerange": "expensive",
                "type": "hotel",
                "area": "none",
                "name": "university arms hotel",
                "bookday": "saturday",
                "bookpeople": "2",
                "bookstay": "2",
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
    }
]