#src/few_shot_examples.py

FEW_SHOT_EXAMPLES = [
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
            }
        },
        
        #Slot Filling (Extract Slot Values from a User Request)
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
            }
        }
    },
    {
        "dialogue_history": "",
        "prev_belief":{
            "restaurant": {
                "area": "centre",
                "pricerange": "moderate",
                "food": "none",
                "name": "none",
                "bookday": "none",
                "bookpeople": "none",
                "booktime": "none",
                "request": []
            }
        },
        "user": "Could I get the phone number of the restaurant?",
        "belief": {
            "restaurant": {
                "area": "centre",
                "pricerange": "moderate",
                "food": "none",
                "name": "none",
                "bookday": "none",
                "bookpeople": "none",
                "booktime": "none",
                "request": ["phone"]
            }
        }
    },
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
            }
        }
    },
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
            }
        }
    },
    {
        "dialogue_history": "USER: Sounds good, could I get that phone number? Also, could you recommend me an expensive hotel?\nSYSTEM: Bedouin's phone is 01223367660. As far as hotels go, I recommend the University Arms Hotel in the center of town.\nYes. Can you book it for me?\nSYSTEM: Sure, when would you like that reservation?",
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