#src/few_shot_examples.py

FEW_SHOT_EXAMPLES = [
    {
        "dialogue_history": "",
        "prev_belief":{
            "restaurant": {
                "area": "none",
                "pricerange": "none",
                "food": "none",
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
                "request": []
            }
        }
    },
    {
        "dialogue_history": "",
        "prev_belief":{
            "restaurant": {
                "area": "none",
                "pricerange": "none",
                "food": "polynesian",
                "request": []
            }
        },

        #Slot Correction (Update an Existing Slot Value)
        "user": "Any other kind of food is fine.",
        "belief": {
            "restaurant": {
                "area": "none",
                "pricerange": "none",
                "food": "dontcare",
                "request": []
            }
        },
    },
    {
        "dialogue_history": "",
        "prev_belief":{
            "restaurant": {
                "area": "centre",
                "pricerange": "moderate",
                "food": "none",
                "request": []
            }
        },

        #Request Slot (Add Information Requests)
        "user": "Could I get the phone number of the restaurant?",
        "belief": {
            "restaurant": {
                "area": "centre",
                "pricerange": "moderate",
                "food": "none",
                "request": ["phone"]
            }
        }
    }
]