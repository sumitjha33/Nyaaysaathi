STATE_HELPLINES = {
    "delhi": {
        "women_helpline": "181",
        "police": "100",
        "domestic_violence": "1091",
        "legal_aid": "1516",
        "emergency": "112",
        "state_commission": "011-23379181"
    },
    "maharashtra": {
        "women_helpline": "103",
        "police": "100",
        "domestic_violence": "1091",
        "legal_aid": "1516",
        "emergency": "112",
        "state_commission": "022-26592707"
    },
    # Add more states...
}

def get_state_helplines(state: str) -> dict:
    state = state.lower()
    return STATE_HELPLINES.get(state, {
        "women_helpline": "181",
        "police": "100",
        "emergency": "112",
        "legal_aid": "1516"
    })
