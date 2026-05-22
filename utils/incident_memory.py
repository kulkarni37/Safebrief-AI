import json
import os

FILE_PATH = "incident_history.json"

# LOAD INCIDENT HISTORY
def load_history():

    if not os.path.exists(FILE_PATH):

        with open(FILE_PATH, "w") as file:
            json.dump([], file)

    with open(FILE_PATH, "r") as file:

        return json.load(file)

# SAVE NEW INCIDENT
def save_incident(summary, solution):

    history = load_history()

    history.append({
        "incident": summary,
        "solution": solution
    })

    with open(FILE_PATH, "w") as file:

        json.dump(history, file, indent=4)

# FIND SIMILAR INCIDENT
def find_similar_incident(text):

    history = load_history()

    text = text.lower()

    for item in history:

        incident = item["incident"].lower()

        if any(word in incident for word in text.split()):

            return item

    return None