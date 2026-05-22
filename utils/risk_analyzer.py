def predict_risk(text):

    text = text.lower()

    high_keywords = [
        "death",
        "fire",
        "explosion",
        "severe",
        "critical",
        "chemical leak"
    ]

    medium_keywords = [
        "injury",
        "damage",
        "machine failure",
        "slip"
    ]

    for word in high_keywords:
        if word in text:
            return "HIGH"

    for word in medium_keywords:
        if word in text:
            return "MEDIUM"

    return "LOW"