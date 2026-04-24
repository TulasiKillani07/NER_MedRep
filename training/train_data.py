# train_data.py
# Seed training examples for the NER model
#
# Entity labels used:
#   SYMPTOM    — fever, headache, knee pain, cough
#   INDICATION — diabetes, hypertension, asthma (known conditions/diagnoses)
#   SEVERITY   — mild, severe, acute, chronic
#   DURATION   — 3 days, since yesterday, for a week
#
# Format: (sentence, {"entities": [(start_char, end_char, label)]})
# start_char and end_char are character positions in the sentence
#
# HOW TO FIND CHARACTER POSITIONS:
#   text = "i have fever"
#   text.index("fever") -> 7
#   7 + len("fever") -> 12
#   so: (7, 12, "SYMPTOM")

TRAIN_DATA = [
    (
        "i have fever headache and knee pain from 3 days",
        {"entities": [(7, 12, "SYMPTOM"), (13, 21, "SYMPTOM"), (26, 35, "SYMPTOM"), (41, 47, "DURATION")]},
    ),
    (
        "i am suffering from fever and cough since yesterday",
        {"entities": [(20, 25, "SYMPTOM"), (30, 35, "SYMPTOM"), (36, 51, "DURATION")]},
    ),
    (
        "having severe headache from 2 days",
        {"entities": [(7, 13, "SEVERITY"), (14, 22, "SYMPTOM"), (28, 34, "DURATION")]},
    ),
    (
        "i have mild fever and body ache",
        {"entities": [(7, 11, "SEVERITY"), (12, 17, "SYMPTOM"), (22, 31, "SYMPTOM")]},
    ),
    (
        "no fever but i have chest pain",
        {"entities": [(3, 8, "SYMPTOM"), (19, 29, "SYMPTOM")]},
    ),
    (
        "i have been having diarrhea and vomiting for 3 days",
        {"entities": [(19, 27, "SYMPTOM"), (32, 40, "SYMPTOM"), (45, 51, "DURATION")]},
    ),
    (
        "shortness of breath since last week",
        {"entities": [(0, 19, "SYMPTOM"), (20, 35, "DURATION")]},
    ),
    (
        "i have back pain and fatigue from 1 week",
        {"entities": [(7, 16, "SYMPTOM"), (21, 28, "SYMPTOM"), (34, 40, "DURATION")]},
    ),
    (
        "suffering from acute chest pain",
        {"entities": [(15, 20, "SEVERITY"), (21, 31, "SYMPTOM")]},
    ),
    (
        "i have sore throat and runny nose for 2 days",
        {"entities": [(7, 18, "SYMPTOM"), (23, 33, "SYMPTOM"), (38, 44, "DURATION")]},
    ),
    (
        "patient denies fever but has joint pain",
        {"entities": [(15, 20, "SYMPTOM"), (29, 39, "SYMPTOM")]},
    ),
    (
        "i have been feeling dizzy and nauseous since morning",
        {"entities": [(20, 25, "SYMPTOM"), (30, 38, "SYMPTOM"), (39, 52, "DURATION")]},
    ),
    (
        "chronic back pain for 3 months",
        {"entities": [(0, 7, "SEVERITY"), (8, 17, "SYMPTOM"), (22, 30, "DURATION")]},
    ),
    (
        "i have swelling in my knee since 2 weeks",
        {"entities": [(7, 15, "SYMPTOM"), (34, 41, "DURATION")]},
    ),
    (
        "no cough no cold but i have mild fever",
        {"entities": [(3, 8, "SYMPTOM"), (12, 16, "SYMPTOM"), (27, 31, "SEVERITY"), (32, 37, "SYMPTOM")]},
    ),
    (
        "patient has diabetes and is complaining of severe fatigue",
        {"entities": [(12, 20, "INDICATION"), (43, 49, "SEVERITY"), (50, 57, "SYMPTOM")]},
    ),
    (
        "known case of hypertension with headache since 3 days",
        {"entities": [(14, 26, "INDICATION"), (32, 40, "SYMPTOM"), (47, 53, "DURATION")]},
    ),
    (
        "i have asthma and i am having mild breathlessness",
        {"entities": [(7, 13, "INDICATION"), (30, 34, "SEVERITY"), (35, 49, "SYMPTOM")]},
    ),
    (
        "fever and chills for the past 2 days with history of malaria",
        {"entities": [(0, 5, "SYMPTOM"), (10, 16, "SYMPTOM"), (17, 37, "DURATION"), (53, 59, "INDICATION")]},
    ),
    (
        "mild fever and moderate cough since yesterday",
        {"entities": [(0, 4, "SEVERITY"), (5, 10, "SYMPTOM"), (15, 23, "SEVERITY"), (24, 29, "SYMPTOM"), (30, 45, "DURATION")]},
    ),
]
