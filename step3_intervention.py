import json
import random

# --- Intervention Function ---
def intervene_summary(summary: str) -> str:
    """
    Modify the discharge summary to insert a plausible intervention.
    Keeps the context realistic but changes key facts.
    """
    interventions = [
        # Diagnosis-related
        ("pneumonia", "diagnosed with mild bronchitis instead of pneumonia"),
        ("heart failure", "diagnosed with mild hypertension instead of heart failure"),
        ("osteoporosis", "no history of osteoporosis"),
        # Lab-related
        ("high creatinine", "creatinine levels were normal"),
        ("elevated WBC", "WBC count was within normal range"),
        ("low hemoglobin", "hemoglobin was in the normal range"),
        # General
        ("unsuccessful", "successful")
    ]

    intervention_texts = [
        "Patient was started on a new low-sodium diet plan.",
        "Follow-up care included physiotherapy sessions twice a week.",
        "Patient's medication regimen was adjusted to reduce side effects.",
        "Blood pressure monitoring was emphasized for at-home care.",
        "Patient's exercise plan was modified for gradual recovery.",
    ]

    new_summary = summary
    replaced = False

    for old_phrase, new_phrase in interventions:
        if old_phrase in new_summary.lower():
            new_summary = new_summary.replace(old_phrase, new_phrase)
            replaced = True

    if not replaced:
        print(f"random intervention")
        # If no targeted phrase found, append a random plausible intervention
        new_summary += " " + random.choice(intervention_texts)

    return new_summary

# --- Main Step 5 Processing ---
input_file = "data/step2.jsonl"  # Output from Step 4
output_file = "data/step3_intervention.jsonl"

# Load Step 4 data
with open(input_file, "r") as f:
    data = [json.loads(line) for line in f]

# Apply intervention to all samples
for sample in data:
    if "discharge_summary" in sample:
        sample["discharge_summary_intervened"] = intervene_summary(sample["discharge_summary"])

# Save modified data
with open(output_file, "w") as f:
    json.dump(data, f, indent=2)

print(f"Intervention applied to {len(data)} samples. Saved to {output_file}")
