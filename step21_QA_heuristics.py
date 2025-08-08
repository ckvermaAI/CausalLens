import pandas as pd
import random
import re

# === CONFIG ===
input_csv = "data/step1.csv" 
output_csv = "data/step21.csv"
num_patients = 1

# === Helper functions ===

def clean_text(text):
    if pd.isnull(text):
        return ""
    return text.replace("\n", " ").replace(";", ",").strip()

def extract_first_item(array_str):
    """Extract the first item from semicolon-separated string"""
    if pd.isnull(array_str) or not isinstance(array_str, str):
        return None
    return array_str.split(";")[0].strip()

def extract_related_lab(medication, lab_results):
    if not isinstance(lab_results, str):
        return None
    med = medication.lower()
    lab_entries = lab_results.split(";")
    for entry in lab_entries:
        if med in entry.lower():
            return entry.strip()
    if "glucose" in lab_results.lower() and med in ["insulin", "metformin"]:
        return "Glucose: high"
    if "wbc" in lab_results.lower() and med in ["vancomycin", "cefepime"]:
        return "WBC: high"
    return None

def generate_factual_question(medication):
    return f"Why was {medication.lower()} administered?"

def generate_counterfactual_question(medication, cause):
    return f"Would {medication.lower()} still be administered if the patient did not have {cause.lower()}?"

# === Load and process data ===

df = pd.read_csv(input_csv)
df = df.head(num_patients)

qa_rows = []

for _, row in df.iterrows():
    subject_id = row['subject_id']
    discharge_summary = clean_text(row['discharge_summary'])
    meds = row.get('medications', "")
    dxs = row.get('diagnoses', "")
    labs = row.get('lab_results', "")

    med_list = [m.strip() for m in meds.split(";") if m.strip()]
    dx_list = [d.strip() for d in dxs.split(";") if d.strip()]

    for med in med_list:
        # Factual Q
        question = generate_factual_question(med)
        
        # Naive answer heuristic
        answer = ""
        if med.lower() in discharge_summary.lower():
            match = re.search(rf"{med}.*?because(.*?)\.", discharge_summary, re.IGNORECASE)
            answer = "Because" + match.group(1).strip() if match else "As mentioned in discharge summary."
        elif dx_list:
            answer = f"To treat {dx_list[0].lower()}."
        else:
            answer = "Clinical reason not explicitly found."

        # Counterfactual Q
        related_lab = extract_related_lab(med, labs) or dx_list[0] if dx_list else "the condition"
        counter_q = generate_counterfactual_question(med, related_lab)
        counter_a = "Probably not." if related_lab else "Cannot determine without clinical context."

        qa_rows.append({
            "subject_id": subject_id,
            "medication": med,
            "factual_question": question,
            "factual_answer": answer,
            "counterfactual_question": counter_q,
            "counterfactual_note": counter_a
        })

# === Save output ===
qa_df = pd.DataFrame(qa_rows)
qa_df.to_csv(output_csv, index=False)
print(f"✅ Saved {len(qa_df)} QA pairs to {output_csv}")
