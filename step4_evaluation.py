import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time

# === CONFIG ===
input_csv = "qa_pairs_qwen_generated.csv"
output_csv = "evaluation_results.csv"
model_id = "Qwen/Qwen3-1.7B"
max_patients = 1  # Set to an integer to limit how many rows to process

# === Load model ===
print("🔁 Loading Qwen model...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# === Evaluation prompt ===
def build_eval_prompt(subject_id, factual_q, factual_a, counterfactual_q, counterfactual_a):
    return f"""
You are a clinical QA evaluator.

Evaluate whether the following question-answer pairs are accurate and logically consistent with the discharge summary. Provide a short justification and mark each answer as Yes or No.

Subject ID: {subject_id}

Factual QA:
Q1: {factual_q}
A1: {factual_a}

Counterfactual QA:
Q2: {counterfactual_q}
A2: {counterfactual_a}

Respond in this format:
Factual Answer Correct: Yes/No
Counterfactual Answer Correct: Yes/No
Comment: <short explanation>
"""

# === Load QA pairs ===
df = pd.read_csv(input_csv)
if max_patients:
    df = df.head(max_patients)

results = []

# === Loop through QA pairs and evaluate ===
print(f"🧠 Evaluating {len(df)} QA pairs...")
for idx, row in df.iterrows():
    subject_id = row["subject_id"]
    factual_q = row["factual_question"]
    factual_a = row["factual_answer"]
    counterfactual_q = row["counterfactual_question"]
    counterfactual_a = row["counterfactual_answer"]

    prompt = build_eval_prompt(subject_id, factual_q, factual_a, counterfactual_q, counterfactual_a)
    try:
        output = generator(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)[0]['generated_text']
        result_text = output[len(prompt):].strip()

        # Extract answers
        factual_correct = "Yes" if "Factual Answer Correct: Yes" in result_text else "No"
        counterfactual_correct = "Yes" if "Counterfactual Answer Correct: Yes" in result_text else "No"

        comment_start = result_text.find("Comment:")
        comment = result_text[comment_start + 8:].strip() if comment_start != -1 else ""

        results.append({
            "subject_id": subject_id,
            "factual_correct": factual_correct,
            "counterfactual_correct": counterfactual_correct,
            "qwen_comment_summary": comment
        })

        print(f"✅ Evaluated subject_id {subject_id}")
        time.sleep(1)  # Avoid rate limit

    except Exception as e:
        print(f"❌ Error evaluating subject_id {subject_id}: {e}")
        results.append({
            "subject_id": subject_id,
            "factual_correct": "Error",
            "counterfactual_correct": "Error",
            "qwen_comment_summary": str(e)
        })

# === Save results ===
pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"\n✅ All evaluations complete. Results saved to {output_csv}")
