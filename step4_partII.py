import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time

# === CONFIG ===
input_jsonl = "data/step24_multientity_QA_filtered2.jsonl"
output_csv = "data/step4_part2.csv"
model_id = "Qwen/Qwen3-8B"
max_samples = 2  # limit to first 2 rows for now

# === Load model ===
print("🔁 Loading Qwen model...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# === Load QA pairs from JSONL ===
rows = []
with open(input_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))
df = pd.DataFrame(rows)
df = df.head(max_samples)

# Parse JSON string in qa_output if it's in structured form
def try_parse_qa(qa_text):
    try:
        return json.loads(qa_text)
    except:
        return None

# === Evaluation prompt ===
def build_eval_prompt(discharge_summary, factual_q, factual_a, counter_q, counter_a):
    return f"""
You are a clinical QA evaluator.

Evaluate whether the following multi-entity causal question-answer pairs are factually correct and logically consistent with the discharge summary context.
Provide a short justification and mark each answer as Yes or No.

discharge summary: {discharge_summary}

Factual QA:
Q1: {factual_q}
A1: {factual_a}

Counterfactual QA:
Q2: {counter_q}
A2: {counter_a}

Finally, Respond strictly in this format:
Factual Answer Correct: Yes/No
Counterfactual Answer Correct: Yes/No
Comment: <short explanation>
"""

results = []

# === Loop through QA pairs and evaluate ===
print(f"🧠 Evaluating {len(df)} QA pairs...")
for row_id, row in df.iterrows():
    # print(row)
    # parsed = try_parse_qa(row)
    parsed = row
    # if parsed:
    factual_q = parsed.get("factual_q", "")
    factual_a = parsed.get("factual_a", "")
    counter_q = parsed.get("counter_q", "")
    counter_a = parsed.get("counter_a", "")
    discharge_summary = parsed.get("discharge_summary", "")
    # else:
    #     print(f"⚠️ Skipping row_id {row['row_id']} due to unparsed qa_output")
    #     continue

    prompt = build_eval_prompt(discharge_summary, factual_q, factual_a, counter_q, counter_a)

    try:
        output = generator(prompt, max_new_tokens=2048, do_sample=True, temperature=0.7)[0]['generated_text']
        result_text = output[len(prompt):].strip()
        print(f"{result_text=}\n")
        factual_correct = "Yes" if "Factual Answer Correct: Yes" in result_text else "No"
        counterfactual_correct = "Yes" if "Counterfactual Answer Correct: Yes" in result_text else "No"

        comment_start = result_text.find("Comment:")
        comment = result_text[comment_start + 8:].strip() if comment_start != -1 else ""

        results.append({
            "row_id": row_id,
            "factual_correct": factual_correct,
            "counterfactual_correct": counterfactual_correct,
            "qwen_comment_summary": comment
        })

        print(f"✅ Evaluated row_id {row['row_id']}")
        time.sleep(1)  # avoid rate limit

    except Exception as e:
        print(f"❌ Error evaluating row_id {row_id}: {e}")
        results.append({
            "row_id": row_id,
            "factual_correct": "Error",
            "counterfactual_correct": "Error",
            "qwen_comment_summary": str(e)
        })

# === Save results ===
pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"\n✅ All evaluations complete. Results saved to {output_csv}")
