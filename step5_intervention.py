import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import difflib

# === Model setup ===
model_id = "Qwen/Qwen3-1.7B"
print("🔁 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# === Data Setup ===
qa_data = [
    {
        "subject_id": 10229390,
        "original_summary": "The patient underwent hemorrhoidectomy. Post-operative pain was managed using acetaminophen and oxycodone.",
        "intervened_summary": "The patient underwent hemorrhoidectomy. The discharge medications included acetaminophen and oxycodone.",
        "factual_question": "Why was acetaminophen prescribed as a discharge medication?",
        "counterfactual_question": "What if acetaminophen was not prescribed?"
    }
]

def ask_qwen(summary, question):
    prompt = f"""
You are a clinical reasoning assistant.

Given the discharge summary, answer the following question as accurately and concisely as possible.

Discharge Summary:
\"\"\"{summary.strip()}\"\"\"

Question: {question}
Answer:"""
    response = generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)[0]['generated_text']
    return response[len(prompt):].strip()

# === Run interventional tests ===
results = []

for item in qa_data:
    subj_id = item["subject_id"]
    print(f"\n🧪 Testing subject_id {subj_id}")

    # Step 1: Original Summary
    fact_original = ask_qwen(item["original_summary"], item["factual_question"])
    cf_original = ask_qwen(item["original_summary"], item["counterfactual_question"])

    # Step 2: Intervened Summary
    fact_new = ask_qwen(item["intervened_summary"], item["factual_question"])
    cf_new = ask_qwen(item["intervened_summary"], item["counterfactual_question"])

    print(f"{fact_original=}\n{cf_original=}\n{fact_new=}\n{cf_new=}")
    # Step 3: Compare
    fact_changed = fact_original.strip() != fact_new.strip()
    cf_changed = cf_original.strip() != cf_new.strip()

    # Step 4: Save
    results.append({
        "subject_id": subj_id,
        "changed_factual": fact_changed,
        "changed_counterfactual": cf_changed,
        "factual_original": fact_original,
        "factual_intervened": fact_new,
        "counterfactual_original": cf_original,
        "counterfactual_intervened": cf_new
    })

# === Save results ===
df_out = pd.DataFrame(results)
df_out.to_csv("step5_interventional_results.csv", index=False)
print("\n✅ Results saved to step5_interventional_results.csv")
