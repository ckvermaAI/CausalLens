import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from sentence_transformers import SentenceTransformer, util
model_emb = SentenceTransformer('all-MiniLM-L6-v2')

def is_meaningfully_changed(orig, new, threshold=0.85):
    emb1 = model_emb.encode(orig, convert_to_tensor=True)
    emb2 = model_emb.encode(new, convert_to_tensor=True)
    sim = util.pytorch_cos_sim(emb1, emb2).item()
    return sim < threshold


# Load the data from the input file
input_file = "data/step3_intervention.jsonl"
with open(input_file, "r") as f:
    data = json.load(f)

# === Model setup ===
model_id = "Qwen/Qwen3-8B"
print("🔁 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

qa_data = data

def ask_qwen(summary, question):
    prompt = f"""
You are a clinical reasoning assistant.

Given the discharge summary, answer the following question as accurately and concisely as possible.

Discharge Summary:
\"\"\"{summary.strip()}\"\"\"

Question: {question}
Answer:"""
    response = generator(prompt, max_new_tokens=64, do_sample=True, temperature=0.7)[0]['generated_text']
    return response[len(prompt):].strip()

# === Run interventional tests ===
results = []

for item in qa_data:
    # Step 1: Original Summary
    fact_original = ask_qwen(item["discharge_summary"], item["factual_q"])
    cf_original = ask_qwen(item["discharge_summary"], item["counter_q"])

    # Step 2: Intervened Summary
    fact_new = ask_qwen(item["discharge_summary_intervened"], item["factual_q"])
    cf_new = ask_qwen(item["discharge_summary_intervened"], item["counter_q"])

    print(f"{fact_original=}\n{cf_original=}\n{fact_new=}\n{cf_new=}")
    # Step 3: Compare
    fact_changed = is_meaningfully_changed(fact_original, fact_new)
    cf_changed = is_meaningfully_changed(cf_original, cf_new)

    # Step 4: Save
    results.append({
        "changed_factual": fact_changed,
        "changed_counterfactual": cf_changed,
        "factual_original": fact_original,
        "factual_intervened": fact_new,
        "counterfactual_original": cf_original,
        "counterfactual_intervened": cf_new
    })

# === Save results ===
out_path = "./data/step4_evaluation.jsonl"
with open(out_path, "w") as f:
    for item in results:
        f.write(json.dumps(item, indent=2) + "\n")
print(f"\n✅ Results saved to {out_path}")
