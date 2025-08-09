import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import json

# Load model and tokenizer
model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)

qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load CSV file
input_csv = "data/step1.csv"
df = pd.read_csv(input_csv)

PROMPT_TEMPLATE = """
You are an expert in clinical reasoning. Your task is to generate multi-entity, medically factual and causal QA pairs based on a discharge summary.

Instructions:
- Each QA pair should involve at least **two distinct medical entities** (e.g., a diagnosis and a lab result, or a diagnosis and a medication).
- The **question** should ask about the reason or cause for both entities or an outcome that results from their interaction.
- The **answer** should explain the **causal relationship** connecting the entities and the outcome.
- Then, generate a **counterfactual version** of the question and answer, explaining what would happen if one or both entities were not present.

Format:

Discharge Summary:
{discharge_summary}

Output format:
Factual QA:
Q: <Multi-entity causal question>
A: <Answer explaining how both entities caused or contributed to outcome>

Counterfactual QA:
Q: <What if... question removing one or both entities>
A: <Answer explaining how absence of one/both entities changes outcome>
"""

df = df.head(2)
results = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    summary = row["discharge_summary"]
    prompt = PROMPT_TEMPLATE.format(discharge_summary=summary)

    try:
        output = qa_pipeline(prompt, max_new_tokens=2048, do_sample=False)[0]['generated_text']
        results.append({
            "row_id": idx,
            "discharge_summary": summary,
            "qa_output": output
        })
    except Exception as e:
        results.append({
            "row_id": idx,
            "discharge_summary": summary,
            "qa_output": f"Error: {str(e)}"
        })

out_path = "data/step24_multientity_QA.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Saved QA results to {out_path}")
