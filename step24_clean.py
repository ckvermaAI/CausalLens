import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import json
from collections import defaultdict

# --------------------
# Load model and tokenizer
# --------------------
model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)

qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# --------------------
# Load CSV file
# --------------------
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

# --------------------
# Run generation
# --------------------
df = df.head(2)  # for testing
raw_results = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    summary = row["discharge_summary"]
    prompt = PROMPT_TEMPLATE.format(discharge_summary=summary)

    try:
        output = qa_pipeline(prompt, max_new_tokens=2048, do_sample=False)[0]['generated_text']
        raw_results.append({
            "row_id": idx,
            "discharge_summary": summary,
            "qa_output": output
        })
    except Exception as e:
        raw_results.append({
            "row_id": idx,
            "discharge_summary": summary,
            "qa_output": f"Error: {str(e)}"
        })

# --------------------
# Deduplication + Diversity Filtering
# --------------------
def parse_qa_output(output_text, discharge_summary=None):
    """Quick parser to extract QAs from generated text, including discharge_summary."""
    blocks = []
    lines = output_text.split("\n")
    current = {}
    for line in lines:
        if line.startswith("Factual QA:"):
            current = {"factual_q": "", "factual_a": "", "counter_q": "", "counter_a": ""}
        elif line.startswith("Q:") and "factual_q" in current and not current["factual_q"]:
            current["factual_q"] = line[2:].strip()
        elif line.startswith("A:") and "factual_a" in current and not current["factual_a"]:
            current["factual_a"] = line[2:].strip()
        elif line.startswith("Counterfactual QA:"):
            continue
        elif line.startswith("Q:") and "counter_q" in current and not current["counter_q"]:
            current["counter_q"] = line[2:].strip()
        elif line.startswith("A:") and "counter_a" in current and not current["counter_a"]:
            current["counter_a"] = line[2:].strip()
            if discharge_summary is not None:
                current["discharge_summary"] = discharge_summary
            blocks.append(current)
    return blocks

# Flatten parsed QAs
parsed_qas = []
for r in raw_results:
    parsed_qas.extend(parse_qa_output(r["qa_output"], r["discharge_summary"]))

# Deduplicate
seen = set()
deduped_qas = []
for qa in parsed_qas:
    key = (qa["factual_q"].lower(), qa["factual_a"].lower(), qa["counter_q"].lower(), qa["counter_a"].lower())
    if key not in seen:
        seen.add(key)
        deduped_qas.append(qa)

print(f"After deduplication: {len(deduped_qas)} QAs (from {len(parsed_qas)})")

# Diversity filter
def get_causal_type(factual_answer):
    fact = factual_answer.lower()
    if "due to" in fact or "because" in fact:
        if "procedure" in fact or "ectomy" in fact or "surgery" in fact:
            return "procedure→complication"
        elif "diagnosis" in fact or "disease" in fact:
            return "diagnosis→effect"
        elif "medication" in fact or "mg" in fact:
            return "medication→effect"
    return "other"

type_counts = defaultdict(int)
max_fraction = 0.4
total_target = len(deduped_qas)

final_qas = []
for qa in deduped_qas:
    ctype = get_causal_type(qa["factual_a"])
    if type_counts[ctype] / total_target <= max_fraction:
        final_qas.append(qa)
        type_counts[ctype] += 1

print(f"After diversity filtering: {len(final_qas)} QAs")

# --------------------
# Save Final Output
# --------------------
out_path = "data/step24_multientity_QA_filtered2.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for qa in final_qas:
        f.write(json.dumps(qa, ensure_ascii=False) + "\n")

print(f"Saved filtered QAs to {out_path}")
