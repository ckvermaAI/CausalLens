import os
import json
from tqdm import tqdm
from sglang import Runtime, ChatCompletion

# Initialize SGLang runtime with Qwen
rt = Runtime(model="Qwen3-1.5B")

# Load input data (discharge summaries only)
with open("data/step1_discharge_summaries.json", "r") as f:
    discharge_data = json.load(f)

qa_pairs = []

# System prompt for causal extraction and QA generation
system_prompt = """
You are a clinical expert. Your task is to extract causal relationships from medical discharge summaries and generate factual and counterfactual QA pairs.

Step 1: Read the discharge summary.
Step 2: Extract 1 or 2 causal statements (in Cause → Effect format).
Step 3: For each, generate:
  - Factual Question and Answer.
  - Counterfactual Question and Answer (negate the cause).

Be medically realistic and accurate.
Output format:
Causal 1:
Cause: <...>
Effect: <...>
Factual Q: <...>
Factual A: <...>
Counterfactual Q: <...>
Counterfactual A: <...>

Causal 2:
...
"""

for row in tqdm(discharge_data):
    patient_id = row["subject_id"]
    discharge_summary = row["discharge_summary"]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Discharge Summary:\n{discharge_summary.strip()}"},
    ]

    try:
        output = ChatCompletion.create(rt, messages=messages, max_tokens=1024)
        answer = output['choices'][0]['message']['content']

        qa_pairs.append({
            "subject_id": patient_id,
            "discharge_summary": discharge_summary,
            "qa_output": answer
        })

    except Exception as e:
        print(f"Error processing patient {patient_id}: {e}")

# Save output
# os.makedirs("output", exist_ok=True)
with open("data/step2_part2_multientity_qa_llm.json", "w") as f:
    json.dump(qa_pairs, f, indent=2)
