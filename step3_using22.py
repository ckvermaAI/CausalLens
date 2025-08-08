from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load Qwen3-1.7B
model_id = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# === Evaluation Prompt ===
discharge_summary = """
Patient underwent hemorrhoidectomy. Pain was managed using acetaminophen and oxycodone.
"""

factual_q = "Why was acetaminophen prescribed for the patient after the hemorrhoidectomy?"
factual_a = "Acetaminophen was prescribed to manage postoperative pain."

counterfactual_q = "Why wasn't acetaminophen prescribed instead of oxycodeine?"
counterfactual_a = ("If acetaminophen had not been prescribed, the patient might have required alternative pain relief, "
                    "such as a different analgesic or a combination of medications. However, the patient's pain was controlled "
                    "with acetaminophen, and oxycodeine was used for additional pain management.")

eval_prompt = f"""
You are a medical QA evaluator. Analyze whether the following question-answer pairs are consistent with the discharge summary and logical from a clinical standpoint.

Discharge Summary:
\"\"\"
{discharge_summary.strip()}
\"\"\"

Factual Question:
Q: {factual_q}
A: {factual_a}

Is this answer correct and grounded in the discharge summary? Explain briefly.

Counterfactual Question:
Q: {counterfactual_q}
A: {counterfactual_a}

Is this counterfactual answer medically reasonable and logically consistent? Explain.
"""

# Generate evaluation
output = generator(eval_prompt, max_new_tokens=300, do_sample=True, temperature=0.7)
print("=== Qwen's Evaluation Response ===\n")
print(output[0]['generated_text'][len(eval_prompt):].strip())
