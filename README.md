# CausalLens: Probing LLMs’ Clinical Reasoning through QA-Driven Causal Inference [Paper](https://github.com/ckvermaAI/CausalLens/blob/main/CausalLens_paper.pdf)

Large Language Models (LLMs) hold significant potential for advancing clinical decision support, yet their capacity for causal reasoning in healthcare remains underexplored. We introduce CausalLens, a structured framework that generates diverse, clinically grounded causal question–answer (QA) pairs from MIMIC-IV discharge summaries. These pairs encompass medications, laboratory findings, and diagnoses, enabling a targeted evaluation of LLMs’ ability to discern cause-and-effect relationships in complex patient scenarios. By systematically probing LLM responses through controlled interventions, CausalLens exposes limitations in LLM causal reasoning and provides a reproducible pipeline for evaluating performance, enabling the development of safer, more reliable AI in clinical decision support.

## Steps to reproduce results

### Step 1: Data extraction 
Execute the step1_extract_data.sql in Google BigQuery or locally to extract the discharge summary for 50 random patients. 

### Step 2: QA generation 
```bash
# 1) Install the requirements
pip install transformers accelerate

# 2) Use the file generated from step1 and run the following command
# 2A) Generate ten single-entity QAs
python step2_QA_generation.py single 10
# 2B) Generate ten multi-entity QAs
python step2_QA_generation.py multi 10
# Save the results in single file (combine data in two generated jsonl file)
```

### Step 3: Intervention
```bash
Use the file (single and multi entity QAs) generated from step2 and intervene the discharge summary
python step3_intevention.py
```

### Step 4: Evaluation
```bash
# Generate and evaluate the qwen response for the generated QA using both original and intervened discharge summary
python step4_evaluation.py
```
