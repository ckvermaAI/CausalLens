# CausalLens: Probing LLMs’ Clinical Reasoning through QA-Driven Causal Inference

## Step 1: Data extraction 
Execute the step1_extract_data.sql in Google BigQuery or locally to extract the discharge summary for 50 random patients. 

## Step 2: QA generation 
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

## Step 3: Intervention
```bash
Use the file (single and multi entity QAs) generated from step2 and intervene the discharge summary
python step3_intevention.py
```

## Step 4: Evaluation
```bash
# Generate and evaluate the qwen response for the generated QA using both original and intervened discharge summary
python step4_evaluation.py
```
