# AIH

## 🧩 **Part I: Single-Entity QA Generation**

| Step  | Description                                                                                         |
| ----- | --------------------------------------------------------------------------------------------------- |
| **1** | Extract discharge summaries, diagnoses, medications, and labs from MIMIC-IV.                        |
| **2** | Generate QA pairs for each entity type (diagnosis, medication, lab) using heuristics and templates. |
| **3** | Use Qwen model to refine/expand factual QA pairs from heuristic outputs.                            |
| **4** | Evaluate QA coverage, quality, and clinical relevance (manual + automated checks).                  |
| **5** | Perform a mini-interventional reasoning test using synthetic variations in inputs.                  |

---

## 🧠 **Part II: Multi-Entity Causal QA Generation**

| Step  | Description                                                                                              |
| ----- | -------------------------------------------------------------------------------------------------------- |
| **1** | Use same extracted discharge summaries (from Part I).                                                    |
| **2** | Extract causal sentences involving multiple entities (e.g., diagnosis + lab) from discharge summaries.   |
| **3** | Use Qwen model to generate both factual and counterfactual QA pairs from causal sentences.               |
| **4** | Extend QA coverage with combinations like (diagnosis + medication) and perform quality control.          |
| **5** | Compare reasoning depth of Qwen and other LLMs across multi-entity QA pairs using simple causal metrics. |

---