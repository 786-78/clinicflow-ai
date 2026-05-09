# 🏥 ClinicFlow AI

A medical triage and patient intake assistant
fine-tuned on Llama 3.2 3B using LoRA adapters.

---

## 📺 Demo Video

[![ClinicFlow AI Demo](https://img.youtube.com/vi/S9XR_-YyiHQ/0.jpg)](https://youtu.be/S9XR_-YyiHQ)

---

## 🚀 Live Demo

👉 [Try ClinicFlow AI Live](https://huggingface.co/spaces/bilalchawdhary/clinicflow-demo)

---

## 📌 What is ClinicFlow AI

ClinicFlow AI is a responsible clinic triage and
patient intake assistant powered by a fine-tuned
Llama 3.2 3B model.

It conducts multi-turn conversations to collect
patient symptoms, asks focused follow-up questions,
and generates a structured clinical summary at end.

---

## 🔧 Model Details

| Detail | Info |
|--------|------|
| Base Model | Llama 3.2 3B Instruct |
| Fine-tuning | LoRA (PEFT) |
| Training Loss | 0.107 |
| Task | Medical Triage |
| Deployed | HuggingFace Spaces |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python + PyTorch | Core framework |
| HuggingFace Transformers | Model loading |
| TRL 1.3.0 | Fine-tuning |
| PEFT + LoRA | Efficient training |
| Gradio | Chat UI |
| Google Colab A100 | Training GPU |

---

## 📁 Repository Structure

clinicflow-ai/
│
├── README.md
├── requirements.txt
├── training/
│   └── training_notebook.ipynb
├── inference/
│   └── inference_demo.ipynb
└── app/
└── app.py




---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install transformers peft accelerate
pip install gradio torch
```

### 2. Run inference
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_name = "unsloth/Llama-3.2-3B-Instruct"
adapter_name = "bilalchawdhary/clinic_flow"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, adapter_name)

prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are ClinicFlow AI, a responsible clinic triage
and patient intake assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

I have a headache and fever since 2 days<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
input_length = inputs["input_ids"].shape[1]

outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)

new_tokens = outputs[0][input_length:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)
print(response)
```

---

## ✅ Capabilities

- Multi-turn patient intake conversations
- Focused symptom follow-up questions
- Never diagnoses or prescribes treatment
- Structured clinical summary output
- Always recommends real doctor consultation

---

## ⚠️ Disclaimer

ClinicFlow AI is for demonstration purposes only.
It is NOT a replacement for a real doctor.
Always consult a qualified medical professional.

---

## 👨‍💻 Author

**Bilal Chawdhary** — Software Engineering Graduate
building AI products and sharing the journey.

| Link | URL |
|------|-----|
| 🎥 Demo Video | [YouTube](https://youtu.be/S9XR_-YyiHQ) |
| 🤗 Live Demo | [HuggingFace Spaces](https://huggingface.co/spaces/bilalchawdhary/clinicflow-demo) |
| 🧠 Model | [HuggingFace Model](https://huggingface.co/bilalchawdhary/clinic_flow) |
| 💼 LinkedIn | [Bilal Chawdhary]([https://linkedin.com/in/bilalchawdhary)](https://www.linkedin.com/in/muhammad-bilal-nasir-0466611a6/) |
