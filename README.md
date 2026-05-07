<div align="center">

```
██╗  ██╗██╗███╗   ██╗ ██████╗ ██╗     ██╗███████╗██╗  ██╗
██║  ██║██║████╗  ██║██╔════╝ ██║     ██║██╔════╝██║  ██║
███████║██║██╔██╗ ██║██║  ███╗██║     ██║███████╗███████║
██╔══██║██║██║╚██╗██║██║   ██║██║     ██║╚════██║██╔══██║
██║  ██║██║██║ ╚████║╚██████╔╝███████╗██║███████║██║  ██║
╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚══════╝╚═╝╚══════╝╚═╝  ╚═╝
          ML TUTOR · HINGLISH EDITION
```

**Bhai, ML ko chai-wali language mein samjhao.**

[![Model on HuggingFace](https://img.shields.io/badge/🤗_Model-dhanashree2311%2Fhinglish--ml--llama--1b-yellow?style=for-the-badge)](https://huggingface.co/dhanashree2311/hinglish-ml-llama-1b)
[![Base Model](https://img.shields.io/badge/Base-LLaMA_3.2_1B_Instruct-orange?style=for-the-badge)](https://huggingface.co/unsloth/Meta-Llama-3.2-1B-Instruct)
[![Method](https://img.shields.io/badge/Method-LoRA_Fine--tuning-blueviolet?style=for-the-badge)](https://arxiv.org/abs/2106.09685)
[![Cost](https://img.shields.io/badge/Total_Cost-₹0-brightgreen?style=for-the-badge)]()
[![GPU](https://img.shields.io/badge/Hardware-Free_Colab_T4-blue?style=for-the-badge&logo=googlecolab)]()

---

> *Every ML tutorial is in formal English or textbook Hindi.*
> *Nobody explains it the way your college senior would — over chai, in the language you actually think in.*
>
> **So I built that college senior.**

</div>

---

## 😤 The Problem

Indian engineering students think in **Hinglish and Marathlish**.

You're debugging a PyTorch tensor shape error at 2am and every Stack Overflow answer, every YouTube tutorial, every documentation page — it's all in formal English. Your brain is doing real-time translation while also trying to understand backpropagation.

That's a tax nobody should have to pay.

---

## 🔥 Before vs After

### Base LLaMA 1B (untuned)

```
You: "Bhai, overfitting kya hota hai?"

Model: "Overfitting ek prakriya hai jahan model ko adhik mahatvapoorn
        upyog karne par samasya aati hai... jisase model ka upyog
        karne par viksit upyog kiya jaata hai..."
```

❌ Repetitive gibberish. Technically wrong. Formally Hindi. Useless.

---

### Fine-tuned Hinglish ML Tutor ✅

```
You: "Bhai, overfitting kya hota hai?"

Model: "Jab training data bahut hoti hai aur model bahut zyada
        parameters hai — overfitting hota hai. Solution: dropout,
        early stopping, data augmentation. Bade datasets mein
        overfitting common hota hai."
```

✅ Clean Hinglish. Technically correct. Sounds like an actual senior.

---

## 📊 The Numbers

| Stat | Value |
|------|-------|
| 🧠 Base Model | LLaMA 3.2-1B-Instruct |
| 🔧 Method | LoRA (r=16, 4-bit quantization) |
| ⚡ Framework | Unsloth (2x faster, 60% less VRAM) |
| 💾 Hardware | Free Google Colab T4 |
| 📝 Dataset | 164 hand-crafted Q&A pairs |
| 🏃 Training | 100 steps · ~20 minutes |
| 📉 Loss Drop | ~2.5 → ~1.0 |
| 💸 Total Cost | ₹0 |

---

## 🛠️ How It Was Built

### Step 1 — The Dataset (Most Important Part)

> Dataset quality > quantity. Every single time.

164 hand-crafted Q&A pairs covering:
- Hinglish ML explanations
- Marathlish technical Q&A
- Real questions Indian engineering students actually ask

**Format:**
```json
{
  "instruction": "Bhai, gradient descent kaise kaam karta hai?",
  "output": "Soch ek pahadi pe ball roll kar rahi hai — woh hamesha neeche jaati hai. Gradient descent bhi waisa hi hai. Loss function ka slope calculate karo, phir weights ko thoda us direction mein move karo jahan loss kam ho. Learning rate decide karta hai kitna bada step lena hai."
}
```

**What NOT to do:**
- ❌ Don't use formal Hindi — nobody talks like a textbook
- ❌ Don't skip spot-checking — bad data = bad model
- ❌ Don't make answers too long — keep under 100 words

---

### Step 2 — Load Model with Unsloth + LoRA

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,  # CRITICAL — fits entire model on free T4
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                          # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)
# Only ~1-2% of parameters actually get trained
```

**What NOT to do:**
- ❌ `load_in_4bit = False` — model won't fit in 15GB VRAM
- ❌ LoRA rank > 16 to start — slower, not better

---

### Step 3 — Train: 100 Steps, ~20 Minutes

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = TrainingArguments(
        per_device_train_batch_size = 2,   # free T4 limit
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 100,                   # enough to validate learning
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
    ),
)

trainer.train()
# Watch the loss: should drop from ~2.5 → ~1.0
# Flat loss = broken data format. Fix data, don't increase steps.
```

**What NOT to do:**
- ❌ `max_steps = 1000` on day one — validate first, scale later
- ❌ `batch_size > 2` — you'll get OOM errors
- ❌ Ignore the loss curve — flat = something is broken

---

### Step 4 — Test + Push to HuggingFace

```python
# Test before you upload — capture before/after, that's your proof
FastLanguageModel.for_inference(model)

inputs = tokenizer(
    [alpaca_prompt.format("Bhai, overfitting kya hota hai?", "")],
    return_tensors = "pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 128)
print(tokenizer.batch_decode(outputs))

# Happy with it? Ship it.
model.push_to_hub("dhanashree2311/hinglish-ml-llama-1b")
tokenizer.push_to_hub("dhanashree2311/hinglish-ml-llama-1b")
```

---

## 🚀 Use the Model Right Now

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "dhanashree2311/hinglish-ml-llama-1b",
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

prompt = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

inputs = tokenizer(
    [prompt.format("Random forest aur decision tree mein kya difference hai?", "")],
    return_tensors = "pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
print(tokenizer.batch_decode(outputs, skip_special_tokens = True))
```

Or just try it directly on [🤗 HuggingFace](https://huggingface.co/dhanashree2311/hinglish-ml-llama-1b).

---

## 💡 Key Takeaways

**1. Free GPU is enough.**
A Colab T4 + Unsloth + LoRA = fine-tuned 1B model in under 30 minutes. No paid compute needed.

**2. Dataset quality > quantity.**
164 well-crafted pairs beat thousands of scraped ones. Spot-check your data. Bad data = bad model. No exceptions.

**3. Marathlish is MORE underserved than Hinglish.**
If you're building for Indian engineers, this is the real gap. Nobody is filling it.

**4. Ship the 100-step model.**
A clear before/after comparison with a live model is worth infinitely more than a perfect model nobody sees. Validate fast, share fast.

**5. Unsloth removes 90% of the friction.**
Don't set up HuggingFace training from scratch. Use their notebooks. They exist for a reason.

---

## 📁 Files in This Repo

| File | What it is |
|------|-----------|
| `llama3_2__1b_and_3b__conversational.py` | Full training script — Unsloth + LoRA + SFTTrainer |
| `dataset/` | 164 Hinglish + Marathlish Q&A pairs |
| `model card` | On HuggingFace |

---

## 🔗 Links

- 🤗 **Model:** [huggingface.co/dhanashree2311/hinglish-ml-llama-1b](https://huggingface.co/dhanashree2311/hinglish-ml-llama-1b)
- 📓 **Base Notebook:** [Unsloth LLaMA 3.2 Conversational](https://github.com/unslothai/unsloth)
- 👩‍💻 **Portfolio:** [portfoliodhanashree.vercel.app](https://portfoliodhanashree.vercel.app)

---

## 👩‍💻 Built By

**Dhanashree Bansode** — AI/ML Engineer · Ex ISRO Intern

- 🌐 [portfoliodhanashree.vercel.app](https://portfoliodhanashree.vercel.app)
- 💼 [linkedin.com/in/dhanashree2311](https://linkedin.com/in/dhanashree2311)
- 🐙 [github.com/dhanashree23112003](https://github.com/dhanashree23112003)

---

<div align="center">

*Built for every Indian engineering student who ever thought in Hinglish*
*but had to Google in English.*

**Samjha? Ab khud try karo. 🚀**

</div>
