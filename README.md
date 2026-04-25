---
title: Personal AI Assistant
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# 🤖 Personal AI Assistant — RL Environment

**🔴 Live Demo: https://jyothisha426-personal-ai-assistant.hf.space**

An **OpenEnv-compatible reinforcement learning environment** where an AI agent learns to handle real-life personal and professional tasks.

**Theme**: World Modeling — Personalized Tasks (Theme 3.2)
**Bonus theme**: Consumer Workflows with Schema Drift (Patronus AI)

---

## What Makes This Stand Out

- **75 unique scenarios** across 5 real-world task types
- **Hybrid grading**: `0.40 × keyword_score + 0.60 × Groq Llama-3 LLM-as-Judge`
- **Graceful fallback**: keyword-only if Groq API unavailable — training never breaks
- **Real GRPO training**: reward improved **0.7675 → 0.7917** (+0.0242, 60 steps, free T4 GPU)
- **Interactive Gradio UI**: try any of the 75 scenarios live at the link above

---

## Tasks

| Task                     | Scenarios | What the agent must do                                           |
| ------------------------ | --------- | ---------------------------------------------------------------- |
| `tough_email_reply`      | 15        | Write professional, empathetic replies to difficult emails       |
| `schedule_conflict`      | 15        | Resolve messy scheduling conflicts with clear priority decisions |
| `personal_message`       | 15        | Reply to personal/WhatsApp-style messages with appropriate tone  |
| `dinner_travel_planning` | 15        | Create concrete plans satisfying all stated constraints          |
| `schema_drift`           | 15        | Adapt mid-task when rules or context change (Patronus bonus)     |

**Total: 75 unique scenarios** across 5 tasks.

---

## Hybrid Grading System

```
Final Score = 0.40 × keyword_score + 0.60 × LLM-as-Judge (Groq Llama-3-8b)
```

Each task's keyword score is multi-dimensional:

**Task 1 — Tough Email Reply**: 25% empathy · 25% relevance · 20% professionalism · 20% actionability · 10% length

**Task 2 — Schedule Conflict**: 35% coverage · 30% must-include items · 25% decision clarity · 10% length

**Task 3 — Personal Message**: 40% empathy keywords · 30% avoids inappropriate phrases · 15% non-refusal · 15% length

**Task 4 — Dinner & Travel Planning**: 40% constraint satisfaction · 30% concreteness · 15% actionability · 15% length

**Task 5 — Schema Drift**: 40% concrete alternative · 30% acknowledges change · 20% actionable resolution · 10% length

All rewards are continuous in **(0.0001, 0.9999)**.

---

## Training Results

| Metric         | Value                                          |
| -------------- | ---------------------------------------------- |
| Model          | `Qwen/Qwen2.5-0.5B-Instruct` (4-bit quantized) |
| Algorithm      | GRPO (HuggingFace TRL GRPOTrainer)             |
| Hardware       | Free T4 GPU (Google Colab)                     |
| Training steps | 60                                             |
| Start reward   | 0.7675                                         |
| End reward     | **0.7917**                                     |
| Improvement    | **+0.0242**                                    |

---

## API Endpoints

| Method | Endpoint                             | Description               |
| ------ | ------------------------------------ | ------------------------- |
| GET    | `/health`                            | Health check              |
| GET    | `/tasks`                             | List all 5 tasks          |
| POST   | `/reset?task_name=tough_email_reply` | Start new episode         |
| POST   | `/step`                              | Submit action, get reward |
| GET    | `/state`                             | Current episode state     |

---

## Running Locally

```bash
git clone https://huggingface.co/spaces/Jyothisha426/personal-ai-assistant
cd personal-ai-assistant
pip install -r requirements.txt

# Gradio UI
python app.py

# OR raw API server
uvicorn main:app --host 0.0.0.0 --port 7860

# Run the inference agent (needs API server running)
export API_BASE_URL="https://api.openai.com/v1"
export API_KEY="your-api-key"
export MODEL_NAME="gpt-4o-mini"
python inference.py
```

---

## Environment Variables

| Variable       | Description                          |
| -------------- | ------------------------------------ |
| `GROQ_API_KEY` | Enables LLM-as-Judge hybrid grading  |
| `API_BASE_URL` | LLM API endpoint                     |
| `API_KEY`      | Your API key (injected by validator) |
| `MODEL_NAME`   | Model to use (default: gpt-4o-mini)  |
| `ENV_BASE_URL` | Environment server URL               |

---

## Notebook & Blog

- 🧪 **Training Notebook (Google Colab)**: https://colab.research.google.com/drive/1Eang1ybDBLhi8373pX3TOvm_c98Br4AK?usp=sharing
- 📝 **Blog Post**: See `blog.md` in this repository
