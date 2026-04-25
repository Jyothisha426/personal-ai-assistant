---
title: Personal AI Assistant
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Personal AI Assistant RL Environment

An **OpenEnv-compatible reinforcement learning environment** where an AI agent learns to handle real-life personal and professional tasks — from tough emails to scheduling conflicts to adapting when plans change.

**Theme**: World Modeling — Personalized Tasks (Theme 3.2)  
**Bonus theme**: Consumer Workflows with Schema Drift (Patronus AI)

---

## What This Environment Does

Real personal assistant tasks are messy, emotionally loaded, and context-dependent. This environment trains an agent across five progressively harder task types:

| Task                     | Scenarios | What the agent must do                                           |
| ------------------------ | --------- | ---------------------------------------------------------------- |
| `tough_email_reply`      | 15        | Write professional, empathetic replies to difficult emails       |
| `schedule_conflict`      | 15        | Resolve messy scheduling conflicts with clear priority decisions |
| `personal_message`       | 15        | Reply to personal/WhatsApp-style messages with appropriate tone  |
| `dinner_travel_planning` | 15        | Create concrete plans satisfying all stated constraints          |
| `schema_drift`           | 15        | Adapt mid-task when rules or context change (Patronus bonus)     |

**Total: 75 unique scenarios** across 5 tasks.

---

## Reward Design

Each task uses multi-dimensional reward scoring:

### Task 1 — Tough email reply

- 25% empathy (acknowledges sender's emotion)
- 25% relevance (addresses actual topic)
- 20% professionalism (appropriate tone)
- 20% actionability (concrete next step offered)
- 10% length (50–200 words ideal)

### Task 2 — Schedule conflict

- 35% coverage (mentions key scheduling elements)
- 30% must-include items (mentions all critical events)
- 25% decision clarity (clear prioritisation)
- 10% length

### Task 3 — Personal message

- 40% empathy keywords (appropriate emotional response)
- 30% avoids inappropriate phrases
- 15% non-refusal
- 15% length

### Task 4 — Dinner & travel planning

- 40% constraint satisfaction (addresses all stated constraints)
- 30% concreteness (specific names, times, costs)
- 15% actionability
- 15% length

### Task 5 — Schema drift

- 40% provides concrete alternative
- 30% acknowledges the change
- 20% actionable resolution
- 10% length

All rewards are continuous in range (0.0001, 0.9999).

---

## API Endpoints

| Method | Endpoint                           | Description               |
| ------ | ---------------------------------- | ------------------------- |
| GET    | /health                            | Health check              |
| GET    | /tasks                             | List all 5 tasks          |
| POST   | /reset?task_name=tough_email_reply | Start new episode         |
| POST   | /step                              | Submit action, get reward |
| GET    | /state                             | Current episode state     |

---

## Running Locally

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/personal-ai-assistant
cd personal-ai-assistant
pip install -r requirements.txt

# Terminal 1 — start environment server
uvicorn main:app --host 0.0.0.0 --port 7860

# Terminal 2 — run the agent
export API_BASE_URL="https://api.openai.com/v1"
export API_KEY="your-api-key"
export MODEL_NAME="gpt-4o-mini"
python inference.py
```

---

## Environment Variables

| Variable       | Description                          |
| -------------- | ------------------------------------ |
| `API_BASE_URL` | LLM API endpoint                     |
| `API_KEY`      | Your API key (injected by validator) |
| `MODEL_NAME`   | Model to use (default: gpt-4o-mini)  |
| `ENV_BASE_URL` | Environment server URL               |

---

## Example Session

```bash
# Start a tough email episode
curl -X POST "http://localhost:7860/reset?task_name=tough_email_reply"

# Submit a reply
curl -X POST "http://localhost:7860/step" \
  -H "Content-Type: application/json" \
  -d '{"response": "Dear customer, I sincerely apologize for the delay..."}'

# Check state
curl "http://localhost:7860/state"
```
