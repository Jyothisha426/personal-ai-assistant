# tasks.py
# ─────────────────────────────────────────────────────────────
# Personal AI Assistant — RL Environment
# Theme 3.2: World Modeling (Personalized Tasks)
#
# SCORES MUST BE STRICTLY BETWEEN 0 AND 1.
# Safe range: 0.0001 (worst) to 0.9999 (best)
# ─────────────────────────────────────────────────────────────

import re
from typing import Any

# ════════════════════════════════════════════════════════════════
# TASK 1 — TOUGH EMAIL REPLY
# Agent must write a professional, empathetic reply to a
# difficult email. Scored on empathy, relevance, professionalism,
# actionability, and length.
# ════════════════════════════════════════════════════════════════

TOUGH_EMAIL_SCENARIOS = [
    {
        "email_id": "email_001",
        "subject": "Extremely disappointed — where is my refund?",
        "body": "I have been waiting 3 weeks for my refund. Nobody responds to my calls. This is completely unacceptable and I will be posting reviews everywhere if this isn't resolved TODAY.",
        "sender": "angry.customer@gmail.com",
        "context": "You are a customer support manager",
        "keywords": ["refund", "apolog", "resolve", "contact", "process", "escalat", "timeline"],
        "empathy_phrases": ["understand", "frustrat", "apologi", "sorry", "concern", "disappoint"],
    },
    {
        "email_id": "email_002",
        "subject": "Why was the deadline missed?",
        "body": "The client presentation was supposed to be delivered yesterday. It wasn't. I need a full explanation by end of day. This reflects poorly on the entire team.",
        "sender": "boss@company.com",
        "context": "You are a team lead explaining a missed deadline",
        "keywords": ["deadline", "delay", "reason", "plan", "deliver", "prevent", "steps"],
        "empathy_phrases": ["acknowledge", "responsibility", "apologi", "understand", "ensure"],
    },
    {
        "email_id": "email_003",
        "subject": "Issues with your recent work",
        "body": "Honestly, the report you submitted had several errors and I had to redo half of it myself. I'm not sure what happened but this can't keep happening.",
        "sender": "colleague@company.com",
        "context": "You are responding to a colleague's passive-aggressive complaint",
        "keywords": ["errors", "improve", "feedback", "review", "quality", "process", "discuss"],
        "empathy_phrases": ["appreciate", "feedback", "understand", "concern", "improve", "sorry"],
    },
    {
        "email_id": "email_004",
        "subject": "We are reconsidering our contract",
        "body": "Given the recent performance issues and missed SLAs, our leadership is seriously considering ending our partnership. We need to have an urgent call this week.",
        "sender": "client.vp@bigcorp.com",
        "context": "You are an account manager responding to a client threat to cancel",
        "keywords": ["call", "meeting", "address", "commit", "improve", "partner", "resolv"],
        "empathy_phrases": ["understand", "concern", "value", "partner", "priorit", "commit"],
    },
    {
        "email_id": "email_005",
        "subject": "Your application was not successful",
        "body": "Thank you for interviewing with us. After careful consideration, we have decided to move forward with another candidate. We wish you all the best.",
        "sender": "hr@dreamcompany.com",
        "context": "You are responding gracefully to a job rejection",
        "keywords": ["thank", "feedback", "opportunit", "future", "learn", "appreciat"],
        "empathy_phrases": ["appreciat", "thank", "grateful", "understand", "consider"],
    },
    {
        "email_id": "email_006",
        "subject": "Price increase effective next month",
        "body": "Please be advised that due to rising operational costs, our service fees will increase by 25% starting next month. The new rates are attached.",
        "sender": "billing@vendor.com",
        "context": "You are responding to a surprise vendor price increase",
        "keywords": ["discuss", "review", "budget", "alternative", "negotiat", "concern", "call"],
        "empathy_phrases": ["understand", "concern", "appreciat", "discuss", "review"],
    },
    {
        "email_id": "email_007",
        "subject": "Your presentation needs significant work",
        "body": "I reviewed your deck from this morning. The structure was unclear, the data was not compelling, and I don't think it's ready for the board. Please rework it completely.",
        "sender": "manager@company.com",
        "context": "You are responding to harsh feedback from your manager",
        "keywords": ["thank", "feedback", "rework", "improve", "clarif", "timeline", "version"],
        "empathy_phrases": ["appreciate", "feedback", "understand", "take on board", "acknowledge"],
    },
    {
        "email_id": "email_008",
        "subject": "This is YOUR fault",
        "body": "The integration broke because your team pushed a change without notifying us. Our system has been down for 2 hours. You need to fix this immediately.",
        "sender": "partner.cto@othercorp.com",
        "context": "You are responding to a partner blaming your team for an outage",
        "keywords": ["investigat", "team", "fix", "coordinat", "resolve", "update", "urgent"],
        "empathy_phrases": ["understand", "priorit", "work together", "resolv", "apologi", "concern"],
    },
    {
        "email_id": "email_009",
        "subject": "I quit",
        "body": "I've decided to resign effective immediately. I've been unhappy for a while and I have a better opportunity. I won't be coming in tomorrow.",
        "sender": "key.engineer@company.com",
        "context": "You are a manager responding to a sudden resignation",
        "keywords": ["understand", "discuss", "transition", "handover", "wish", "appreciat"],
        "empathy_phrases": ["appreciate", "understand", "respect", "grateful", "support"],
    },
    {
        "email_id": "email_010",
        "subject": "URGENT: VIP customer is furious",
        "body": "Our biggest client just called me directly. They said your support team was rude and their issue still isn't fixed after 5 days. This needs executive attention NOW.",
        "sender": "ceo@company.com",
        "context": "You are responding to an escalation from the CEO about a VIP customer",
        "keywords": ["escalat", "priorit", "personal", "resolve", "update", "contact", "immedi"],
        "empathy_phrases": ["understand", "priorit", "personally", "escalat", "immedi", "ensure"],
    },
    {
        "email_id": "email_011",
        "subject": "Investor update request",
        "body": "It's been 6 weeks since our last update. We haven't seen the growth metrics we expected. Can you share where things stand and your revised projections?",
        "sender": "investor@vc.com",
        "context": "You are a founder responding to an investor when growth is slower than expected",
        "keywords": ["update", "metric", "plan", "growth", "strateg", "transparenc", "project"],
        "empathy_phrases": ["appreciate", "understand", "transparenc", "honest", "commit"],
    },
    {
        "email_id": "email_012",
        "subject": "Your team member filed a complaint",
        "body": "We have received a formal HR complaint from one of your direct reports regarding management style. We need to schedule a meeting with you and HR this week.",
        "sender": "hr.director@company.com",
        "context": "You are a manager responding to an HR complaint about you",
        "keywords": ["meeting", "cooperat", "understand", "address", "discuss", "open"],
        "empathy_phrases": ["cooperat", "understand", "open", "address", "support", "listen"],
    },
    {
        "email_id": "email_013",
        "subject": "You sent me the wrong order AGAIN",
        "body": "This is the third time I've received the wrong item. I've been a loyal customer for 5 years. I want a full refund AND a replacement sent express. This is ridiculous.",
        "sender": "loyal.customer@gmail.com",
        "context": "You are customer support responding to a repeat error for a loyal customer",
        "keywords": ["refund", "replac", "express", "apologi", "loyal", "priorit", "resolv"],
        "empathy_phrases": ["sincerely", "apologi", "loyal", "understand", "frustrat", "value"],
    },
    {
        "email_id": "email_014",
        "subject": "Legal notice — cease and desist",
        "body": "Our client believes your product infringes on their intellectual property. Please cease all use immediately and contact our legal team within 48 hours.",
        "sender": "lawyer@lawfirm.com",
        "context": "You are responding to a legal notice (professional, non-committal, routing to legal team)",
        "keywords": ["legal", "team", "review", "respond", "48 hours", "counsel", "contact"],
        "empathy_phrases": ["noted", "forward", "legal", "review", "respond appropriately"],
    },
    {
        "email_id": "email_015",
        "subject": "Why wasn't I promoted?",
        "body": "I just found out that someone with less experience than me got the promotion I was expecting. I deserve an explanation. I've given everything to this company.",
        "sender": "employee@company.com",
        "context": "You are an HR manager responding to an upset employee about a missed promotion",
        "keywords": ["meet", "discuss", "feedback", "recogniz", "value", "opportunit", "explain"],
        "empathy_phrases": ["understand", "value", "appreciat", "discuss", "feedback", "hear"],
    },
]


def grade_email_reply(response: str, scenario: dict) -> float:
    if not response or len(response.strip()) < 20:
        return 0.0001

    text = response.lower()
    words = text.split()
    word_count = len(words)

    # 1. Length score (10%) — 50-200 words ideal
    if 50 <= word_count <= 200:
        length_score = 1.0
    elif 30 <= word_count < 50 or 200 < word_count <= 280:
        length_score = 0.6
    elif word_count < 30:
        length_score = 0.2
    else:
        length_score = 0.4

    # 2. Empathy score (25%) — acknowledges sender's emotion
    empathy_hits = sum(1 for p in scenario["empathy_phrases"] if p in text)
    empathy_score = min(empathy_hits / 2, 1.0)

    # 3. Relevance score (25%) — addresses actual topic
    keyword_hits = sum(1 for k in scenario["keywords"] if k in text)
    relevance_score = min(keyword_hits / 3, 1.0)

    # 4. Professionalism score (20%)
    prof_indicators = ["dear", "sincerely", "regards", "thank you", "please", "would", "could", "appreciate"]
    unprofessional = ["wtf", "hell", "damn", "stupid", "idiot", "whatever", "not my problem"]
    prof_hits = sum(1 for p in prof_indicators if p in text)
    unpro_hits = sum(1 for u in unprofessional if u in text)
    professionalism_score = min(prof_hits / 3, 1.0) * (0.0 if unpro_hits > 0 else 1.0)

    # 5. Actionability score (20%) — offers a concrete next step
    action_indicators = ["will", "schedule", "contact", "call", "send", "arrange", "provide",
                         "ensure", "follow up", "reach out", "let me", "i'll", "we'll", "by"]
    action_hits = sum(1 for a in action_indicators if a in text)
    actionability_score = min(action_hits / 2, 1.0)

    # Weighted total
    total = (
        0.10 * length_score +
        0.25 * empathy_score +
        0.25 * relevance_score +
        0.20 * professionalism_score +
        0.20 * actionability_score
    )

    # Clamp to safe range
    return round(min(max(total, 0.0001), 0.9999), 4)


# ════════════════════════════════════════════════════════════════
# TASK 2 — SCHEDULE CONFLICT RESOLUTION
# Agent receives a messy scheduling situation and must produce
# a clear resolution: what to reschedule, who to notify, priority.
# ════════════════════════════════════════════════════════════════

SCHEDULE_CONFLICT_SCENARIOS = [
    {
        "conflict_id": "sched_001",
        "situation": "You have a client presentation at 3pm, a dentist appointment at 3pm, and your manager just asked for an urgent 1-on-1 also at 3pm today.",
        "constraints": ["client is highest priority", "dentist can be rescheduled", "manager needs 15 min"],
        "resolution_keywords": ["client", "reschedule", "dentist", "manager", "priorit", "notif"],
        "must_include": ["client"],
    },
    {
        "conflict_id": "sched_002",
        "situation": "Team standup is at 9am. A candidate interview is at 9:30am. Your flight to Bangalore for a conference is at 11am and the airport is 90 min away.",
        "constraints": ["standup is 15 min", "interview is 45 min", "need 2 hrs before flight"],
        "resolution_keywords": ["standup", "interview", "flight", "airport", "time", "leave", "rescheduled"],
        "must_include": ["flight", "airport"],
    },
    {
        "conflict_id": "sched_003",
        "situation": "You promised your child's school event at 4pm. A critical production bug was just reported at 3pm and your team needs you. Your spouse is out of town.",
        "constraints": ["production bug is urgent", "school event is 4pm", "no backup childcare"],
        "resolution_keywords": ["bug", "school", "team", "delegat", "priorit", "handle", "join"],
        "must_include": ["bug", "school"],
    },
    {
        "conflict_id": "sched_004",
        "situation": "Three back-to-back meetings from 2-5pm, a project deadline at 5pm, and a mandatory HR training that was just scheduled for 4pm.",
        "constraints": ["deadline cannot move", "HR training is mandatory", "meetings have external attendees"],
        "resolution_keywords": ["deadline", "HR", "training", "meeting", "rescheduled", "delegat", "prior"],
        "must_include": ["deadline", "HR"],
    },
    {
        "conflict_id": "sched_005",
        "situation": "Weekly team lunch is at 1pm. A sales prospect asked to meet at 1pm today only — they're flying out tonight. Your doctor called about urgent test results at 1pm.",
        "constraints": ["prospect meeting is a big deal", "doctor call is health-related", "team lunch is social"],
        "resolution_keywords": ["prospect", "doctor", "lunch", "reschedule", "health", "sales", "priorit"],
        "must_include": ["prospect", "doctor"],
    },
    {
        "conflict_id": "sched_006",
        "situation": "You're leading a workshop from 10am-12pm. Your manager called a surprise all-hands at 11am. A server outage alert just came in at 10:45am.",
        "constraints": ["workshop has 20 external attendees", "all-hands is company-wide mandatory", "outage needs immediate attention"],
        "resolution_keywords": ["workshop", "all-hands", "outage", "delegat", "join", "team", "handle"],
        "must_include": ["outage", "workshop"],
    },
    {
        "conflict_id": "sched_007",
        "situation": "Performance reviews are due today at 6pm. You have 8 reviews to complete. Two interviews are scheduled at 3pm and 5pm.",
        "constraints": ["reviews cannot be late", "candidates flew in for interviews", "each review takes 30 min"],
        "resolution_keywords": ["review", "interview", "deadline", "time", "complet", "priorit", "schedul"],
        "must_include": ["review", "interview"],
    },
    {
        "conflict_id": "sched_008",
        "situation": "Board meeting at 2pm needs your slides which aren't ready. A customer is having a critical issue that only you can solve. It's 1pm.",
        "constraints": ["board meeting cannot be delayed", "customer is VIP", "1 hour total available"],
        "resolution_keywords": ["board", "slides", "customer", "priorit", "delegat", "time", "resolv"],
        "must_include": ["board", "customer"],
    },
    {
        "conflict_id": "sched_009",
        "situation": "You're on vacation tomorrow. Today you have 6 hrs of meetings, a report due EOD, and your replacement just called in sick.",
        "constraints": ["vacation is booked and paid", "report is for CEO", "replacement is unavailable"],
        "resolution_keywords": ["vacation", "report", "delegat", "replacement", "CEO", "plan", "cover"],
        "must_include": ["report", "vacation"],
    },
    {
        "conflict_id": "sched_010",
        "situation": "Two team members scheduled their farewell lunches on the same day at the same time. Both report to you. You can only attend one.",
        "constraints": ["both employees are leaving", "both lunches are at 12:30pm", "skipping one will cause hurt feelings"],
        "resolution_keywords": ["attend", "split", "both", "celebrat", "lunch", "time", "fair"],
        "must_include": ["both"],
    },
    {
        "conflict_id": "sched_011",
        "situation": "A critical feature demo for a VC is at 3pm. The lead engineer just texted that the demo environment is broken. It's 2pm.",
        "constraints": ["VC meeting cannot be rescheduled", "engineer needs 2 hrs to fix", "1 hour available"],
        "resolution_keywords": ["demo", "fix", "VC", "backup", "plan", "delay", "alternative", "engineer"],
        "must_include": ["demo", "VC"],
    },
    {
        "conflict_id": "sched_012",
        "situation": "You scheduled deep focus work from 9-11am. 4 people sent urgent Slack messages. Your manager wants a sync at 9:30am.",
        "constraints": ["deep work was blocked in calendar", "messages vary in urgency", "manager outranks your calendar block"],
        "resolution_keywords": ["manager", "sync", "message", "priorit", "respond", "focus", "block"],
        "must_include": ["manager"],
    },
    {
        "conflict_id": "sched_013",
        "situation": "Two timezone-conflicting calls: New York client at 8am IST (very early) and Singapore partner at 10pm IST (very late). Both are today.",
        "constraints": ["both clients are important", "you cannot do both without exhaustion", "one can potentially be delegated"],
        "resolution_keywords": ["timezone", "delegat", "attend", "priorit", "both", "time", "NY", "Singapore"],
        "must_include": ["both"],
    },
    {
        "conflict_id": "sched_014",
        "situation": "Your child's birthday is tonight. A product launch crisis erupted at 5pm. The CEO expects you to lead the incident response.",
        "constraints": ["birthday cannot be moved", "CEO expects you specifically", "incident is high severity"],
        "resolution_keywords": ["birthday", "incident", "CEO", "delegat", "balance", "time", "lead", "family"],
        "must_include": ["birthday", "incident"],
    },
    {
        "conflict_id": "sched_015",
        "situation": "Sprint planning is Monday 10am. A client emergency just came up Monday 10am. The team cannot start the sprint without you.",
        "constraints": ["sprint planning involves 8 engineers", "client emergency is revenue at risk", "planning takes 2 hrs"],
        "resolution_keywords": ["sprint", "client", "team", "rescheduled", "delegat", "priorit", "morning"],
        "must_include": ["client", "sprint"],
    },
]


def grade_schedule_resolution(response: str, scenario: dict) -> float:
    if not response or len(response.strip()) < 20:
        return 0.0001

    text = response.lower()
    words = text.split()

    # 1. Length (10%) — a resolution needs substance
    if 40 <= len(words) <= 250:
        length_score = 1.0
    elif 20 <= len(words) < 40:
        length_score = 0.5
    else:
        length_score = 0.3

    # 2. Covers key elements (35%)
    keyword_hits = sum(1 for k in scenario["resolution_keywords"] if k in text)
    coverage_score = min(keyword_hits / 4, 1.0)

    # 3. Mentions all must-include items (30%)
    must_hits = sum(1 for m in scenario["must_include"] if m in text)
    must_score = must_hits / len(scenario["must_include"])

    # 4. Has a clear decision / action (25%)
    decision_words = ["will", "should", "reschedule", "priorit", "delegat", "attend", "skip",
                      "cancel", "move", "handle", "plan", "first", "then", "next"]
    decision_hits = sum(1 for d in decision_words if d in text)
    decision_score = min(decision_hits / 3, 1.0)

    total = (
        0.10 * length_score +
        0.35 * coverage_score +
        0.30 * must_score +
        0.25 * decision_score
    )
    return round(min(max(total, 0.0001), 0.9999), 4)


# ════════════════════════════════════════════════════════════════
# TASK 3 — PERSONAL MESSAGE HANDLING
# Agent must reply to personal/WhatsApp-style messages with
# appropriate empathy and tone. Not business-formal.
# ════════════════════════════════════════════════════════════════

PERSONAL_MESSAGE_SCENARIOS = [
    {
        "msg_id": "msg_001",
        "from": "Best friend",
        "message": "Hey I'm so sorry but I can't make it to your birthday dinner tonight. Something came up at work. I feel terrible.",
        "tone_expected": "warm, understanding, not guilt-tripping",
        "keywords": ["okay", "understand", "another time", "no worries", "reschedule", "miss", "celebrate"],
        "avoid": ["angry", "hurt", "selfish", "typical", "always"],
    },
    {
        "msg_id": "msg_002",
        "from": "Parent",
        "message": "Beta/dear, why don't you call anymore? We haven't heard from you in weeks. We worry about you.",
        "tone_expected": "warm, apologetic, reassuring",
        "keywords": ["sorry", "busy", "love", "call", "soon", "miss", "think"],
        "avoid": ["leave me alone", "busy", "stop", "nagging"],
    },
    {
        "msg_id": "msg_003",
        "from": "Close friend",
        "message": "I got rejected from my dream company today. I don't know what to do. I feel like a failure.",
        "tone_expected": "empathetic, encouraging, not dismissive",
        "keywords": ["sorry", "hear", "hard", "proud", "next", "believe", "support", "there for you"],
        "avoid": ["get over it", "not a big deal", "others have it worse", "just apply"],
    },
    {
        "msg_id": "msg_004",
        "from": "Roommate",
        "message": "Hey can you please clean up after yourself in the kitchen? It's been like this for 3 days and it's bothering me.",
        "tone_expected": "apologetic, non-defensive, agreeable",
        "keywords": ["sorry", "clean", "today", "forgot", "will do", "fair", "noted"],
        "avoid": ["it was your turn", "overreacting", "relax", "always"],
    },
    {
        "msg_id": "msg_005",
        "from": "Friend",
        "message": "I need to borrow ₹5000 until next month. I'm in a tight spot. Please help.",
        "tone_expected": "caring but honest — can say yes, no, or offer alternative",
        "keywords": ["help", "understand", "situation", "right now", "able", "alternatively", "month"],
        "avoid": ["never", "poor planning", "always asking", "irresponsible"],
    },
    {
        "msg_id": "msg_006",
        "from": "Sibling",
        "message": "I don't think you've been treating Mum well lately. She told me she feels ignored by you. Can we talk?",
        "tone_expected": "non-defensive, open to conversation, caring",
        "keywords": ["thank", "hear", "talk", "aware", "work on", "family", "important"],
        "avoid": ["wrong", "fault", "none of your business", "mum is exaggerating"],
    },
    {
        "msg_id": "msg_007",
        "from": "Old friend reconnecting",
        "message": "Hey! It's been 5 years! I'm in your city this weekend. Would love to catch up if you're free!",
        "tone_expected": "warm, enthusiastic or honest about availability",
        "keywords": ["great", "hear", "would love", "free", "meet", "let me know", "plan"],
        "avoid": ["busy", "can't", "not interested"],
    },
    {
        "msg_id": "msg_008",
        "from": "Colleague / friend",
        "message": "I saw you got credit for the project in the all-hands but I did most of the work. That really stings.",
        "tone_expected": "validating, apologetic, willing to correct the record",
        "keywords": ["sorry", "fair", "acknowledge", "credit", "mention", "contribution", "right"],
        "avoid": ["my idea", "i did more", "ungrateful", "your fault"],
    },
    {
        "msg_id": "msg_009",
        "from": "Partner",
        "message": "You've been really distant lately. I feel like I'm not a priority. Can we please talk tonight?",
        "tone_expected": "warm, honest, non-dismissive, willing to talk",
        "keywords": ["sorry", "hear", "talk", "tonight", "important", "there", "priority"],
        "avoid": ["busy", "overreacting", "stop", "dramatic"],
    },
    {
        "msg_id": "msg_010",
        "from": "Friend",
        "message": "My mum just passed away this morning. I don't know how to feel right now.",
        "tone_expected": "deeply empathetic, gentle, no advice-giving",
        "keywords": ["so sorry", "here for you", "loss", "love", "anything", "there", "heart"],
        "avoid": ["at least", "better place", "move on", "time heals", "stay strong"],
    },
    {
        "msg_id": "msg_011",
        "from": "Friend",
        "message": "I think I messed up badly at work today. I may have accidentally sent confidential data to the wrong person. I'm panicking.",
        "tone_expected": "calm, practical, supportive",
        "keywords": ["breathe", "okay", "report", "IT", "manager", "help", "step", "soon"],
        "avoid": ["careless", "fired", "always", "told you"],
    },
    {
        "msg_id": "msg_012",
        "from": "Acquaintance",
        "message": "Hey! Could you write me a LinkedIn recommendation? I'm applying for jobs.",
        "tone_expected": "honest — can agree or politely decline if don't know them well",
        "keywords": ["happy to", "know you well", "honest", "recomm", "let me know", "would"],
        "avoid": ["no", "busy", "barely know you", "annoying"],
    },
    {
        "msg_id": "msg_013",
        "from": "Friend",
        "message": "I just found out my partner has been lying to me for months. I'm devastated. I don't know what to do.",
        "tone_expected": "empathetic, supportive, non-judgmental",
        "keywords": ["so sorry", "here", "devastat", "listen", "support", "whatever", "need"],
        "avoid": ["told you so", "leave them", "your fault", "move on"],
    },
    {
        "msg_id": "msg_014",
        "from": "Neighbor",
        "message": "Your music was really loud last night. My kids couldn't sleep. Please be more considerate.",
        "tone_expected": "apologetic, non-defensive, promising to fix it",
        "keywords": ["sorry", "apologize", "didn't realize", "considerate", "keep down", "future"],
        "avoid": ["wasn't that loud", "your problem", "kids should sleep early"],
    },
    {
        "msg_id": "msg_015",
        "from": "Friend",
        "message": "I'm thinking of quitting my stable job to start a business. Everyone thinks I'm crazy. What do you think?",
        "tone_expected": "supportive, honest, encouraging without dismissing risk",
        "keywords": ["exciting", "plan", "think through", "support", "consider", "risk", "believe"],
        "avoid": ["crazy", "don't do it", "secure job", "failure"],
    },
]


def grade_personal_message(response: str, scenario: dict) -> float:
    if not response or len(response.strip()) < 10:
        return 0.0001

    text = response.lower()
    words = text.split()

    # 1. Length (15%) — personal messages should be conversational, not essays
    if 15 <= len(words) <= 120:
        length_score = 1.0
    elif 8 <= len(words) < 15 or 120 < len(words) <= 180:
        length_score = 0.6
    else:
        length_score = 0.3

    # 2. Empathy / tone keywords (40%)
    keyword_hits = sum(1 for k in scenario["keywords"] if k in text)
    keyword_score = min(keyword_hits / 3, 1.0)

    # 3. Avoids inappropriate phrases (30%)
    avoid_hits = sum(1 for a in scenario["avoid"] if a in text)
    avoid_score = 1.0 if avoid_hits == 0 else max(0.0, 1.0 - avoid_hits * 0.4)

    # 4. Not a refusal (15%)
    refusal_phrases = ["cannot help", "i can't respond", "as an ai", "i am not able"]
    refusal_hits = sum(1 for r in refusal_phrases if r in text)
    refusal_score = 0.0 if refusal_hits > 0 else 1.0

    total = (
        0.15 * length_score +
        0.40 * keyword_score +
        0.30 * avoid_score +
        0.15 * refusal_score
    )
    return round(min(max(total, 0.0001), 0.9999), 4)


# ════════════════════════════════════════════════════════════════
# TASK 4 — DINNER & TRAVEL PLANNING
# Agent receives a planning request with constraints and must
# produce a concrete, actionable plan satisfying all constraints.
# ════════════════════════════════════════════════════════════════

PLANNING_SCENARIOS = [
    {
        "plan_id": "plan_001",
        "request": "Plan a dinner tonight for 4 people. One is vegetarian, one has a nut allergy. Budget is ₹2000. We're near Koramangala, Bangalore.",
        "constraints": ["vegetarian option", "no nuts", "under ₹2000", "Koramangala area"],
        "must_address": ["vegetarian", "nut", "budget"],
    },
    {
        "plan_id": "plan_002",
        "request": "I need to travel from Hyderabad to Bangalore tomorrow morning, arriving before 10am for a 10:30am meeting. What's the best option?",
        "constraints": ["arrive before 10am", "Hyderabad to Bangalore", "tomorrow morning"],
        "must_address": ["flight", "time", "arrive"],
    },
    {
        "plan_id": "plan_003",
        "request": "Plan a surprise birthday dinner for my wife this Saturday. She loves Italian food. We have two kids (ages 5 and 8). Budget ₹5000.",
        "constraints": ["Saturday", "Italian food", "kid-friendly", "budget ₹5000", "surprise"],
        "must_address": ["Italian", "kids", "budget"],
    },
    {
        "plan_id": "plan_004",
        "request": "I have a 6-hour layover in Mumbai. I'm arriving at Terminal 2 at 10am. Suggest what I can do.",
        "constraints": ["6 hours", "Terminal 2 Mumbai", "must return for flight", "practical"],
        "must_address": ["time", "return", "terminal"],
    },
    {
        "plan_id": "plan_005",
        "request": "Plan a team outing for 12 people in Chennai. Mix of activities and food. Budget ₹1500 per person. Must be done by 6pm.",
        "constraints": ["12 people", "Chennai", "₹1500 per person", "done by 6pm", "activities + food"],
        "must_address": ["budget", "time", "team", "activity"],
    },
    {
        "plan_id": "plan_006",
        "request": "I want to take my parents (both 65+) for a 3-day trip from Delhi. They can't walk too much. Budget ₹30,000 total.",
        "constraints": ["elderly parents", "limited walking", "3 days", "Delhi departure", "₹30000"],
        "must_address": ["walk", "3 days", "budget"],
    },
    {
        "plan_id": "plan_007",
        "request": "Plan a date night in Mumbai tonight. My partner loves seafood. Budget ₹3000. We're in Bandra.",
        "constraints": ["tonight", "seafood", "Bandra Mumbai", "₹3000", "romantic"],
        "must_address": ["seafood", "Bandra", "budget"],
    },
    {
        "plan_id": "plan_008",
        "request": "I need to attend 3 meetings across different parts of Bangalore tomorrow: Whitefield at 9am, Indiranagar at 12pm, Electronic City at 4pm.",
        "constraints": ["3 locations", "Whitefield 9am", "Indiranagar 12pm", "Electronic City 4pm", "same day"],
        "must_address": ["Whitefield", "Indiranagar", "Electronic City", "travel time"],
    },
    {
        "plan_id": "plan_009",
        "request": "Plan a healthy week of dinners for a family of 3. One person is diabetic. No processed food. Budget ₹3000 for the week.",
        "constraints": ["diabetic-friendly", "no processed food", "7 dinners", "₹3000 total", "family of 3"],
        "must_address": ["diabetic", "week", "budget"],
    },
    {
        "plan_id": "plan_010",
        "request": "I want to propose to my girlfriend this weekend in Bangalore. Make it special but not over the top. Budget ₹10,000.",
        "constraints": ["Bangalore", "this weekend", "romantic but not excessive", "₹10,000"],
        "must_address": ["proposal", "location", "budget"],
    },
    {
        "plan_id": "plan_011",
        "request": "We have guests from the US visiting Bangalore for 2 days. They've never been to India. Plan an itinerary showing the best of the city.",
        "constraints": ["2 days", "Bangalore", "first-time India visitors", "cultural + food experience"],
        "must_address": ["day 1", "day 2", "food", "experience"],
    },
    {
        "plan_id": "plan_012",
        "request": "Plan a work-from-cafe day in Chennai. I need good WiFi, quiet, open 9am-6pm, with good coffee. Suggest 2 options.",
        "constraints": ["good WiFi", "quiet", "Chennai", "9am-6pm", "coffee", "2 options"],
        "must_address": ["WiFi", "quiet", "Chennai"],
    },
    {
        "plan_id": "plan_013",
        "request": "My flight is at 6am from Bangalore airport. I live in HSR Layout. Plan my morning to reach on time.",
        "constraints": ["6am flight", "HSR Layout", "need to reach 2 hrs early", "4am wake time area"],
        "must_address": ["time", "airport", "HSR", "wake"],
    },
    {
        "plan_id": "plan_014",
        "request": "Plan a Diwali party at home for 20 people. Mix of vegetarian and non-vegetarian food. Budget ₹8000 including decorations.",
        "constraints": ["20 people", "Diwali theme", "veg + non-veg", "₹8000 total", "home party"],
        "must_address": ["veg", "budget", "Diwali", "decoration"],
    },
    {
        "plan_id": "plan_015",
        "request": "I want to run my first 5K race next month. I've never run before. Create a 4-week training plan.",
        "constraints": ["4 weeks", "beginner", "5K goal", "no gym required"],
        "must_address": ["week", "run", "plan", "5K"],
    },
]


def grade_planning_response(response: str, scenario: dict) -> float:
    if not response or len(response.strip()) < 20:
        return 0.0001

    text = response.lower()
    words = text.split()

    # 1. Length (15%) — plans need detail
    if 60 <= len(words) <= 350:
        length_score = 1.0
    elif 30 <= len(words) < 60:
        length_score = 0.5
    else:
        length_score = 0.3

    # 2. Addresses all constraints (40%)
    must_hits = sum(1 for m in scenario["must_address"] if m.lower() in text)
    constraint_score = must_hits / len(scenario["must_address"])

    # 3. Concreteness — specific details, not vague advice (30%)
    concrete_indicators = ["recommend", "suggest", "option", "restaurant", "place",
                           "time", "cost", "budget", "step", "day", "hour", "km",
                           "₹", "minute", "first", "then", "next", "finally"]
    concrete_hits = sum(1 for c in concrete_indicators if c in text)
    concrete_score = min(concrete_hits / 4, 1.0)

    # 4. Actionability (15%) — clear next steps
    action_words = ["go to", "book", "call", "order", "take", "start", "arrive",
                    "leave", "plan", "prepare", "visit", "check"]
    action_hits = sum(1 for a in action_words if a in text)
    action_score = min(action_hits / 2, 1.0)

    total = (
        0.15 * length_score +
        0.40 * constraint_score +
        0.30 * concrete_score +
        0.15 * action_score
    )
    return round(min(max(total, 0.0001), 0.9999), 4)


# ════════════════════════════════════════════════════════════════
# TASK 5 — SCHEMA DRIFT (Patronus AI bonus theme)
# Agent handles a multi-step workflow where rules/context
# change mid-task. Tests adaptability.
# ════════════════════════════════════════════════════════════════

SCHEMA_DRIFT_SCENARIOS = [
    {
        "drift_id": "drift_001",
        "initial_task": "Book a dinner reservation for 2 at an Italian restaurant in Koramangala for tonight at 8pm.",
        "drift_event": "The restaurant you were about to book just marked itself as closed tonight due to a private event.",
        "new_constraint": "Must find an alternative Italian restaurant in the same area, same time.",
        "keywords": ["alternative", "another", "instead", "closed", "option", "still", "Italian"],
        "adaption_keywords": ["closed", "alternative", "instead", "new", "option"],
    },
    {
        "drift_id": "drift_002",
        "initial_task": "Schedule a Zoom meeting with the client for tomorrow at 3pm and send a calendar invite.",
        "drift_event": "The client just emailed saying they don't use Zoom and want to use Google Meet instead.",
        "new_constraint": "Must reschedule on Google Meet, resend the invite.",
        "keywords": ["Google Meet", "Meet", "resend", "invite", "update", "link", "client"],
        "adaption_keywords": ["Google Meet", "update", "resend", "instead"],
    },
    {
        "drift_id": "drift_003",
        "initial_task": "Order a birthday cake from the bakery for pickup tomorrow at 10am.",
        "drift_event": "The bakery just called — they're out of your requested flavour (chocolate). They offer vanilla or red velvet.",
        "new_constraint": "Must choose an alternative flavour and confirm the new order.",
        "keywords": ["vanilla", "red velvet", "alternative", "confirm", "instead", "flavour", "order"],
        "adaption_keywords": ["instead", "alternative", "confirm", "flavour"],
    },
    {
        "drift_id": "drift_004",
        "initial_task": "Book a cab to the airport for 5am tomorrow. The trip takes 45 min.",
        "drift_event": "Your cab app shows a surge — 3x pricing. Total fare would be ₹900 vs normal ₹300.",
        "new_constraint": "Must find a more economical option or justify the surge cost.",
        "keywords": ["surge", "alternative", "auto", "metro", "cost", "₹", "time", "option"],
        "adaption_keywords": ["surge", "alternative", "instead", "cost"],
    },
    {
        "drift_id": "drift_005",
        "initial_task": "Send a gift hamper to a colleague who just had a baby. Budget ₹2000, delivery tomorrow.",
        "drift_event": "The delivery service says tomorrow delivery isn't available in that area. Earliest is 3 days later.",
        "new_constraint": "Must find an alternative delivery option or adjust the plan.",
        "keywords": ["delivery", "alternative", "local", "pickup", "3 days", "instead", "option", "plan"],
        "adaption_keywords": ["delivery", "alternative", "instead", "option"],
    },
    {
        "drift_id": "drift_006",
        "initial_task": "Plan a movie night at home. You were going to stream a specific film on Netflix.",
        "drift_event": "The movie was just removed from Netflix in your region.",
        "new_constraint": "Must find another way to watch it or suggest an alternative film.",
        "keywords": ["rent", "Amazon", "YouTube", "buy", "alternative", "instead", "film", "watch"],
        "adaption_keywords": ["alternative", "instead", "rent", "buy", "another"],
    },
    {
        "drift_id": "drift_007",
        "initial_task": "You're coordinating a team offsite and booked a venue for 15 people.",
        "drift_event": "Three more people confirmed attendance — now it's 18. The venue max capacity is 15.",
        "new_constraint": "Must upgrade venue or find overflow solution.",
        "keywords": ["larger", "upgrade", "capacity", "alternative", "venue", "18", "accommodate"],
        "adaption_keywords": ["upgrade", "larger", "alternative", "accommodate", "capacity"],
    },
    {
        "drift_id": "drift_008",
        "initial_task": "Arrange a video call with the US team at 9am IST tomorrow.",
        "drift_event": "Daylight saving time just changed in the US — 9am IST is now 11:30pm their time, not a reasonable hour.",
        "new_constraint": "Must find a mutually workable time accounting for the DST change.",
        "keywords": ["DST", "timezone", "adjust", "new time", "reschedule", "workable", "IST"],
        "adaption_keywords": ["adjust", "reschedule", "timezone", "DST", "new time"],
    },
    {
        "drift_id": "drift_009",
        "initial_task": "You planned to pay the vendor by bank transfer today.",
        "drift_event": "The bank's online portal is down for maintenance until tomorrow.",
        "new_constraint": "Must find an alternative payment method for today.",
        "keywords": ["UPI", "NEFT", "alternative", "payment", "today", "instead", "method", "portal"],
        "adaption_keywords": ["alternative", "instead", "UPI", "payment", "method"],
    },
    {
        "drift_id": "drift_010",
        "initial_task": "Book a conference room for a 10-person meeting at 2pm tomorrow.",
        "drift_event": "All conference rooms are now booked for tomorrow afternoon due to a company all-hands.",
        "new_constraint": "Must find an alternative space or format for the meeting.",
        "keywords": ["alternative", "remote", "Zoom", "cafe", "offsite", "instead", "option", "space"],
        "adaption_keywords": ["alternative", "instead", "remote", "option", "space"],
    },
    {
        "drift_id": "drift_011",
        "initial_task": "Order office supplies — specifically blue pens, A4 paper, and sticky notes — for delivery this week.",
        "drift_event": "A4 paper is out of stock on the platform with 2-week delivery. Rest are available.",
        "new_constraint": "Must handle the A4 paper shortage — find alternative or partial order.",
        "keywords": ["A4", "alternative", "elsewhere", "stock", "partial", "pens", "notes", "order"],
        "adaption_keywords": ["A4", "alternative", "stock", "elsewhere", "partial"],
    },
    {
        "drift_id": "drift_012",
        "initial_task": "You're managing RSVPs for a company event and need a headcount by 5pm today.",
        "drift_event": "Your RSVP form tool just went down and isn't recoverable today.",
        "new_constraint": "Must collect RSVPs through an alternative method before 5pm.",
        "keywords": ["email", "WhatsApp", "Google Form", "alternative", "5pm", "headcount", "collect"],
        "adaption_keywords": ["alternative", "email", "WhatsApp", "instead", "collect"],
    },
    {
        "drift_id": "drift_013",
        "initial_task": "Plan a team lunch at a Thai restaurant near the office.",
        "drift_event": "Two team members just mentioned they have severe shellfish allergies, and Thai food often contains shellfish.",
        "new_constraint": "Must choose a different cuisine or find a Thai place that can accommodate the allergy.",
        "keywords": ["allergy", "shellfish", "alternative", "cuisine", "accommodate", "instead", "safe"],
        "adaption_keywords": ["allergy", "alternative", "accommodate", "instead", "safe"],
    },
    {
        "drift_id": "drift_014",
        "initial_task": "You're using a specific API to fetch weather data for your app.",
        "drift_event": "The API provider just changed their pricing — free tier now limits to 100 calls/day, but you need 500.",
        "new_constraint": "Must find an alternative free API or justify the paid plan.",
        "keywords": ["alternative", "API", "free", "OpenWeather", "paid", "limit", "calls", "instead"],
        "adaption_keywords": ["alternative", "API", "instead", "free", "limit"],
    },
    {
        "drift_id": "drift_015",
        "initial_task": "Schedule a guest speaker for your team's Friday session via the company's standard invite system.",
        "drift_event": "The guest speaker doesn't have a corporate email and can't receive invites from the system.",
        "new_constraint": "Must find an alternative way to get them the invite/link.",
        "keywords": ["personal email", "WhatsApp", "alternative", "link", "send", "instead", "directly"],
        "adaption_keywords": ["alternative", "instead", "personal", "send", "directly"],
    },
]


def grade_schema_drift(response: str, scenario: dict) -> float:
    if not response or len(response.strip()) < 20:
        return 0.0001

    text = response.lower()
    words = text.split()

    # 1. Length (10%)
    if 40 <= len(words) <= 300:
        length_score = 1.0
    elif 20 <= len(words) < 40:
        length_score = 0.5
    else:
        length_score = 0.3

    # 2. Acknowledges the drift/change (30%)
    drift_hits = sum(1 for a in scenario["adaption_keywords"] if a in text)
    drift_score = min(drift_hits / 2, 1.0)

    # 3. Provides a concrete alternative (40%)
    keyword_hits = sum(1 for k in scenario["keywords"] if k in text)
    alternative_score = min(keyword_hits / 3, 1.0)

    # 4. Actionable resolution (20%)
    action_words = ["will", "can", "should", "recommend", "suggest", "try", "use",
                    "book", "send", "find", "contact", "choose", "switch", "instead"]
    action_hits = sum(1 for a in action_words if a in text)
    action_score = min(action_hits / 2, 1.0)

    total = (
        0.10 * length_score +
        0.30 * drift_score +
        0.40 * alternative_score +
        0.20 * action_score
    )
    return round(min(max(total, 0.0001), 0.9999), 4)


# ════════════════════════════════════════════════════════════════
# TASK REGISTRY — used by main.py
# ════════════════════════════════════════════════════════════════

TASK_REGISTRY = {
    "tough_email_reply": {
        "description": "Write a professional, empathetic reply to a difficult email. Be specific, address the sender's concern, and offer a concrete next step.",
        "scenarios": TOUGH_EMAIL_SCENARIOS,
        "grader": grade_email_reply,
        "observation_keys": ["email_id", "subject", "body", "sender", "context"],
    },
    "schedule_conflict": {
        "description": "Resolve the scheduling conflict. State clearly what you would prioritize, what you would reschedule, and who you would notify.",
        "scenarios": SCHEDULE_CONFLICT_SCENARIOS,
        "grader": grade_schedule_resolution,
        "observation_keys": ["conflict_id", "situation", "constraints"],
    },
    "personal_message": {
        "description": "Reply to this personal message with an appropriate, human-feeling response. Match the tone to the situation.",
        "scenarios": PERSONAL_MESSAGE_SCENARIOS,
        "grader": grade_personal_message,
        "observation_keys": ["msg_id", "from", "message", "tone_expected"],
    },
    "dinner_travel_planning": {
        "description": "Create a concrete, actionable plan that satisfies all the stated constraints. Be specific about locations, times, and costs.",
        "scenarios": PLANNING_SCENARIOS,
        "grader": grade_planning_response,
        "observation_keys": ["plan_id", "request", "constraints"],
    },
    "schema_drift": {
        "description": "Something has changed mid-task. Acknowledge the change and provide a concrete alternative plan or solution.",
        "scenarios": SCHEMA_DRIFT_SCENARIOS,
        "grader": grade_schema_drift,
        "observation_keys": ["drift_id", "initial_task", "drift_event", "new_constraint"],
    },
}
