from crewai import Task
import json
from agents import classifier_agent, task_agent, query_agent, final_answer_agent


# ## create tasks


# Task 1 : classifier task ( General , specific)

# step 1 : Inject Dynamic calssification input
"""
# 6. Execute with test questions
test_cases = [
    {"question": "How does the Matching Score work?", "user_type": "candidate"}, # General
    {"question": "Show my recent applications", "user_type": "candidate"}, # Specific
    {"question": "Can HR managers post jobs?", "user_type": "hr"} # General
]
"""
question = "What job roles am I best suited for based on my profile?"
user_type = "candidate"
user_data = {}

def build_classifier_task(question: str, user_type: str) -> Task:
    formatted_description = (
        f"Classify this question from a {user_type} user:\n"
        f"---\n{question}\n---\n\n"
        "Classification rules:\n"
        "1. If the question is **about the Qaddemly system**, its features, how it works, or platform-wide functionality → classify as GENERAL.\n"
        "2. If the question **depends on the user's own data** (profile, jobs, resume, applications, history, or dynamic recommendations) → classify as SPECIFIC.\n\n"
        "**GENERAL Examples**:\n"
        "- How do I register?\n"
        "- What is the Matching Score?\n"
        "- What filters can I use to search jobs?\n"
        "- What is Role-Based Access Control?\n\n"
        "**SPECIFIC Examples**:\n"
        "- What jobs are best for me?\n"
        "- Have I applied to Google before?\n"
        "- Can you improve my resume?\n"
        "- What companies follow my profile?\n"
        "- What’s the score for this job match for me?\n\n"
        "** Hint:** If a question includes terms like **'my', 'me', 'based on profile/resume'**, it likely requires SPECIFIC classification.\n\n"
        "Respond ONLY with: GENERAL or SPECIFIC"
    )

    return Task(
        description=formatted_description,
        agent=classifier_agent,
        expected_output="GENERAL or SPECIFIC",
    )


# Task 2 : task_classifier_task function


def build_task_classifier_task(question: str, user_type: str) -> Task:
    return Task(
        description=f"""
You are receiving a question from a user of type: **{user_type}**.

Question:
---
{question}
---

Your job is to classify this question into the correct Qaddemly system **feature** based on the user's intent.

Return ONLY ONE of the following labels:

- RECOMMENDATION
- MATCHING_SCORE
- COVER_LETTER
- RESUME_BUILDER
- JOB_SEARCH
- COMPANY_SEARCH
- APPLICATION_TRACKING
- PROFILE
- NOTIFICATIONS
- MESSAGING
- OTHER

💡 REMEMBER:
- Only use RECOMMENDATION if the user is explicitly asking to use the system’s automated job-matching feature.
- If the user is asking for career **advice** or suggestions from the AI/chatbot (e.g., “What jobs suit me?”), return OTHER.
- Be strict and accurate. Output one UPPERCASE value only — no punctuation.

Respond ONLY with the label from the list above.
""",
        agent=task_agent,
        expected_output="One of the listed feature keywords",
    )


# task 3 : query task ( what is needed data form DB)


def build_query_task(question: str, user_type: str) -> Task:
    return Task(
        description=f"""
You are receiving a user question from a {user_type}:

---
{question}
---

Determine whether the question requires backend MongoDB data. If so, list ONLY the necessary collections from the next:

- USER_PROFILE


If none are needed, reply with: NOTNEEDED_DATA

Respond only with either USER_PROFILE or NOTNEEDED_DATA.
""",
        agent=query_agent,
        expected_output="Comma-separated MongoDB collections or NOTNEEDED_DATA",
    )


# Task 4 : Final response


def build_final_answer_task(question: str, user_type: str, data: dict) -> Task:
    return Task(
        description=f"""
You are given the following:

- User Type: {user_type}
- Question:
---
{question}
---

- Retrieved Data (if any):
{json.dumps(data, indent=2)}

Your job is to write the most helpful, clear, and personalized answer based on the question and data.

If data is present, use it.
If not, answer based on system knowledge or assumptions.
""",
        agent=final_answer_agent,
        expected_output="A complete answer to the user question",
    )
