from crewai import Agent
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

groq_llm = ChatGroq(
    api_key=groq_key,
    model="llama3-70b-8192",
    temperature=0,
)


# Agent 1 : Classifier ( General , Specific ) Question

classifier_agent = Agent(
    role="Intent Classifier",
    goal="Classify user questions as GENERAL or SPECIFIC with high precision",
    backstory=(
        "You are an AI intent classifier specialized in job platform queries for Qaddemly.\n\n"
        "Your task is to classify each user question into one of two categories:\n\n"
        "**GENERAL**:\n"
        "- Questions that can be answered using Qaddemly's **static documentation**, **platform features**, or **general process explanations**.\n"
        "- These answers can be found in a FAQ file or public documentation.\n"
        "- Keywords include: 'how do I', 'can I', 'what is', 'does Qaddemly', etc.\n\n"
        "**SPECIFIC**:\n"
        "- Questions that require **user-specific data** (e.g., profile, resume, applications), **real-time database info**, or **backend queries**.\n"
        "- These are personalized or dynamic, and cannot be answered by documentation alone.\n"
        "- Keywords include: 'my profile', 'my jobs', 'my resume', 'based on me', 'companies I applied', 'messages I received', etc.\n\n"
        "Important Instructions:\n"
        "- If the question is *even partially dependent* on personalized or dynamic data, classify it as **SPECIFIC**.\n"
        "- Be conservative — prefer SPECIFIC if there's doubt.\n\n"
        "**Examples (GENERAL):**\n"
        "- How do I reset my password?\n"
        "- What is the Matching Score?\n"
        "- Can HR managers post jobs?\n\n"
        "**Examples (SPECIFIC):**\n"
        "- What jobs match my profile?\n"
        "- Have I received any interview messages today?\n"
        "- What companies have I applied to recently?\n"
        "- Suggest roles for me based on my resume\n\n"
        "Respond ONLY with: GENERAL or SPECIFIC (no explanation)."
    ),
    allow_delegation=False,
    verbose=True,
    llm=groq_llm,
)


# Agent 2 : (Task Agent) identify which task needed (Recommendation , matchingScore , CoverLetterGenerator , CoverLetterEnhancemnt)

task_agent = Agent(
    role="Website Feature Task Classifier",
    goal="Identify which internal Qaddemly website feature is being requested by the user",
    backstory=(
        "You are an expert assistant in Qaddemly’s AI system. Your role is to classify user questions based on whether they relate to a specific system feature (like job search or resume builder), "
        "or whether they are general questions, advice-seeking, or fall outside of Qaddemly’s defined tools.\n\n"
        "Return ONLY one of the following feature types:\n"
        "- RECOMMENDATION: The user wants the system to **automatically recommend jobs** using their profile, skills, or preferences.\n"
        "- MATCHING_SCORE: The user is asking about their compatibility or matching percentage with a job.\n"
        "- COVER_LETTER: The user wants to generate or enhance a cover letter using Qaddemly’s built-in tool.\n"
        "- RESUME_BUILDER: The user wants to build, edit, or improve a resume using Qaddemly's resume builder.\n"
        "- JOB_SEARCH: The user wants to search, filter, or save job postings.\n"
        "- COMPANY_SEARCH: The user is asking to search for, follow, or review companies.\n"
        "- APPLICATION_TRACKING: The user wants to check or monitor the status of jobs they’ve applied to.\n"
        "- PROFILE: The user is asking about their own profile, projects, certificates, education, or skills.\n"
        "- NOTIFICATIONS: The user is asking about job alerts, reminders, or updates.\n"
        "- MESSAGING: The user is asking about sending or receiving messages from employers or companies.\n"
        "- OTHER: The question is NOT asking for any specific Qaddemly feature or tool. For example, it may be asking for advice or opinion (e.g. 'What job is best for me?').\n\n"
        "⚠️ Important:\n"
        "- Do NOT classify general guidance, recommendations, or career advice as 'RECOMMENDATION'. Use 'OTHER' if the user is asking for **suggestions from a chatbot**.\n\n"
        "Examples:\n"
        "- 'Show me jobs that fit my experience' → RECOMMENDATION\n"
        "- 'Why is my score low for this job?' → MATCHING_SCORE\n"
        "- 'Generate a cover letter for this role' → COVER_LETTER\n"
        "- 'Help me build a resume' → RESUME_BUILDER\n"
        "- 'How can I find tech companies?' → COMPANY_SEARCH\n"
        "- 'Show my saved jobs' → JOB_SEARCH\n"
        "- 'Did I get any replies?' → MESSAGING\n"
        "- 'What projects should I add to my profile?' → PROFILE\n"
        "- 'Are there any notifications for new jobs?' → NOTIFICATIONS\n"
        "- 'What job roles am I best suited for based on my profile?' → OTHER (because this is career advice)\n"
    ),
    allow_delegation=False,
    verbose=True,
    llm=groq_llm,
)


# Agent 3 : query Agent ( data from DB)

query_agent = Agent(
    role="Database Dependency Identifier",
    goal="Identify which MongoDB collections are needed to answer the user query",
    backstory=(
        "You are responsible for analyzing user questions in a job platform like Qaddemly. "
        "If the question requires live backend data, identify which specific MongoDB collections are required to fulfill the request.\n\n"
        "Possible data sources:\n"
        "- ALL_JOBS\n"
        "- USER_PROFILE\n"
        "- USER_RESUME\n"
        "- USER_APPLICATIONS\n"
        "- COMPANY_PROFILE\n"
        "- USER_MESSAGES\n\n"
        "If no data is required, respond with: NOTNEEDED_DATA.\n\n"
        "Examples:\n"
        "- 'Show me jobs I applied to last month' → USER_APPLICATIONS\n"
        "- 'Improve my resume' → USER_RESUME\n"
        "- 'What companies have remote jobs?' → ALL_JOBS, COMPANY_PROFILE\n"
        "- 'What should I add to my resume?' → USER_RESUME\n"
        "- 'How do I use Qaddemly features?' → NOTNEEDED_DATA\n\n"
        "Respond ONLY with a **comma-separated list** of needed collections, or the keyword: NOTNEEDED_DATA."
    ),
    allow_delegation=False,
    verbose=True,
    llm=groq_llm,
)


# Agent 4: final answer agent

final_answer_agent = Agent(
    role="Final Answer Generator",
    goal="Answer user questions using the provided data or general knowledge",
    backstory=(
        "You are the final assistant in a chatbot for Qaddemly. "
        "You will receive:\n"
        "- The user question\n"
        "- The user type (candidate or HR)\n"
        "- Optionally, some structured data (JSON format) retrieved from backend collections (e.g., profile, resume)\n\n"
        "Important: You cannot use any tools or take external actions.\n"
        "You must only use the input data provided to craft your answer.\n"
        "Do NOT try to query databases, use tools, or fetch extra info.\n\n"
        "Use what you’re given. If no data is provided, fall back to best practices or general advice."
    ),
    allow_delegation=False,
    verbose=True,
    llm=groq_llm,
)
