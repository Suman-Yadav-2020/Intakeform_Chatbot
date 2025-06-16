from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from crewai import Agent, Task, Crew, LLM
from textwrap import dedent
import logging
import pdfplumber
import os
import json
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)

llm = LLM(
    model="groq/gemma2-9b-it",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY", "your_fallback_key")
)

session_store = {}

class AnswerInput(BaseModel):
    session_id: str
    answer: str | list[str]

# ----------------------- LLM Agents -----------------------

def classify_symptom_with_llm(user_input: str) -> str:
    classifier_agent = Agent(
        role="Medical Condition Classifier",
        goal="Classify symptoms into a medical condition",
        backstory="Classifies patient issues into intake categories",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    task = Task(
        description=dedent(f"""
        Classify the following into a simple condition like:
        fever, orthopedic, dental, diabetes, eye, etc.

        Input:
        "{user_input}"
        """),
        expected_output="One-word label",
        agent=classifier_agent,
    )
    crew = Crew(agents=[classifier_agent], tasks=[task], verbose=True)
    return crew.kickoff(inputs={"input": user_input}).raw.strip().lower()

def generate_questions_from_text(text: str) -> list:
    question_agent = Agent(
        role="Medical Intake Question Extractor",
        goal="Extract questions from intake form",
        backstory="Parses forms and creates structured questions",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    task = Task(
        description=dedent(f"""
You are analyzing a medical intake form.

Your task is to extract **clear, patient-facing questions**. Convert any short labels or fragments into full questions. For example:
- "Full Name" → "What is your full name?"
- "DOB" → "What is your date of birth?"
- "City" → "What city do you live in?"

Return a list of questions with:
- "question": Full, grammatically correct question
- "type": one of "text", "date", "radio", "checkbox", "select"
- "options": List of options, only if type is radio, checkbox, or select

Example format:
[
  {{ "question": "What is your full name?", "type": "text" }},
  {{ "question": "What is your gender?", "type": "radio", "options": ["Male", "Female", "Other"] }},
  ...
]

Use the following form content to generate questions:
---
{text}
---
"""),
        expected_output="JSON list of questions",
        agent=question_agent
    )
    crew = Crew(agents=[question_agent], tasks=[task], verbose=True)
    return crew.kickoff(inputs={"input": text})

def prefill_answers_from_questions(user_input: str, questions: list) -> dict:
    prefill_agent = Agent(
        role="Medical Form Pre-Filler",
        goal="Pre-fill form based on input",
        backstory="Matches free-text user input to form questions",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    q_string = "\n".join(f"- {q['question']}" for q in questions)
    task = Task(
        description=dedent(f"""
Based on this input:
---
{user_input}
---
Try to prefill the following questions:
{q_string}

Return a JSON like:
{{
    "What is your name?": "John",
    "What is your city?": "Austin"
}}
Only include questions you can confidently answer.
"""),
        expected_output="JSON of question: answer pairs",
        agent=prefill_agent,
    )
    crew = Crew(agents=[prefill_agent], tasks=[task], verbose=True)
    result = crew.kickoff(inputs={"input": user_input})
    prefilled = json.loads(result.raw)

    # Fill today’s date for relevant date fields if missing
    today = datetime.today().strftime("%Y-%m-%d")
    for q in questions:
        if q.get("type") == "date" and q["question"] not in prefilled:
            q_text = q["question"].lower()
            if any(keyword in q_text for keyword in ["visit", "today", "appointment", "current date", "date of visit"]):
                prefilled[q["question"]] = today

    return prefilled

def get_next_unanswered_index(answers: list):
    for idx, ans in enumerate(answers):
        if ans is None:
            return idx
    return None

def generate_summary_from_answers(answers: list, questions: list):
    combined = {q['question']: a for q, a in zip(questions, answers) if a is not None}
    summary_agent = Agent(
        role="Medical Summary Generator",
        goal="Summarize completed intake answers",
        backstory="Builds a readable summary from patient responses",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    task = Task(
        description=dedent(f"""
Use the following completed answers to summarize the patient intake:
{json.dumps(combined, indent=2)}

Return a clear, concise paragraph summarizing the patient information.
"""),
        expected_output="Text summary",
        agent=summary_agent
    )
    crew = Crew(agents=[summary_agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    return result.raw

# ----------------------- API Endpoints -----------------------

@app.get("/load-form")
def load_form_from_description(description: str):
    try:
        condition = classify_symptom_with_llm(description)
        logging.info(f"Condition: {condition}")

        pdf_path = os.path.join("forms", f"{condition}.pdf")
        if not os.path.exists(pdf_path):
            pdf_path = os.path.join("forms", "generic.pdf")
        if not os.path.exists(pdf_path):
            return {"error": "Form not found."}

        with pdfplumber.open(pdf_path) as pdf:
            text = ''.join(page.extract_text() or '' for page in pdf.pages)

        if not text.strip():
            raise ValueError("No extractable text in PDF")

        questions = generate_questions_from_text(text)
        questions_list = json.loads(questions.raw)

        prefilled = prefill_answers_from_questions(description, questions_list)

        answers = []
        for q in questions_list:
            ans = prefilled.get(q['question'], None)
            answers.append(ans)

        unanswered_index = get_next_unanswered_index(answers)

        session_id = f"user_{condition}"
        session_store[session_id] = {
            "questions": questions_list,
            "answers": answers,
            "current_index": unanswered_index
        }

        if unanswered_index is None:
            summary = generate_summary_from_answers(answers, questions_list)
            return {
                "message": "Form fully prefilled!",
                "answers": answers,
                "summary": summary,
                "next_question": None
            }

        return {
            "next_question": questions_list[unanswered_index],
            "session_id": session_id,
            "prefilled_answers": prefilled
        }

    except Exception as e:
        logging.error("Error in load-form", exc_info=True)
        return {"error": str(e)}

@app.post("/next")
def next_question(data: AnswerInput):
    session = session_store.get(data.session_id)
    if not session:
        return {"error": "Session not found"}

    idx = session["current_index"]
    session["answers"][idx] = data.answer

    next_idx = get_next_unanswered_index(session["answers"])

    if next_idx is None:
        summary = generate_summary_from_answers(session["answers"], session["questions"])
        return {
            "message": "Form complete",
            "answers": session["answers"],
            "summary": summary,
            "next_question": None
        }

    session["current_index"] = next_idx
    return {
        "next_question": session["questions"][next_idx]
    }

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        filepath = f"temp_{file.filename}"
        with open(filepath, "wb") as f:
            f.write(contents)

        with pdfplumber.open(filepath) as pdf:
            text = ''.join(page.extract_text() or '' for page in pdf.pages)

        if not text.strip():
            raise ValueError("No extractable text in PDF.")

        questions = generate_questions_from_text(text)
        questions_list = json.loads(questions.raw)
        session_id = "upload_user"
        session_store[session_id] = {
            "questions": questions_list,
            "answers": [None] * len(questions_list),
            "current_index": 0
        }

        return {"next_question": questions_list[0], "session_id": session_id}

    except Exception as e:
        logging.error("Upload failed", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

# ----------------------- Start App -----------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
