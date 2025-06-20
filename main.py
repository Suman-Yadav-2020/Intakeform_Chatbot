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
import re
from datetime import datetime
from typing import Optional, Union

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
     allow_origins=["http://localhost:5173"],  # frontend origin
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)

llm = LLM(
    model="groq/gemma2-9b-it",
    temperature=0.7,
    api_key="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"
)

session_store = {}

class DescriptionRequest(BaseModel):
    description: str

class FollowupStepInput(BaseModel):
    session_id: str
    question: Optional[str] = None
    answer: Optional[str] = None

class AnswerInput(BaseModel):
    session_id: str
    answer: Union[str, list[str]]
# ----------------------- LLM Agents -----------------------


def generate_questions_from_text(text: str) -> list:
    question_agent = Agent(
        role="Medical Intake Question Extractor",
        goal="Extract questions from intake form",
        backstory="Parses forms and creates structured questions, including signature and consent",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    task = Task(
        description=dedent(f"""
You are analyzing a medical intake form.

Your task is to extract:
1. **Patient-facing questions** from the form fields
2. **Consent or declaration statements** (like "I agree", "I consent", "I declare", etc.) and turn them into checkbox-type confirmation questions
3. **Signature and date fields**, especially near consent sections

**Examples:**
- "Full Name" → {{ "question": "What is your full name?", "type": "text" }}
- "DOB" → {{ "question": "What is your date of birth?", "type": "date" }}
- "I consent to treatment" → {{ "question": "Do you agree to the following: 'I consent to treatment'?", "type": "checkbox", "options": ["I agree"] }}
- "Signature" → {{ "question": "Please provide your digital signature", "type": "signature" }}
- "Date" → {{ "question": "What is today's date?", "type": "date" }}

**Return a JSON list of questions in the following format:**
[
  {{
    "question": "What is your full name?",
    "type": "text"
  }},
  {{
    "question": "Do you agree to the following: 'I agree to the terms and conditions'?",
    "type": "checkbox",
    "options": ["I agree" ,I don't agree"]
  }},
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

def validate_answer(answer, question):
    qtype = question.get("type")
    options = question.get("options")

    # Skip validation for signature or text questions
    if qtype in ["signature", "text"]:
        return True

    # Rule-based: date format check
    if qtype == "date":
        if isinstance(answer, str) and answer.strip() == "":
            return True  # allow empty
        try:
            datetime.strptime(answer, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Answer must be a valid date in YYYY-MM-DD format")

    # Rule-based: options check
    if options:
        if isinstance(answer, str) and answer not in options:
            raise ValueError(f"Answer must be one of: {options}")
        if isinstance(answer, list):
            if not all(a in options for a in answer):
                raise ValueError(f"All selected answers must be among: {options}")

    # Skip LLM validation for checkbox or date fields (already rule-based)
    if qtype in ["checkbox", "date"]:
        return True

    # Only run LLM validation for unknown or custom types
    try:
        validator_agent = Agent(
            role="Form Answer Validator",
            goal="Verify if the answer is logically valid for the question",
            backstory="Helps validate answers in a form-filling context and explains why if not valid",
            verbose=False,
            allow_delegation=False,
            llm=llm
        )
        task = Task(
            description=dedent(f"""
            You are validating a form answer. Given the question and answer, determine if the answer makes sense.

            Question: "{question['question']}"
            Answer: "{answer}"

            If the answer is valid, reply exactly: VALID

            If the answer is not valid, reply in the format:
            INVALID: <Short explanation why it's invalid>
            """),
            expected_output="VALID or INVALID: <reason>",
            agent=validator_agent
        )
        crew = Crew(agents=[validator_agent], tasks=[task], verbose=False)
        result = crew.kickoff().raw.strip()

        if result.upper().startswith("VALID"):
            return True
        elif result.upper().startswith("INVALID"):
            reason = result.partition(":")[2].strip()
            raise ValueError(f"LLM Validation failed: {reason}")
        else:
            raise ValueError(f"Unexpected validation output: {result}")
    except Exception as e:
        raise ValueError(f"Answer validation failed: {e}")

# ----------------------- API Endpoints -----------------------
def clean_llm_json(text: str) -> str:
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)  # Remove control characters
    text = re.sub(r",(\s*[}\]])", r"\1", text)  # Trailing commas
    text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE)  # Remove markdown blocks
    return text.strip()

# Dummy session_store (use a database or Redis in production)
session_store = {}

def ask_next_followup_step(original_input: str, previous_answers: dict) -> dict:
    followup_agent = Agent(
        role="Telemedicine Diagnostic Assistant",
        goal="Identify exact single-word disease name or ask follow-up",
        backstory="Acts as a step-by-step medical assistant in a telemedicine chatbot, returning exact disease names for routing to appropriate forms.",
        verbose=False,
        allow_delegation=False,
        llm=llm
    )

    qa_str = "\n".join(f"- {q}: {a}" for q, a in previous_answers.items())

    task = Task(
    description=dedent(f"""
    You are assisting in a telemedicine triage session.

    Symptom: "{original_input}"

    Patient has already answered:
    {qa_str if qa_str else "None"}

    ⚠️ Do not repeat previous questions.

    🤖 Use the **answers to guide your logic**:
    - If they say “No” to exertion-based symptoms, consider **non-cardiac** causes.
    - If they say “Yes” to heartburn or meals, consider **digestive**.

    👉 If more clarification is needed, return:
    {{
    "next_question": {{
        "question": "...",
        "type": "checkbox",
        "options": ["Yes", "No"]
    }}
    }}

    👉 If confident, return:
    {{
    "disease": "gastritis"
    }}

    ⚠️ Disease must be one lowercase word. No commentary.
    """),
    expected_output="Strict JSON with either 'next_question' or 'disease'",
    agent=followup_agent
    )

    crew = Crew(agents=[followup_agent], tasks=[task], verbose=False)
    result = crew.kickoff()
    raw_output = crew.kickoff().raw.strip()
    logging.error(f"[LLM RAW OUTPUT] {raw_output}")  # 👈 critical

    try:
        cleaned = clean_llm_json(raw_output)
        logging.error(f"[CLEANED OUTPUT] {cleaned}")  # 👈 see if it's valid JSON now
        parsed = json.loads(cleaned)
        return parsed
    except Exception as e:
        logging.exception("❌ JSON parsing failed in follow-up step")
        raise ValueError("Failed to parse LLM output for follow-up step")

def start_form_filling_flow(session_id, input_text, disease):
    # Step 1: Classify detailed disease to category using LLM
    category = classify_disease_to_category(disease)

    # Step 2: Load the form PDF based on category
    pdf_path = os.path.join("forms", f"{category}.pdf")
    if not os.path.exists(pdf_path):
        pdf_path = os.path.join("forms", "generic.pdf")

    print(f"🎯 Disease: {disease}")
    print(f"📂 Category: {category}")
    print(f"📄 PDF picked: {pdf_path}")
    # Step 3: Extract text from the PDF
    with pdfplumber.open(pdf_path) as pdf:
        text = ''.join(page.extract_text() or '' for page in pdf.pages)

    if not text.strip():
        raise ValueError("No extractable text in PDF")

    # Step 4: Generate questions from form content
    questions = generate_questions_from_text(text)
    raw_output = clean_llm_json(questions.raw)
    questions_list = json.loads(raw_output)

    # Step 5: Try to prefill answers from user input
    prefilled = prefill_answers_from_questions(input_text, questions_list)
    answers = [prefilled.get(q['question'], None) for q in questions_list]
    unanswered_index = get_next_unanswered_index(answers)

    # Step 6: Store session state
    session_store[session_id] = {
        "questions": questions_list,
        "answers": answers,
        "current_index": unanswered_index,
        "current_phase": "form_filling",
        "disease": disease,
        "category": category
    }

    # Step 7: Return next step
    if unanswered_index is None:
        summary = generate_summary_from_answers(answers, questions_list)
        return {
            "message": "Form fully prefilled!",
            "answers": answers,
            "summary": summary,
            "next_question": None,
            "current_phase": "form_filling"
        }

    return {
        "next_question": questions_list[unanswered_index],
        "session_id": session_id,
        "prefilled_answers": prefilled,
        "current_phase": "form_filling"
    }

def classify_disease_to_category(disease_name: str) -> str:
    classifier_agent = Agent(
        role="Medical Disease Category Classifier",
        goal="Map detailed diseases to general one-word categories",
        backstory="Helps group specific diagnoses into simplified categories for form routing (e.g., cardiac, respiratory, digestive).",
        verbose=False,
        allow_delegation=False,
        llm=llm
    )

    task = Task(
        description=dedent(f"""
You are classifying a medical disease into a general one-word category.

Given the disease: "{disease_name}"

Choose ONE of the following categories:
cardiac
respiratory
neurology
mental
digestive
endocrine
fever
dermatology
pain
urinary
general

Return ONLY the category name. No punctuation. No extra words. Lowercase only.

Examples:
"myocardial infarction" → "cardiac"
"panic attack" → "mental"
"gastritis" → "digestive"
        """),
        expected_output="One of the listed category names",
        agent=classifier_agent
    )

    crew = Crew(agents=[classifier_agent], tasks=[task], verbose=False)
    result = crew.kickoff()
    return result.raw.strip().lower()

@app.post("/load-form")
def load_form_from_description(request: DescriptionRequest):
    try:
        description = request.description
        session_id = f"clarify_{hash(description)}"
        session_store[session_id] = {
            "original_input": description,
            "clarification_state": {"previous_answers": {}},
            "current_phase": "clarification"
        }

        result = ask_next_followup_step(description, {})

        if "next_question" in result:
            session_store[session_id]["clarification_state"]["current_question"] = result["next_question"]["question"]
            return {
                "next_question": result["next_question"],
                "session_id": session_id,
                "current_phase": "clarification"
            }

        elif "disease" in result:
            disease = result["disease"]
            return start_form_filling_flow(session_id, description, disease)

        return {"error": "Unable to classify or continue"}

    except Exception as e:
        logging.error("Error in /load-form", exc_info=True)
        return {"error": str(e)}

@app.post("/next")
def next_question(data: AnswerInput):
    session = session_store.get(data.session_id)
    if not session:
        return {"error": "Session not found"}

    if session.get("current_phase") != "form_filling":
        return {"error": "Not in form filling phase"}

    idx = session["current_index"]
    question = session["questions"][idx]

    try:
        validate_answer(data.answer, question)
    except ValueError as ve:
        return {"error": str(ve)}

    session["answers"][idx] = data.answer
    next_idx = get_next_unanswered_index(session["answers"])

    if next_idx is None:
        summary = generate_summary_from_answers(session["answers"], session["questions"])
        return {
            "message": "Form complete",
            "answers": session["answers"],
            "summary": summary
        }

    session["current_index"] = next_idx
    return {"next_question": session["questions"][next_idx]}

@app.post("/followup-step")
def followup_step(data: FollowupStepInput):
    session = session_store.get(data.session_id)
    if not session or session.get("current_phase") != "clarification":
        return {"error": "Invalid session or not in clarification phase"}

    clarification = session.setdefault("clarification_state", {"previous_answers": {}})
    if data.question and data.answer:
        clarification["previous_answers"][data.question] = data.answer

    result = ask_next_followup_step(
        original_input=session["original_input"],
        previous_answers=clarification["previous_answers"]
    )

    if "next_question" in result:
        session["clarification_state"]["current_question"] = result["next_question"]["question"]
        return {"next_question": result["next_question"]}

    elif "disease" in result:
        disease = result["disease"]
        return start_form_filling_flow(data.session_id, session["original_input"], disease)

    return {"error": "Unexpected result from follow-up logic"}


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
