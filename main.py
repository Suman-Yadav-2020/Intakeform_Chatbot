from fastapi import FastAPI, File, UploadFile,Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from crewai import Agent, Task, Crew, LLM
from textwrap import dedent
import logging
import pdfplumber
import os
import uvicorn
from crewai import Agent, Task, Crew
from textwrap import dedent
import subprocess
import json
import re
app = FastAPI()
import traceback
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)




logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG if needed
    format='%(asctime)s - %(levelname)s - %(message)s'
)


llm = LLM(
    model="groq/gemma2-9b-it",
    temperature=0.7,
    api_key="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"
)



# Mock in-memory store
session_store = {}

class AnswerInput(BaseModel):
    session_id: str
    answer: str

@app.get("/start")
def start_form():
    session_id = "user123"  # Generate a real session ID
    questions = extract_questions(read_pdf())  # Get all questions once
    session_store[session_id] = {
        "questions": questions,
        "answers": [],
        "current_index": 0
    }
    first_q = questions[0]
    return {"question": first_q, "session_id": session_id}

@app.post("/next")
def next_question(data: AnswerInput):
    session = session_store.get(data.session_id)
    if not session:
        return {"error": "Session not found."}

    # Save current answer
    session["answers"].append(data.answer)

    # Move to next question
    session["current_index"] += 1
    idx = session["current_index"]

    if idx >= len(session["questions"]):
        return {"message": "Form complete", "answers": session["answers"]}

    next_q = session["questions"][idx]
    return {"question": next_q}




def generate_questions_from_text(text: str) -> list:
  
    # Create the agent
    question_extractor = Agent(
        role='Medical Intake Question Extractor',
        goal='Extract clear and concise intake questions and question types (e.g., text, date) from any medical or hospital form',
        backstory='You are an expert at reading and interpreting medical forms to turn them into patient-facing questions.',
        verbose=True,
        allow_delegation=False,
        llm=llm  
    )

    # Define the task
    task = Task(
        description=dedent(f"""
            You are provided with the following intake form content:

            ---
            {text}
            ---

            Your job is to extract and return a list of questions that should be asked to the patient.
            
        """),
        expected_output="A JSON object containing a list of questions at a time and question types(e.g., text, date ).",
        agent=question_extractor,
    )

    # Run the crew
    crew = Crew(
        agents=[question_extractor],
        tasks=[task],
        verbose=True
    )

    result = crew.kickoff(inputs={"input": text})
   
    print(result)
    return result

import re
import json

def generate_questions_array(text):
    lines = text.strip().split("\n")
    questions_list = []

    for line in lines:
        line = line.strip()
        print("🔍 Checking line:", line)
        # Match pattern: * Question: type
        match = re.match(r"\* (.+?):\s*(\w+)", line)
        if match:
            question_text, qtype = match.groups()
            questions_list.append({
                "question": question_text.strip(),
                "type": qtype.strip().lower()
            })
        else:
            print("❌ No match for:", line)

    print("\n✅ Final parsed output:")
    print(json.dumps(questions_list, indent=2))
    return questions_list


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    # logging.info("📩 /upload-pdf endpoint was called")

    try:
        contents = await file.read()
        # logging.info("✅ PDF file read successfully")
        
      
        with open("temp.pdf", "wb") as f:
            f.write(contents)
        # logging.info("📄 PDF saved to temp.pdf")

        # Now try reading it
        with pdfplumber.open("temp.pdf") as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text() or ''

        # logging.info(f"📝 Extracted text length: {len(text)} characters")

        if not text.strip():
            raise ValueError("No extractable text found in PDF.")

        questions = generate_questions_from_text(text)

        print("✅ Questions generated successfully111",questions.raw[0])
        session_id = "user12343343"  # Generate a real session ID
        questions_new = json.loads(questions.raw)  
        session_store[session_id] = {
        "questions": questions_new,
        "answers": [],
        "current_index": 0
        }
        first_q = questions_new[0]
        return {"question": first_q, "session_id": session_id}
        # generate_questions_array(questions.raw)
        # return {"questions1": json.loads(questions.raw)}

    except Exception as e:
        logging.error("❌ Exception occurred:", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)