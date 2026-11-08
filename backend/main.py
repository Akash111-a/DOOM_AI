import os
import json
import tempfile
from pathlib import Path
from typing import Optional, Generator

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from groq import Groq
from pydantic import BaseModel, Field
from pypdf import PdfReader


# -----------------------------------
# Load environment variables
# -----------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from the .env file")

client = Groq(api_key=GROQ_API_KEY)
# openai/gpt-oss-120b
MODEL = "openai/gpt-oss-120b"


# -----------------------------------
# FastAPI application
# -----------------------------------

app = FastAPI(title="DoomAI Resume Chatbot")


# -----------------------------------
# CORS configuration
# -----------------------------------

app.add_middleware(
    CORSMiddleware,
    # allow_origins=[
    #     "http://localhost:5500",
    #     "http://127.0.0.1:5500",
    #     "http://localhost:3000",
    #     "http://127.0.0.1:3000",
    # ],
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------
# Pydantic models
# -----------------------------------

class Experience(BaseModel):
    company: Optional[str] = None
    role: Optional[str] = None
    duration: Optional[str] = None
    description: Optional[str] = None


class Resume(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    summary: Optional[str] = None


class ChatRequest(BaseModel):
    question: str


# -----------------------------------
# Store uploaded resume
# -----------------------------------

current_resume: Optional[Resume] = None


# -----------------------------------
# Read PDF file
# -----------------------------------

def read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text.strip()


# -----------------------------------
# Parse resume using Groq
# -----------------------------------

def parse_resume(resume_text: str) -> Resume:

    prompt = f"""
You are an expert resume parser.

Extract the following information from the resume:

- name
- email
- phone
- skills
- education
- experience
- projects
- summary

Return only valid JSON.

Use this JSON structure:

{{
    "name": "",
    "email": "",
    "phone": "",
    "skills": [],
    "education": [],
    "experience": [
        {{
            "company": "",
            "role": "",
            "duration": "",
            "description": ""
        }}
    ],
    "projects": [],
    "summary": ""
}}

Resume:
{resume_text}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert resume parser. Return only valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content

    try:
        resume_data = json.loads(content)
        return Resume(**resume_data)

    except (json.JSONDecodeError, TypeError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not parse resume JSON: {error}"
        )


# -----------------------------------
# Home route
# -----------------------------------

@app.get("/")
def home():
    return {
        "message": "Welcome to DoomAI",
        "status": "Backend is running"
    }


# -----------------------------------
# Upload resume route
# -----------------------------------

@app.post("/upload")
async def upload_resume(file: UploadFile = File(...)):

    global current_resume

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Please select a file"
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(await file.read())
            temp_path = temp_file.name

        resume_text = read_pdf(temp_path)

        if not resume_text:
            raise HTTPException(
                status_code=400,
                detail="No text found in the PDF"
            )

        current_resume = parse_resume(resume_text)

        return {
            "message": "Resume uploaded successfully",
            "resume": current_resume.model_dump()
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Resume upload failed: {error}"
        )

    finally:
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.remove(temp_path)


# -----------------------------------
# Get uploaded resume
# -----------------------------------

@app.get("/resume")
def get_resume():

    if current_resume is None:
        raise HTTPException(
            status_code=404,
            detail="No resume uploaded yet"
        )

    return current_resume.model_dump()


# -----------------------------------
# Streaming chatbot function
# -----------------------------------

def ask_candidate(
    question: str,
    resume: Resume
) -> Generator[str, None, None]:

    resume_json = resume.model_dump_json(indent=2)

    system_prompt = f"""
You are DoomAI, an intelligent resume assistant.

You are answering questions about the candidate's resume.

Use the following resume information:

{resume_json}

Instructions:
- Answer based on the resume.
- Do not invent experience, skills, education, or projects.
- If the answer is not available in the resume, say so.
- Give clear and helpful answers.
- Use simple language.
"""

    # IMPORTANT:
    # stream=True enables streaming from Groq.
    # Only one API request is made here.

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ],
        stream=True
    )

    # Send each generated text chunk to the frontend

    for chunk in stream:

        if not chunk.choices:
            continue

        content = chunk.choices[0].delta.content

        if content:
            yield content


# -----------------------------------
# Chat route with streaming
# -----------------------------------

@app.post("/chat")
def chat(request: ChatRequest):

    if current_resume is None:
        raise HTTPException(
            status_code=400,
            detail="Please upload your resume first"
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    return StreamingResponse(
        ask_candidate(question, current_resume),
        media_type="text/plain; charset=utf-8"
    )