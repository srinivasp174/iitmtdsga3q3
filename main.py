import os
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI()

# Enable CORS (required)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# Request & Response Models
# -------------------------------

class CodeRequest(BaseModel):
    code: str


class CodeResponse(BaseModel):
    error: List[int]
    result: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


# -------------------------------
# TOOL FUNCTION
# -------------------------------

def execute_python_code(code: str) -> dict:
    """
    Execute Python code and return exact output.

    Returns:
        {
            "success": bool,
            "output": str
        }
    """

    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code)
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}

    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}

    finally:
        sys.stdout = old_stdout


# -------------------------------
# AI ERROR ANALYSIS
# -------------------------------

def analyze_error_with_ai(code: str, tb: str) -> List[int]:
    """
    Uses Gemini structured output to return error line numbers.
    """

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    prompt = f"""
You are a Python debugging assistant.

Given the Python code and its traceback,
identify the line number(s) in the original code
where the error occurred.

Return ONLY the line numbers.

CODE:
{code}

TRACEBACK:
{tb}
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "error_lines": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.INTEGER),
                    )
                },
                required=["error_lines"],
            ),
        ),
    )

    result = ErrorAnalysis.model_validate_json(response.text)
    return result.error_lines


# -------------------------------
# MAIN ENDPOINT
# -------------------------------

@app.post("/code-interpreter", response_model=CodeResponse)
def code_interpreter(request: CodeRequest):

    # Step 1: Execute code
    execution_result = execute_python_code(request.code)

    # Step 2: If success → return immediately
    if execution_result["success"]:
        return {
            "error": [],
            "result": execution_result["output"]
        }

    # Step 3: If failure → analyze with AI
    error_lines = analyze_error_with_ai(
        request.code,
        execution_result["output"]
    )

    # Step 4: Return traceback + AI lines
    return {
        "error": error_lines,
        "result": execution_result["output"]
    }
    
@app.get("/code-interpreter")
def code_interpreter_health_check():
    return {"status": "ready"}