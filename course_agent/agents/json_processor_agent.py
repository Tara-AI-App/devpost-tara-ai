"""
JSON Post-Processing Agent using ADK output_schema.
This agent takes raw text output and converts it to structured JSON.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from google.adk.agents import Agent


# Define the structured output schema using Pydantic
class QuizChoices(BaseModel):
    A: str
    B: str
    C: str
    D: Optional[str] = None


class Quiz(BaseModel):
    question: str
    choices: QuizChoices
    answer: str


class Lesson(BaseModel):
    title: str
    index: int
    content: str


class Module(BaseModel):
    title: str
    index: int
    lessons: List[Lesson]
    quiz: List[Quiz]


class CourseOutput(BaseModel):
    """Structured course output schema."""
    title: str = Field(description="The course title")
    description: str = Field(description="Course description")
    difficulty: str = Field(description="Course difficulty level: Beginner, Intermediate, or Advanced")
    estimated_duration: int = Field(description="Estimated duration in hours")
    learning_objectives: List[str] = Field(description="List of learning objectives")
    skills: List[str] = Field(description="List of skills covered")
    modules: List[Module] = Field(description="List of course modules")
    source_from: List[str] = Field(description="List of source URLs")


def create_json_processor_agent() -> Agent:
    """
    Create an agent that processes raw course content into structured JSON.

    This agent uses output_schema to ensure clean JSON output.
    Note: Since output_schema is used, this agent CANNOT use tools.
    """

    instruction = """
    You are a JSON formatting expert. Your task is to extract course information from text
    and format it into a valid JSON structure.

    CRITICAL RULES:
    1. Extract ALL information from the input text
    2. Preserve ALL content, code examples, and formatting
    3. Properly escape special characters in JSON strings:
       - Newlines: Use \\n
       - Tabs: Use \\t
       - Backslashes: Use \\\\
       - Quotes: Use \\"
    4. Ensure all arrays and objects are properly closed
    5. Set estimated_duration as a number (not string)
    6. Set module and lesson indices as numbers
    7. If any field is missing, use reasonable defaults:
       - estimated_duration: 10
       - skills: []
       - source_from: []

    IMPORTANT: If the input contains JSON in markdown code blocks (```json...```),
    extract the JSON content directly. If the input is plain text describing a course,
    structure it according to the schema.

    Your output will automatically be validated against the CourseOutput schema.
    """

    return Agent(
        model='gemini-2.0-flash-exp',  # Fast model for post-processing
        name='json_processor',
        description='Processes course content into structured JSON format',
        instruction=instruction,
        output_schema=CourseOutput,  # This enforces structured output
        output_key='processed_course'  # Store result in session state
    )
