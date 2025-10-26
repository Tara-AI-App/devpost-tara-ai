from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import httpx
import os
import uuid
import json
import re
import asyncio
import time
import logging
from google.adk.cli.fast_api import get_fast_api_app
from google.genai import types as genai_types

# Configure logger
logger = logging.getLogger(__name__)

# Import follow_up_agent
from follow_up_agent.agent import root_agent

# Import ADK runner for follow_up_agent
from google.adk.runners import InMemoryRunner

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("WARNING: python-dotenv not installed")

# Ensure required environment variables are set
if not os.getenv("GOOGLE_CLOUD_PROJECT"):
    print("WARNING: GOOGLE_CLOUD_CLOUD_PROJECT environment variable not set. Add it to your .env file.")

# Get the ADK FastAPI app (this serves the agents via /run endpoint)
adk_app = get_fast_api_app(
    agents_dir="./",
    allow_origins=["*"],
    web=False
)

# Create our custom FastAPI app
app = FastAPI(title="Course Generator API", version="1.0.0")

# Mount the ADK app to handle /run, /apps, etc. endpoints
app.mount("/adk", adk_app)

# Initialize follow_up_agent runner
follow_up_runner = InMemoryRunner(agent=root_agent, app_name="follow_up_agent")

class CourseRequest(BaseModel):
    user_id: str
    token_github: str
    token_drive: str
    prompt: str
    files_url: str
    cv: str

class GuideRequest(BaseModel):
    user_id: str
    token_github: str
    token_drive: str
    prompt: str

class Lesson(BaseModel):
    content: str
    title: str
    index: int

class QuizChoice(BaseModel):
    A: str
    B: str
    C: str
    D: str = None

class Quiz(BaseModel):
    question: str
    choices: QuizChoice
    answer: str

class Module(BaseModel):
    lessons: List[Lesson]
    title: str
    index: int
    quiz: List[Quiz]

class CourseResponse(BaseModel):
    learning_objectives: List[str]
    description: str
    estimated_duration: int
    modules: List[Module]
    title: str
    source_from: List[str]
    difficulty: str
    skills: List[str]

class GuideResponse(BaseModel):
    title: str
    description: str
    content: str
    source_from: List[str] = []

# Follow-up agent models (ADK API server pattern)
class MessagePart(BaseModel):
    text: str

class Message(BaseModel):
    parts: List[MessagePart]
    role: str

class RunRequest(BaseModel):
    app_name: str
    user_id: str
    session_id: str
    new_message: Message

class SessionResponse(BaseModel):
    status: str
    message: str
    user_id: str = None
    session_id: str = None
    app_name: str = None

# ============================================================================
# Helper Functions
# ============================================================================

async def retry_with_exponential_backoff(
    func,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0
):
    """
    Retry a function with exponential backoff for handling rate limits.

    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay between retries (default: 60.0)
        exponential_base: Base for exponential backoff calculation (default: 2.0)

    Returns:
        Result from the successful function execution

    Raises:
        The last exception if all retries are exhausted or non-rate-limit error occurs
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            error_str = str(e)

            # Check if it's a rate limit error (429, RESOURCE_EXHAUSTED, quota exceeded)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                if attempt < max_retries:
                    logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Retrying in {delay:.1f}s... Error: {error_str[:100]}"
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)
                    continue
                else:
                    logger.error(f"Rate limit retry exhausted after {max_retries + 1} attempts")

            # If it's not a rate limit error, raise immediately
            raise

    # If all retries exhausted, raise the last exception
    raise last_exception

async def run_adk_agent(
    app_name: str,
    user_id: str,
    prompt: str,
    github_token: str = None,
    drive_token: str = None,
    max_retries: int = 3
) -> str:
    """
    Execute agent using InMemoryRunner directly (no HTTP calls).
    
    This function:
    1. Sets tokens in environment (for this request)
    2. Creates agent instance with user-specific tokens
    3. Runs agent via InMemoryRunner
    4. Extracts and returns the final text response
    """
    # Store original tokens
    original_github_token = os.environ.get('GITHUB_PERSONAL_ACCESS_TOKEN')
    original_drive_token = os.environ.get('GOOGLE_DRIVE_TOKEN')
    original_user_id = os.environ.get('USER_ID')

    try:
        # Set tokens for this request
        if github_token:
            os.environ['GITHUB_PERSONAL_ACCESS_TOKEN'] = github_token
        if drive_token:
            os.environ['GOOGLE_DRIVE_TOKEN'] = drive_token
            os.environ['USER_ID'] = user_id

        # Generate session ID
        session_id = f"session-{uuid.uuid4()}"

        # Define the agent execution logic to be retried
        async def execute_agent():
            # Import agent based on app_name
            if app_name == "course_agent":
                from course_agent.agents.course_agent import create_course_agent
                agent_instance = create_course_agent(
                    github_token=github_token,
                    drive_token=drive_token,
                    user_id=user_id
                )
                agent = agent_instance.get_agent()
                runner = InMemoryRunner(agent=agent, app_name="course_agent")
            elif app_name == "guide_agent":
                from guide_agent.agents.guide_agent import create_guide_agent
                agent_instance = create_guide_agent(
                    github_token=github_token,
                    drive_token=drive_token,
                    user_id=user_id
                )
                agent = agent_instance.get_agent()
                runner = InMemoryRunner(agent=agent, app_name="guide_agent")
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Unknown app: {app_name}"
                )

            # Step 1: Create session
            await runner.session_service.create_session(
                user_id=user_id,
                session_id=session_id,
                app_name=app_name
            )

            # Step 2: Run agent
            new_message = genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=prompt)]
            )

            # Collect events and extract final text
            final_text = None
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=new_message
            ):
                if hasattr(event, 'content') and event.content:
                    if hasattr(event.content, 'parts'):
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                final_text = part.text

            if not final_text:
                raise HTTPException(
                    status_code=500,
                    detail="No text response from agent"
                )

            return final_text

        # Execute with retry logic
        return await retry_with_exponential_backoff(execute_agent, max_retries=max_retries)

    finally:
        # Restore original tokens
        if original_github_token:
            os.environ['GITHUB_PERSONAL_ACCESS_TOKEN'] = original_github_token
        elif 'GITHUB_PERSONAL_ACCESS_TOKEN' in os.environ:
            del os.environ['GITHUB_PERSONAL_ACCESS_TOKEN']

        if original_drive_token:
            os.environ['GOOGLE_DRIVE_TOKEN'] = original_drive_token
        elif 'GOOGLE_DRIVE_TOKEN' in os.environ:
            del os.environ['GOOGLE_DRIVE_TOKEN']

        if original_user_id:
            os.environ['USER_ID'] = original_user_id
        elif 'USER_ID' in os.environ:
            del os.environ['USER_ID']

@app.post("/course/generate", response_model=CourseResponse)
async def generate_course(request: CourseRequest):
    try:
        # Call ADK agent internally
        response_text = await run_adk_agent(
            app_name="course_agent",
            user_id=request.user_id,
            prompt=request.prompt,
            github_token=request.token_github,
            drive_token=request.token_drive
        )

        # Parse JSON response
        import json
        import re

        def extract_and_parse_json(text):
            """Extract and parse JSON from agent response with multiple fallback strategies."""
            # Strategy 1: Try direct parsing
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            # Strategy 2: Extract from markdown code blocks
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

            # Strategy 3: Find JSON object with brace matching
            start_idx = text.find('{')
            if start_idx != -1:
                brace_count = 0
                in_string = False
                escape_next = False

                for i in range(start_idx, len(text)):
                    char = text[i]

                    if escape_next:
                        escape_next = False
                        continue

                    if char == '\\':
                        escape_next = True
                        continue

                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue

                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_str = text[start_idx:i+1]
                                try:
                                    # Fix common escape issues
                                    json_str = json_str.replace('\\n', '\\\\n')
                                    json_str = json_str.replace('\\"', '"')
                                    return json.loads(json_str)
                                except json.JSONDecodeError:
                                    continue

            raise ValueError("Could not extract valid JSON from agent response")

        try:
            course_json = extract_and_parse_json(response_text)
        except ValueError as e:
            # Fallback: Use JSON processor agent
            try:
                from course_agent.agents.json_processor_agent import create_json_processor_agent
                from google.adk.runners import InMemoryRunner

                json_agent = create_json_processor_agent()
                runner = InMemoryRunner(agent=json_agent)

                processor_user_id = str(uuid.uuid4())
                processor_session_id = str(uuid.uuid4())

                await runner.session_service.create_session(
                    user_id=processor_user_id,
                    session_id=processor_session_id,
                    app_name=runner.app_name
                )

                import google.genai.types as genai_types
                processor_message = genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=f"Convert this course content to JSON:\\n\\n{response_text}")]
                )

                processor_response = ""
                async for event in runner.run_async(
                    user_id=processor_user_id,
                    session_id=processor_session_id,
                    new_message=processor_message
                ):
                    if hasattr(event, 'content') and event.content:
                        if hasattr(event.content, 'parts'):
                            for part in event.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    processor_response += part.text

                course_json = json.loads(processor_response)

            except Exception as processor_error:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to parse JSON even with processor: {str(processor_error)}"
                )

        # Validate and fix schema issues (same as before)
        if 'estimated_duration' not in course_json:
            course_json['estimated_duration'] = 10
        elif isinstance(course_json.get('estimated_duration'), str):
            duration_str = course_json['estimated_duration']
            match = re.search(r'(\d+)', duration_str)
            if match:
                course_json['estimated_duration'] = int(match.group(1))
            else:
                course_json['estimated_duration'] = 10

        # Add missing fields
        if 'modules' in course_json:
            for mod_idx, module in enumerate(course_json['modules'], 1):
                if 'index' not in module:
                    module['index'] = mod_idx
                if 'lessons' not in module:
                    module['lessons'] = []
                if 'lessons' in module and isinstance(module['lessons'], list):
                    for lesson_idx, lesson in enumerate(module['lessons'], 1):
                        if 'index' not in lesson:
                            lesson['index'] = lesson_idx
                if 'quiz' not in module:
                    module['quiz'] = []

        if 'skills' not in course_json:
            course_json['skills'] = []
        if not isinstance(course_json.get('source_from'), list):
            course_json['source_from'] = []
        if not isinstance(course_json.get('learning_objectives'), list):
            course_json['learning_objectives'] = []

        return CourseResponse(**course_json)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Course generation failed: {str(e)}"
        )

@app.post("/guide/generate", response_model=GuideResponse)
async def generate_guide(request: GuideRequest):
    try:
        # Call ADK agent internally
        response_text = await run_adk_agent(
            app_name="guide_agent",
            user_id=request.user_id,
            prompt=request.prompt,
            github_token=request.token_github,
            drive_token=request.token_drive
        )

        # Parse JSON response
        import json
        try:
            guide_json = json.loads(response_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'```json\s*(\{.*\})\s*```', response_text, re.DOTALL)
            if json_match:
                guide_json = json.loads(json_match.group(1))
            else:
                raise ValueError("Could not parse JSON from agent response")

        # Ensure source_from is a list
        if 'source_from' not in guide_json:
            guide_json['source_from'] = []
        elif not isinstance(guide_json.get('source_from'), list):
            guide_json['source_from'] = []

        return GuideResponse(**guide_json)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Guide generation failed: {str(e)}"
        )

@app.post("/apps/{app_name}/users/{user_id}/sessions/{session_id}", response_model=SessionResponse)
async def create_session(app_name: str, user_id: str, session_id: str):
    """
    Create or update a session for the follow_up_agent.

    This follows the ADK API server pattern for session management.
    """
    try:
        if app_name != "follow_up_agent":
            raise HTTPException(status_code=404, detail=f"App '{app_name}' not found")

        # Create session using the runner's session service
        await follow_up_runner.session_service.create_session(
            user_id=user_id,
            session_id=session_id,
            app_name=app_name
        )

        return SessionResponse(
            status="success",
            message="Session created successfully",
            user_id=user_id,
            session_id=session_id,
            app_name=app_name
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create session: {str(e)}"
        )

@app.post("/run")
async def run_agent(request: RunRequest):
    """
    Run the follow_up_agent with a message and return all events.

    This endpoint allows for multi-turn conversations by maintaining session context.
    """
    try:
        if request.app_name != "follow_up_agent":
            raise HTTPException(status_code=404, detail=f"App '{request.app_name}' not found")

        # Convert the request message to ADK format
        new_message = genai_types.Content(
            role=request.new_message.role,
            parts=[genai_types.Part(text=part.text) for part in request.new_message.parts]
        )

        # Run the agent and collect all events
        events = []
        async for event in follow_up_runner.run_async(
            user_id=request.user_id,
            session_id=request.session_id,
            new_message=new_message
        ):
            # Convert event to dict format for JSON response
            event_dict = {
                "content": {}
            }

            if hasattr(event, 'content') and event.content:
                event_dict["content"]["role"] = event.content.role if hasattr(event.content, 'role') else None
                event_dict["content"]["parts"] = []

                if hasattr(event.content, 'parts'):
                    for part in event.content.parts:
                        part_dict = {}
                        if hasattr(part, 'text') and part.text:
                            part_dict["text"] = part.text
                        if hasattr(part, 'functionCall') and part.functionCall:
                            part_dict["functionCall"] = {
                                "name": part.functionCall.name,
                                "args": dict(part.functionCall.args) if part.functionCall.args else {}
                            }
                        if hasattr(part, 'functionResponse') and part.functionResponse:
                            part_dict["functionResponse"] = {
                                "name": part.functionResponse.name,
                                "response": dict(part.functionResponse.response) if part.functionResponse.response else {}
                            }
                        event_dict["content"]["parts"].append(part_dict)

            events.append(event_dict)

        return events

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Course Generator API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "course_generation": "/course/generate",
            "guide_generation": "/guide/generate",
            "follow_up_agent": {
                "create_session": "/apps/{app_name}/users/{user_id}/sessions/{session_id}",
                "run": "/run"
            }
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
