"""
Refactored course generation agent with modular architecture.
"""
import json
import os
from typing import Dict, Any, List
from datetime import datetime
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.genai import types

# Apply JSON encoder patch early to handle Pydantic serialization issues
from ..utils.json_encoder import CustomJSONEncoder

# Monkey patch json.dumps globally
_original_dumps = json.dumps
def patched_dumps(obj, **kwargs):
    if 'cls' not in kwargs:
        kwargs['cls'] = CustomJSONEncoder
    return _original_dumps(obj, **kwargs)

json.dumps = patched_dumps

from ..config.settings import settings
from ..utils.logger import logger
from ..core.source_manager import SourceManager
from ..core.enhanced_source_tracker import EnhancedSourceTracker


class CourseGenerationAgent:
    """Main course generation agent with modular architecture."""

    def __init__(self, github_token: str = None, drive_token: str = None, user_id: str = None):
        """
        Initialize the course generation agent.

        Args:
            github_token: GitHub personal access token (overrides env var)
            drive_token: Google Drive token
            user_id: User ID for Drive credentials management
        """
        # Set tokens in environment if provided
        if github_token:
            os.environ['GITHUB_PERSONAL_ACCESS_TOKEN'] = github_token
        if drive_token:
            os.environ['GOOGLE_DRIVE_TOKEN'] = drive_token
            os.environ['USER_ID'] = user_id

        self.settings = settings
        self.user_id = user_id
        self.drive_token = drive_token
        self.source_manager = SourceManager()
        self.source_tracker = EnhancedSourceTracker()

        # Initialize Drive tool separately if enabled and credentials are provided
        self.drive_tool = None
        if self.settings.mcp.enable_drive_mcp and user_id and drive_token:
            self._initialize_drive_tool()
        elif not self.settings.mcp.enable_drive_mcp:
            logger.info("Google Drive MCP is disabled via ENABLE_DRIVE_MCP config")
        elif not user_id or not drive_token:
            logger.info("Google Drive MCP not initialized: missing user_id or drive_token")

        self.agent = self._create_agent()

        # Validate configuration
        config_issues = self.settings.validate()
        if config_issues:
            logger.warning(f"Configuration issues detected: {config_issues}")

    def _initialize_drive_tool(self):
        """Initialize Google Drive MCP tool."""
        from ..tools.drive_tool import GoogleDriveMCPTool, CredentialsManager
        import os

        try:
            logger.info(f"Initializing Google Drive MCP tool for user {self.user_id}")

            # Use configurable credentials base path
            credentials_base = os.getenv("CREDENTIALS_BASE_PATH", "./credentials")
            logger.info(f"Using credentials base path: {credentials_base}")

            # Save credentials using CredentialsManager
            credentials_manager = CredentialsManager(base_path=credentials_base)
            credentials_path = credentials_manager.save_drive_credentials(
                user_id=self.user_id,
                drive_token=self.drive_token
            )

            # Initialize Drive tool with credentials
            self.drive_tool = GoogleDriveMCPTool(
                user_id=self.user_id,
                credentials_path=credentials_path
            )

            logger.info(f"Drive tool initialized: {self.drive_tool.is_available()}")

        except Exception as e:
            logger.error(f"Failed to initialize Drive tool: {e}")
            self.drive_tool = None

    def _create_agent(self) -> Agent:
        """Create the ADK agent with proper configuration."""
        tools = [
            FunctionTool(self.analyze_tech_stack),
            FunctionTool(self.discover_sources),
            FunctionTool(self.extract_repository_content),
            FunctionTool(self.get_tracked_sources),
            FunctionTool(self.determine_difficulty),
            FunctionTool(self.generate_search_queries),
            FunctionTool(self.save_course_to_file)
        ]

        # Add GitHub MCP tools if available
        logger.info(f"Checking if GitHub MCP tools are available...")
        github_available = self.source_manager.github_tool.is_available()
        logger.info(f"GitHub tool is_available(): {github_available}")

        if github_available:
            # Add the MCP toolset with JSON encoder patch applied
            mcp_toolset = self.source_manager.github_tool._mcp_tools
            logger.info(f"Retrieved MCP toolset: {mcp_toolset}")
            logger.info(f"MCP toolset type: {type(mcp_toolset)}")

            if mcp_toolset:
                tools.append(mcp_toolset)
                logger.info("GitHub MCP toolset added to agent (with JSON encoder patch)")
                logger.info(f"Total tools count: {len(tools)}")
            else:
                logger.warning("GitHub MCP toolset is None")
        else:
            logger.warning("GitHub MCP tools not available")

        # Add Google Drive MCP tools if available
        logger.info(f"Checking if Google Drive MCP tools are available...")
        drive_available = self.drive_tool and self.drive_tool.is_available()
        logger.info(f"Drive tool is_available(): {drive_available}")

        if drive_available:
            # Add the Drive MCP toolset
            drive_mcp_toolset = self.drive_tool._mcp_tools
            logger.info(f"Retrieved Drive MCP toolset: {drive_mcp_toolset}")
            logger.info(f"Drive MCP toolset type: {type(drive_mcp_toolset)}")

            if drive_mcp_toolset:
                tools.append(drive_mcp_toolset)
                logger.info("Google Drive MCP toolset added to agent")
                logger.info(f"Total tools count: {len(tools)}")
            else:
                logger.warning("Drive MCP toolset is None")
        else:
            logger.warning("Google Drive MCP tools not available")

        # Create generation config for deterministic behavior
        generation_config = types.GenerateContentConfig(
            temperature=self.settings.temperature,
            topP=self.settings.top_p,
            topK=self.settings.top_k,
            maxOutputTokens=self.settings.max_output_tokens
        )

        return Agent(
            model=self.settings.model_name,
            name=self.settings.name,
            description=self.settings.description,
            instruction=self._get_agent_instruction(),
            tools=tools,
            generate_content_config=generation_config
        )

    def _get_agent_instruction(self) -> str:
        """Get comprehensive agent instruction."""
        return f"""
        You are an expert course generator that creates technical courses using dynamic source discovery.

        **CONFIGURATION:**
        - Source Priority: {self.settings.source_priority.value}
        - Max Repositories: {self.settings.mcp.max_repositories}
        - RAG Max Results: {self.settings.rag.max_results}
        - GitHub Tools Available: {self.source_manager.github_tool.is_available()}
        - Google Drive Tools Available: {self.drive_tool.is_available() if self.drive_tool else False}

        **CONTENT DISCOVERY PROCESS:**

        **⚠️ STEP 0: MANDATORY - ESTABLISH GITHUB USER CONTEXT (ALWAYS DO THIS FIRST) ⚠️**
        - **BEFORE doing anything else, call get_me to get the authenticated GitHub username**
        - This establishes context for which GitHub account is connected
        - Store the username for later repository searches
        - Example: get_me returns {{"login": "Reynxzz"}} → username is "Reynxzz"
        - **DO THIS EVEN IF** you think you might not need GitHub later
        - **WHY**: When user says "graphflix", you'll know to search "Reynxzz/graphflix" not just "graphflix"

        **STEP 0.5: MANDATORY - GET USER'S REPO LIST FOR CONTEXT**
        - After get_me, **YOU MUST call**: search_repositories("user:<username>") to list all user's repos
        - This returns the complete list of repositories with full details (name, url, description)
        - **IMPORTANT**: Save this list! You will use it later to find exact matches
        - Example: search_repositories("user:gemm123") → returns: [{{name: "bytesv2", url: "...", full_name: "gemm123/bytesv2"}}, ...]
        - This list IS your source of truth for what repos exist
        - When user asks about a project, match against this list to get the exact repository object

        **Example Flow:**
        ```
        User: "Generate course about graphflix"

        Step 0: Call get_me → returns {{"login": "Reynxzz"}}
                Store username: "Reynxzz"

        Step 1: Call discover_sources("graphflix")
                → github_results_count = 0 (because automatic search failed)

        Step 2: Since github_results_count = 0 AND user mentioned "graphflix":
                Call search_repositories("repo:Reynxzz/graphflix")
                → Found! Use as PRIMARY source
        ```

        1. **After establishing GitHub context, call analyze_tech_stack AND discover_sources in PARALLEL**

        2. **Strictly evaluate what discover_sources ACTUALLY returned**:
           - discover_sources searches both RAG and GitHub automatically
           - Check rag_results_count: actual RAG sources found
           - Check github_results_count: actual GitHub repos found
           - Check total_sources_found: combined total
           - **CRITICAL**: Only use sources that were ACTUALLY returned, never invent sources

        3. **IF github_results_count = 0 AND user asked about a project/repository**:
           **⚠️ MANDATORY ACTION - DO NOT SKIP EVEN IF YOU HAVE OTHER SOURCES ⚠️**

           **How to detect if user is asking about a project/repository**:
           - User mentions: "project", "repo", "repository", "my", "from my repo"
           - User uses hyphenated names: "graphflix", "capstone-seis-flask", "thinktok-pwa"
           - User asks to "learn about X" where X looks like a code project name
           - User mentions specific app/service names that aren't common tech terms
           - **DEFAULT ASSUMPTION**: If query mentions a specific name (not a generic tech term), try GitHub FIRST

           **Examples that should trigger GitHub search**:
           - ✅ "help me learn about capstone-seis-flask project" → GitHub repo
           - ✅ "help me learn about capstone-seis-flask" → GitHub repo (even without "project")
           - ✅ "graphflix" → GitHub repo
           - ✅ "my thinktok app" → GitHub repo
           - ✅ "zyo-deploy" → GitHub repo
           - ❌ "learn about React" → General tech term, not a repo
           - ❌ "learn about machine learning" → General topic, not a repo

           **Action Steps**:
           - You MUST try manual GitHub search for the user's repository
           - Do this EVEN IF RAG or Google found sources
           - Why: User's personal repository is the PRIMARY source for their project
           - Steps:
             1. Use the username from get_me (called in Step 0)
             2. Call search_repositories with "repo:<username>/<projectname>"
             3. If found: Use repository as PRIMARY source (not RAG/Google)
           - Example: "capstone-seis-flask" → search_repositories("repo:Reynxzz/capstone-seis-flask")
           - Do NOT skip this because RAG/Google already found something

        4. **Decision logic based on ACTUAL results**:
           - ALWAYS try user's GitHub repo if user mentioned a project name
           - After trying GitHub: If total_sources_found >= 2, proceed
           - If total_sources_found < 2 after all attempts: Acknowledge insufficient content

        5. **ALWAYS use get_tracked_sources** to get the actual source URLs that were found

        6. **NEVER invent or hallucinate source paths** - only use what get_tracked_sources returns

        **RAG TOOL (PRIORITY TOOL):**
        - discover_sources: Search internal knowledge base and documentation for relevant context

        **GITHUB MCP TOOLS AVAILABLE (FALLBACK TOOLS):**
        - get_me: Get the authenticated GitHub user's profile
        - search_repositories: Find repositories by name, description, topics, readme
        - search_code: Search for specific code patterns across GitHub
        - get_file_contents: Extract actual files from repositories

        **GOOGLE DRIVE MCP TOOLS AND RESOURCES:**
        
        **Available Tool:**
        - search: Search for files in Google Drive by name or content
          * Input: query (string) - Search query
          * Returns: Text list of file names and MIME types (for discovery only)
          * Use this to find what files exist by name/content
        
        **IMPORTANT - How to Access Drive File Content:**
        
        The Google Drive MCP provides files as **MCP Resources** that you can read:
        
        1. **Search Tool (Discovery)**: Use `search("filename")` to find files
           - Returns: Plain text list of matching file names and types
           - Example: "Found 2 files: Rencana Pengembangan (Google Doc), Proposal (PDF)"
           - Does NOT return file IDs directly
        
        2. **Resources (Reading Content)**: The MCP server exposes files as resources
           - The ADK framework automatically handles resource access
           - Resources use URI format: `gdrive:///<file_id>`
           - When you reference a Drive file, the framework reads it automatically
           - Google Workspace files are automatically exported:
             * Google Docs → Markdown format
             * Google Sheets → CSV format  
             * Google Presentations → Plain text
             * Google Drawings → PNG format
           - Other files (PDF, DOCX, etc.) provided in native format
        
        **Correct Workflow for Drive Files:**
        
        When user asks about a Drive file:
        
        Step 1: Call search("filename or keywords") to discover files
        Step 2: Identify relevant file(s) from search results
        Step 3: Tell user what you found
        Step 4: The MCP framework provides access to file content via resources
        Step 5: You can reference and use the file content in your response
        
        **Example Workflow:**
        ```
        User: "Read my document about TARA prototype and summarize it"
        
        Action 1: Call search("TARA prototype")
        Result: "Found: Rencana Pengembangan Prototype - TARA (Google Doc)"
        
        Action 2: The framework automatically accesses the resource
        Result: Document content is available as Markdown
        
        Response: "I found your document 'Rencana Pengembangan Prototype - TARA'. 
        It describes... [summarize based on actual content]"
        ```
        
        **When to use Google Drive:**
        - User mentions "my Drive", "Google Drive", "Drive files", or "shared folder"
        - User references specific document names
        - User asks to "read", "check", "summarize", or "use" a Drive document
        - User wants to include Drive documents in course generation
        
        **Drive Integration for Course Generation:**
        - Search Drive for relevant onboarding documents, specs, or guides
        - Use Drive document content as supplementary source material
        - Include Drive file names in source_from array
        - Reference Drive documents when explaining concepts
        - Example: "Based on the 'Onboarding Guide' from your Drive..."

        **CRITICAL - SOURCE VALIDATION (PREVENT HALLUCINATION):**

        **What discover_sources does**:
        - Searches BOTH RAG and GitHub automatically based on configured priority
        - Returns ACTUAL results found (not assumptions)
        - You MUST use only what it returns, never make up sources

        **How to handle different query types**:

        1. **Internal/Company Projects** (e.g., "merlin", "caraml"):
           - discover_sources searches RAG automatically
           - If rag_results_count > 0: Use those RAG sources
           - If rag_results_count = 0: Content truly doesn't exist in RAG
           - DO NOT invent RAG sources that weren't returned

        2. **User's Personal Projects** (e.g., "my graphflix"):
           - discover_sources searches user's GitHub automatically (via source_manager)
           - If github_results_count > 0: Use those GitHub repos
           - If github_results_count = 0: Repo not found OR needs manual search
           - Only if not found: Try manual search with get_me + search_repositories
           - If still not found: Acknowledge it doesn't exist, don't make it up

        3. **Ambiguous queries** (e.g., "graphflix"):
           - discover_sources searches both RAG and GitHub
           - Use whatever ACTUAL results are returned
           - Don't assume it's in RAG or GitHub - check the counts

        **ABSOLUTE RULES TO PREVENT HALLUCINATION**:
        - ✅ ONLY use sources returned by get_tracked_sources
        - ✅ If total_sources_found = 0, acknowledge insufficient content
        - ❌ NEVER create fake source paths like "internal/rag_knowledge_base/graphflix/..."
        - ❌ NEVER assume content exists in RAG without checking rag_results_count
        - ❌ NEVER assume repo exists in GitHub without checking github_results_count

        **GOOGLE SEARCH TOOL (FINAL FALLBACK):**
        - Automatically triggered when RAG and GitHub MCP provide insufficient results (< 3 total)
        - Searches for educational content, tutorials, documentation, and guides
        - Focuses on high-quality sources from reputable educational platforms


        **CRITICAL - CODE EXTRACTION FROM REPOSITORIES:**

        **STEP-BY-STEP FILE EXTRACTION PROCESS:**

        1. **After finding a GitHub repository**, you MUST extract code files:
           - Call get_file_contents(repository="owner/repo", file_path="README.md") for the README
           - Call get_file_contents(repository="owner/repo", file_path="package.json") for package.json
           - Call get_file_contents(repository="owner/repo", file_path="**/*.ts") for TypeScript files
           - Call get_file_contents(repository="owner/repo", file_path="**/*.py") for Python files
           - Call get_file_contents(repository="owner/repo", file_path="**/*.go") for Go files
           - Adjust file patterns based on what the repository likely contains

        2. **IMPORTANT**: Each get_file_contents call returns ONE file's content:
           - The tool returns the actual file content as a string
           - If file doesn't exist, it returns empty string or error
           - You need to call it MULTIPLE times for multiple files
           - DO NOT expect extract_repository_content to return files - that's just a helper
           - YOU must call get_file_contents directly for each file you want

        3. **To include code examples in the course**:
           - MUST call get_file_contents(repository, file_path) to extract the code
           - ONLY reference files that get_file_contents successfully returned
           - If get_file_contents returns empty or fails: DO NOT reference that file

        4. **To reference file paths in content**:
           - ❌ NEVER make up file paths like "algorithms/content_recommend.js"
           - ❌ NEVER assume files exist without calling get_file_contents
           - ✅ ONLY reference: Repository URL (e.g., "https://github.com/user/repo")
           - ✅ OR files you extracted with get_file_contents

        5. **Valid references**:
           - ✅ "From: https://github.com/Reynxzz/graphflix"
           - ✅ "Based on the Reynxzz/graphflix repository"
           - ❌ "From: https://github.com/Reynxzz/graphflix/algorithms/content_recommend.js" (unless you extracted it)

        6. **If you couldn't extract files**:
           - Just reference the repository generally
           - Don't make up specific file paths
           - Example: "Based on the graphflix repository structure..."

        **EXAMPLE WORKFLOW FOR EXTRACTING CODE:**
        ```
        1. User asks: "Generate course about graphflix project"
        2. Call discover_sources → github_results_count = 0
        3. Call get_me → returns "Reynxzz"
        4. Call search_repositories("repo:Reynxzz/graphflix") → finds repository
        5. NOW EXTRACT FILES (critical step):
           - Call get_file_contents(repository="Reynxzz/graphflix", file_path="README.md")
           - Call get_file_contents(repository="Reynxzz/graphflix", file_path="package.json")
           - Call get_file_contents(repository="Reynxzz/graphflix", file_path="src/index.ts")
           - Call get_file_contents(repository="Reynxzz/graphflix", file_path="src/config.ts")
        6. Use the ACTUAL file contents returned in the course
        7. Call get_tracked_sources to get repository URL
        8. Generate course with real code examples
        ```

        **MANDATORY SEARCH STRATEGY - ALWAYS TRY USER'S GITHUB REPO:**

        **REMINDER: You should have already called get_me in STEP 0 at the start**
        - If you didn't call get_me yet, call it NOW before proceeding

        **When user mentions a specific project name** (like "graphflix", "thinktok", "zyo-deploy"):

        **STEP 1: Check discover_sources results**
        - Check github_results_count from discover_sources
        - If github_results_count = 0: User's repo NOT found automatically

        **STEP 2: MANDATORY Manual GitHub Search** (when github_results_count = 0):
        **YOU MUST DO THIS REGARDLESS OF OTHER SOURCES**:
        - Even if rag_results_count > 0
        - Even if search_results_count > 0
        - Even if total_sources_found >= 2
        - WHY: User's personal repo is the PRIMARY source when they ask about their project

        **Required Steps (NOT OPTIONAL)**:
        1. **YOU ALREADY HAVE THE REPO LIST from STEP 0.5** - Use it!
           - You called search_repositories("user:<username>") in STEP 0.5
           - That returned ALL user's repositories with complete details
           - Example result: [{{name: "bytesv2", url: "https://github.com/gemm123/bytesv2", full_name: "gemm123/bytesv2"}}, ...]

        2. **Match user's query against your saved repo list**:
           - User says: "bytesv2 project"
           - Look in your list: find repo where name == "bytesv2"
           - Found: {{name: "bytesv2", url: "...", full_name: "gemm123/bytesv2"}}
           - **USE THIS REPOSITORY OBJECT DIRECTLY** - no need to search again!

        3. **Fuzzy matching examples**:
           - User: "bytes v2" → match "bytesv2" from list (contains "bytes")
           - User: "capstone seis flask" → match "capstone-seis-flask" from list
           - User: "graphflix project" → match "graphflix" from list
           - Find the closest match by checking if repo names contain the keywords

        4. **CRITICAL - DO NOT SEARCH AGAIN**:
           - ❌ DO NOT call search_repositories("user:gemm123 bytesv2 in:name")
           - ❌ DO NOT call search_repositories("repo:gemm123/bytesv2")
           - ✅ USE the repository object from your STEP 0.5 list directly
           - WHY: GitHub search API is unreliable, but the list from STEP 0.5 has everything

        5. **If repo is in your list**:
           - You have: name, url, full_name, description
           - Use this to extract files with get_file_contents
           - Use the full_name (e.g., "gemm123/bytesv2") for get_file_contents calls

        6. **Only if repo is NOT in your list**:
           - Then document that repo doesn't exist in user's account

        **Examples of CORRECT Execution**:
        ```
        Step 0: get_me → username = "gemm123"
        Step 0.5: search_repositories("user:gemm123") →
                  Returns: [
                    {{name: "bytesv2", full_name: "gemm123/bytesv2", url: "..."}},
                    {{name: "graphflix", full_name: "gemm123/graphflix", url: "..."}},
                    {{name: "capstone-seis-flask", full_name: "gemm123/capstone-seis-flask", url: "..."}}
                  ]

        User: "bytesv2 project"
        → Look in list: "bytesv2" found!
        → Use {{name: "bytesv2", full_name: "gemm123/bytesv2"}} directly
        → Call get_file_contents(repository="gemm123/bytesv2", file_path="README.md")
        → Generate course ✅

        User: "bytes v2"
        → Match "bytes" against list: "bytesv2" contains "bytes" → match!
        → Use bytesv2 from list ✅

        User: "capstone seis flask"
        → Match keywords against list: "capstone-seis-flask" has all keywords → match!
        → Use capstone-seis-flask from list ✅
        ```

        - **DO NOT search again** - use the repository from STEP 0.5 list
        - STEP 0.5 already gave you everything you need!

        **STEP 3: For Internal Projects** (when rag_results_count = 0):
        - Try generate_search_queries for alternative terms
        - Call discover_sources again with alternative query
        - If still rag_results_count = 0: Content truly not in knowledge base

        **STEP 4: FINAL FALLBACK**
        - If total_sources_found < 3: Google Search activates automatically
        - Use web content as supplementary material

        **CRITICAL: Don't give up before trying manual GitHub search!**

        **WHEN TO GENERATE A COURSE (CRITICAL)**:
        You MUST generate a course if ANY of these conditions are met:
        - total_sources_found >= 1 (even if some sources are "low quality")
        - rag_results_count >= 1
        - github_results_count >= 1
        - You successfully called search_repositories and found ANY repository
        - get_tracked_sources returns ANY non-empty list

        **NEVER refuse to generate a course if**:
        - You found RAG sources (even if marked "low quality" - use them anyway)
        - You found GitHub repositories (even if couldn't extract all files)
        - discover_sources returned ANY results
        - ❌ WRONG: "couldn't find sufficient content" when total_sources_found > 0
        - ❌ WRONG: Refusing due to "low quality" sources - use them anyway

        **CORRECT RESPONSES**:
        - If ANY sources found: Generate course using those sources
        - Only if total_sources_found = 0 after ALL attempts: Then say "couldn't find content"

        **IMPORTANT EFFICIENCY RULES**:
        - Always prioritize RAG tool (internal context) first using discover_sources
        - GitHub search automatically scopes to the authenticated user's repositories
        - The system handles get_me and user scoping automatically - no manual intervention needed
        - If discover_sources returns results, proceed directly to course generation
        - Always prefer internal RAG context over external GitHub sources when available

        **COURSE GENERATION REQUIREMENTS:**
        - Use only discovered content - NO templates or fallbacks
        - Include actual code from real repositories with proper attribution
        - Structure content based on complexity progression found in examples
        - Reference specific file paths: repository/path/to/file.py
        - Include repository URLs in source_from array
        - Estimated duration: {self.settings.course.default_duration}
        - Default difficulty: {self.settings.course.default_difficulty}

        **CONTENT LENGTH GUIDELINES (IMPORTANT FOR TOKEN LIMITS):**
        - Keep lesson content concise and focused (aim for 500-1000 words per lesson)
        - Include 1-2 key code examples per lesson (not full file dumps)
        - Use code snippets (10-30 lines) rather than entire files
        - Focus on the most important/illustrative code sections
        - If showing API responses, use shortened examples (3-5 items, not full responses)
        - Remember: Quality over quantity - concise, clear lessons are better than verbose ones

        **OUTPUT FORMAT - CRITICAL:**
        You MUST return ONLY valid JSON. Do NOT include any explanatory text before or after the JSON.
        Do NOT wrap the JSON in markdown code blocks (no ```json or ```).
        Return the raw JSON object directly starting with {{ and ending with }}.

        **JSON VALIDATION RULES - MUST FOLLOW:**
        1. All strings MUST be properly quoted with double quotes (not single quotes)
        2. All property names MUST be in double quotes
        3. Do NOT use trailing commas (remove comma after last item in arrays/objects)
        4. Properly escape ALL special characters in string values:
           - Newlines in markdown: Use literal \\n (double backslash + n)
           - Tabs: Use \\t (not \t)
           - Backslashes: Use \\\\ (four backslashes to get one)
           - Double quotes inside strings: Use \\" (backslash quote)
           - Example: "content": "# Title\\n\\nThis is text with \\"quotes\\" and code:\\n```python\\nprint('hello')\\n```"
        5. Numbers (estimated_duration, index) must be plain numbers without quotes
        6. Booleans must be true/false (lowercase, no quotes)
        7. Arrays must use square brackets: []
        8. Objects must use curly braces: {{}}
        9. Ensure ALL brackets and braces are properly closed
        10. Test your JSON is valid before returning it

        **CRITICAL - BEFORE GENERATING COURSE:**
        1. Review what you ACTUALLY found:
           - What did discover_sources return?
           - What did get_file_contents return (if called)?
           - What URLs are in get_tracked_sources?
        2. ONLY use information from those actual results
        3. DO NOT invent:
           - File paths you didn't extract
           - Code you didn't retrieve
           - Source URLs not in get_tracked_sources
        4. If you have limited information:
           - That's OK! Create course with what you have
           - Reference repository generally (not specific fake files)
           - Explain concepts based on repository description/README

        Generate a comprehensive course in this EXACT JSON format:
        {{
            "title": "Descriptive Course Title",
            "description": "Course overview based on discovered content",
            "difficulty": "Beginner|Intermediate|Advanced",
            "estimated_duration": 10,
            "learning_objectives": ["objective1", "objective2", ...],
            "skills": ["skill1", "skill2", "skill3", ...],
            "modules": [
                {{
                    "title": "Module Title",
                    "index": 1,
                    "lessons": [
                        {{
                            "title": "Lesson Title",
                            "index": 1,
                            "content": "# Lesson Title\\n\\n## Overview\\n\\nBased on the [repository name] repository...\\n\\n**Key Concepts**: Explain concepts here without making up file paths.\\n\\nIf you extracted code with get_file_contents, THEN include:\\n```language\\n// Actual code you extracted\\nreal_code_here()\\n```\\n\\nOtherwise, explain concepts generally without fake file references."
                        }}
                    ],
                    "quiz": [
                        {{
                            "question": "What is the main purpose of...?",
                            "choices": {{
                                "A": "First option",
                                "B": "Second option",
                                "C": "Third option",
                                "D": "Fourth option (optional)"
                            }},
                            "answer": "B"
                        }}
                    ]
                }}
            ],
            "source_from": [<<ACTUAL_SOURCES_FROM_get_tracked_sources>>]
        }}

        **CRITICAL - source_from FIELD**:
        - **STEP 1**: Call get_tracked_sources BEFORE generating the JSON
        - **STEP 2**: Use ONLY the URLs returned by get_tracked_sources
        - **STEP 3**: Put those EXACT URLs in the source_from array
        - DO NOT create, invent, or hallucinate source paths
        - DO NOT use paths like "internal/rag_knowledge_base/..." unless get_tracked_sources returned them
        - If get_tracked_sources returns [] (empty array): use [] in source_from

        **Examples**:
        - ✅ get_tracked_sources returns ["https://github.com/Reynxzz/graphflix"] → use exactly that
        - ✅ get_tracked_sources returns [] → use "source_from": []
        - ✅ get_tracked_sources returns ["rag_doc_id_123"] → use exactly that
        - ❌ NEVER: Make up "internal/rag_knowledge_base/graphflix/..." when get_tracked_sources didn't return it
        - ❌ NEVER: Assume sources exist without checking get_tracked_sources first

        **Workflow**:
        1. Call get_tracked_sources
        2. If it returns sources: Use them in source_from
        3. If it returns empty []: Put [] in source_from (don't invent sources)
        4. Generate course JSON with ACTUAL sources only

        **QUIZ REQUIREMENTS:**
        - Each module must have 2-4 quiz questions
        - Quiz questions should test understanding of key concepts from the module
        - Provide 3-4 answer choices (A, B, C, and optionally D)
        - Mark the correct answer with the letter (A/B/C/D)
        - Make questions specific to the content, not generic

        **SKILLS EXTRACTION:**
        - Extract 8-12 relevant skills from the course content
        - Include technologies, frameworks, platforms, and concepts
        - List both broad skills (e.g., "Machine Learning") and specific ones (e.g., "XGBoost", "Vertex AI")
        - Skills should reflect what learners will gain from the course

        CRITICAL: All code examples must be real code from discovered repositories with proper attribution.
        """

    def analyze_tech_stack(self, topic: str) -> Dict[str, Any]:
        """Analyze technology stack and complexity for the topic."""
        logger.info(f"Analyzing tech stack for topic: {topic}")

        words = topic.lower().split()

        # Enhanced technology categorization
        tech_categories = {
            "machine_learning": ["ml", "machine", "learning", "ai", "tensorflow", "pytorch", "xgboost", "sklearn", "merlin"],
            "cloud_computing": ["cloud", "aws", "gcp", "azure", "kubernetes", "docker", "serverless"],
            "web_development": ["web", "react", "vue", "angular", "flask", "django", "fastapi", "node"],
            "data_engineering": ["data", "pipeline", "etl", "spark", "airflow", "kafka"],
            "devops": ["devops", "ci", "cd", "jenkins", "github", "actions", "deployment"]
        }

        # Determine primary category
        category = "software_development"  # default
        for cat, keywords in tech_categories.items():
            if any(word in words for word in keywords):
                category = cat
                break

        # Determine complexity based on topic keywords
        complexity_indicators = {
            "advanced": ["production", "scaling", "distributed", "optimization", "mlops", "enterprise"],
            "beginner": ["introduction", "basics", "getting", "started", "tutorial", "hello", "simple"],
            "intermediate": ["deployment", "implementation", "building", "creating"]
        }

        complexity = self.settings.course.default_difficulty.lower()
        for level, indicators in complexity_indicators.items():
            if any(indicator in topic.lower() for indicator in indicators):
                complexity = level.capitalize()
                break

        result = {
            "primary_technology": words[0] if words else "unknown",
            "category": category,
            "complexity": complexity,
            "related_technologies": words[1:],
            "recommended_duration": self.settings.course.default_duration,
            "analysis_timestamp": datetime.now().isoformat()
        }

        logger.info(f"Tech stack analysis complete: {category} - {complexity}")
        return result

    async def discover_sources(self, topic: str) -> Dict[str, Any]:
        """Discover content sources for the topic."""
        logger.info(f"Starting content discovery for: {topic}")

        try:
            discovery_result = await self.source_manager.discover_content(topic)

            # Track discovered sources
            for source_result in discovery_result['rag_results']:
                self.source_tracker.add_source_result(source_result)

            for source_result in discovery_result['github_results']:
                self.source_tracker.add_source_result(source_result)

            for source_result in discovery_result.get('search_results', []):
                self.source_tracker.add_source_result(source_result)

            # Validate source quality
            validation_issues = self.source_tracker.validate_sources()
            if validation_issues:
                logger.warning(f"Source validation issues: {validation_issues}")

            return {
                "total_sources_found": discovery_result['total_results'],
                "sources_used": discovery_result['used_sources'],
                "rag_results_count": len(discovery_result['rag_results']),
                "github_results_count": len(discovery_result['github_results']),
                "search_results_count": len(discovery_result.get('search_results', [])),
                "discovery_strategy": self.settings.source_priority.value,
                "validation_issues": validation_issues
            }

        except Exception as e:
            logger.error(f"Content discovery failed: {e}")
            raise

    async def extract_repository_content(self, repository: str, file_patterns: List[str]) -> Dict[str, str]:
        """Extract specific content from a repository."""
        logger.info(f"Extracting content from repository: {repository}")

        try:
            content = await self.source_manager.get_repository_content(repository, file_patterns)
            logger.info(f"Extracted {len(content)} files from {repository}")
            return content
        except Exception as e:
            logger.error(f"Repository content extraction failed: {e}")
            return {}

    def get_tracked_sources(self) -> List[str]:
        """Get all tracked source URLs and paths."""
        return self.source_tracker.get_source_urls()

    def determine_difficulty(self, topic: str) -> str:
        """Determine course difficulty based on topic analysis."""
        analysis = self.analyze_tech_stack(topic)
        return analysis.get("complexity", self.settings.course.default_difficulty)

    def generate_search_queries(self, topic: str) -> Dict[str, Any]:
        """Generate multiple search queries for better repository discovery."""
        topic_lower = topic.lower()

        # Base queries
        queries = [topic]

        # Component extraction
        components = []

        # ML frameworks
        ml_frameworks = ["lgbm", "lightgbm", "xgboost", "tensorflow", "pytorch", "sklearn"]
        for framework in ml_frameworks:
            if framework in topic_lower:
                components.append(framework)
                queries.append(f"{framework} tutorial")
                queries.append(f"{framework} deployment")

        # Cloud platforms
        cloud_platforms = ["gcp", "google cloud", "aws", "azure"]
        for platform in cloud_platforms:
            if platform in topic_lower or platform.replace(" ", "") in topic_lower:
                components.append(platform)
                queries.append(f"machine learning {platform}")
                queries.append(f"ml deployment {platform}")

        # ML concepts
        ml_concepts = ["deployment", "machine learning", "model", "training"]
        for concept in ml_concepts:
            if concept in topic_lower:
                components.append(concept)

        # Generate combination queries
        if len(components) >= 2:
            queries.append(f"{components[0]} {components[1]}")

        # Add general fallbacks
        queries.extend([
            "machine learning deployment",
            "ml model deployment",
            "mlops tutorial"
        ])

        return {
            "original_topic": topic,
            "search_queries": list(set(queries[:8])),  # Remove duplicates, limit to 8
            "components_found": components,
            "strategy": "multi_query_approach"
        }


    def save_course_to_file(self, course_content: Dict[str, Any], filename: str) -> Dict[str, str]:
        """Save course with enhanced validation and tracking."""
        logger.info(f"Saving course to file: {filename}")

        # Validate required fields
        required_fields = ["title", "description", "modules"]
        missing_fields = [field for field in required_fields if not course_content.get(field)]
        if missing_fields:
            raise ValueError(f"Course content missing required fields: {missing_fields}")

        # Add enhanced tracking information
        course_content["source_tracking"] = self.source_tracker.get_summary()
        course_content["source_from"] = self.get_tracked_sources()

        # Add generation metadata
        course_content["generation_metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "agent_version": "2.0.0",
            "configuration": {
                "source_priority": self.settings.source_priority.value,
                "github_enabled": self.source_manager.github_tool.is_available(),
                "rag_enabled": self.source_manager.rag_tool.is_available()
            }
        }

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)

            # Save course with proper formatting
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(course_content, f, indent=2, ensure_ascii=False)

            logger.info(f"Course saved successfully: {filename}")
            return {"status": "success", "filename": filename, "sources_tracked": len(self.get_tracked_sources())}

        except Exception as e:
            logger.error(f"Failed to save course: {e}")
            raise



    def get_agent(self) -> Agent:
        """Get the configured ADK agent."""
        return self.agent

    def get_configuration_status(self) -> Dict[str, Any]:
        """Get comprehensive configuration and status information."""
        return {
            "agent_name": self.settings.name,
            "source_priority": self.settings.source_priority.value,
            "github_available": self.source_manager.github_tool.is_available(),
            "drive_available": self.drive_tool.is_available() if self.drive_tool else False,
            "rag_available": self.source_manager.rag_tool.is_available(),
            "configuration_issues": self.settings.validate(),
            "max_repositories": self.settings.mcp.max_repositories,
            "max_rag_results": self.settings.rag.max_results,
            "log_level": self.settings.log_level.value
        }


# Create the main agent instance
def create_course_agent(github_token: str = None, drive_token: str = None, user_id: str = None) -> CourseGenerationAgent:
    """
    Factory function to create a configured course generation agent.

    Args:
        github_token: GitHub personal access token (overrides env var)
        drive_token: Google Drive token
        user_id: User ID for Drive credentials management
    """
    agent = CourseGenerationAgent(github_token=github_token, drive_token=drive_token, user_id=user_id)
    logger.info(f"Course generation agent created: {agent.get_configuration_status()}")
    return agent