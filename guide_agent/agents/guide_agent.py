"""
Guide generation agent - simplified version based on course agent.
Creates guides with title, description, content, and source_from.
"""
import json
import os
from typing import Dict, Any, List
from datetime import datetime
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.genai import types

# Apply JSON encoder patch early
from course_agent.utils.json_encoder import CustomJSONEncoder

# Monkey patch json.dumps globally
_original_dumps = json.dumps
def patched_dumps(obj, **kwargs):
    if 'cls' not in kwargs:
        kwargs['cls'] = CustomJSONEncoder
    return _original_dumps(obj, **kwargs)

json.dumps = patched_dumps

# Import from course_agent
from course_agent.config.settings import settings
from course_agent.utils.logger import logger
from course_agent.core.source_manager import SourceManager
from course_agent.core.enhanced_source_tracker import EnhancedSourceTracker


class GuideGenerationAgent:
    """Guide generation agent with same tools as course agent but simpler output."""

    def __init__(self, github_token: str = None, drive_token: str = None, user_id: str = None):
        """
        Initialize the guide generation agent.

        Args:
            github_token: GitHub personal access token
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

        # Initialize Roadmap tool for learning path structure
        from course_agent.tools.roadmap_tool import RoadmapTool
        self.roadmap_tool = RoadmapTool()
        if self.roadmap_tool.is_available():
            logger.info(f"Roadmap tool initialized successfully")
        else:
            logger.warning("Roadmap tool not available")

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
        from course_agent.tools.drive_tool import GoogleDriveMCPTool
        import os

        try:
            logger.info(f"Initializing Google Drive MCP tool for user {self.user_id}")

            # Initialize Drive tool with access token (no file storage!)
            self.drive_tool = GoogleDriveMCPTool(
                user_id=self.user_id,
                access_token=self.drive_token
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
            FunctionTool(self.extract_drive_content),
            FunctionTool(self.get_tracked_sources),
            FunctionTool(self.determine_difficulty),
            FunctionTool(self.generate_search_queries),
        ]

        # Add Roadmap tools for learning structure
        if self.roadmap_tool and self.roadmap_tool.is_available():
            tools.extend([
                FunctionTool(self.roadmap_tool.query_learning_structure),
                FunctionTool(self.roadmap_tool.get_roadmap_structure),
                FunctionTool(self.roadmap_tool.find_learning_resources),
                FunctionTool(self.roadmap_tool.list_available_roadmaps),
                FunctionTool(self.roadmap_tool.search_roadmap_topics)
            ])
            logger.info("Added 5 roadmap tools to guide agent")

        # Add GitHub MCP tools if available
        logger.info(f"Checking if GitHub MCP tools are available...")
        github_available = self.source_manager.github_tool.is_available()
        logger.info(f"GitHub tool is_available(): {github_available}")

        if github_available:
            mcp_toolset = self.source_manager.github_tool._mcp_tools
            logger.info(f"Retrieved MCP toolset: {mcp_toolset}")
            logger.info(f"MCP toolset type: {type(mcp_toolset)}")

            if mcp_toolset:
                tools.append(mcp_toolset)
                logger.info("GitHub MCP toolset added to guide agent")
                logger.info(f"Total tools count: {len(tools)}")
            else:
                logger.warning("GitHub MCP toolset is None")
        else:
            logger.warning("GitHub MCP tools not available")

        # Note: Drive tools are called directly via extract_drive_content() method
        # We don't add them as ADK tools since we use subprocess to call the MCP server
        logger.info(f"Checking if Google Drive MCP tools are available...")
        drive_available = self.drive_tool and self.drive_tool.is_available()
        logger.info(f"Drive tool is_available(): {drive_available}")

        if drive_available:
            logger.info("Google Drive tools are available (via direct MCP server calls)")
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
            name="guide_generator",
            description="Technical guide generator with dynamic source discovery",
            instruction=self._get_agent_instruction(),
            tools=tools,
            generate_content_config=generation_config
        )

    def _get_agent_instruction(self) -> str:
        """Get comprehensive agent instruction for guide generation."""
        return f"""
        You are an expert guide generator that creates concise technical guides using dynamic source discovery.

        **⚠️ CRITICAL FIRST STEPS - DO THESE BEFORE ANYTHING ELSE:**

        1. **ALWAYS call get_me() first** → get GitHub username
        2. **ALWAYS call get_teams() next** → get user's organizations list
        3. **ALWAYS call search_repositories("user:<username>")** → get personal repos
        4. **For each org: call search_repositories("org:<orgname>")** → get org repos
        5. **SAVE the combined repo list** → you will match against this list later

        Example:
        - get_me() → username="gemm123"
        - get_teams() → orgs=["ionify"]
        - search_repositories("user:gemm123") → [personal repos]
        - search_repositories("org:ionify") → [org repos including "tara"]
        - When user asks about "Tara", match it in your saved list!

        **CONFIGURATION:**
        - Max Repositories: {self.settings.mcp.max_repositories}
        - GitHub Tools Available: {self.source_manager.github_tool.is_available()}
        - Google Drive Tools Available: {self.drive_tool.is_available() if self.drive_tool else False}
        - Roadmap Structure Tools Available: {self.roadmap_tool.is_available() if self.roadmap_tool else False}

        **GUIDE GENERATION WORKFLOW:**

        **STEP 1: USE ROADMAP.SH STRUCTURE (ALWAYS START HERE)**
        - **Call list_available_roadmaps()** to see available learning paths (59 roadmaps available)
        - **Call search_roadmap_topics(keyword, roadmap)** to find relevant topics for the guide subject
        - **Call get_roadmap_structure(roadmap_id)** to get proper learning order for topics
        - Use roadmap structure to organize guide sections in industry-standard order
        - **Roadmaps have 7,800+ topics** - you will ALWAYS find relevant structure

        **STEP 2: ESTABLISH GITHUB USER CONTEXT**
        - **Call get_me()** to get the authenticated GitHub username
        - This establishes context for which GitHub account is connected
        - Store the username for later repository searches
        - Example: get_me returns {{"login": "Reynxzz"}} → username is "Reynxzz"

        **STEP 3: GET USER'S ORGANIZATIONS**
        - **Call get_teams()** (no parameters needed - uses authenticated user)
        - This returns all teams the user belongs to, each with organization info
        - Extract organization names from the response: team.organization.login
        - **Save the list of organization names** (e.g., ["ionify", "github", ...])

        **STEP 4: GET ALL REPOSITORIES (PERSONAL + ORG)**
        - **Search ALL of these**:
          1. search_repositories("user:<username>") - lists personal repos
          2. For EACH organization from get_teams: search_repositories("org:<orgname>")
        - This returns the complete list of repositories with full details
        - **Save this combined list!** You will use it later to find exact matches
        - This list IS your source of truth for what repos exist

        **STEP 5: EXTRACT FILES FROM REPOSITORIES**
        - **If user asked about a specific project**: Match project name against your saved repo list
        - **Extract files** with get_file_contents(repository="owner/repo-name", file_path="...")
        - Extract README, code files, configuration files as needed

        **STEP 6 (OPTIONAL): EXTRACT DRIVE CONTENT**
        - **If Drive is available**: Call extract_drive_content(search_query)
        - Use specific queries for better results (e.g., "TARA prototype")

        **STEP 7: GENERATE GUIDE**
        - Combine: Roadmap structure + GitHub code examples + Drive documentation
        - Call get_tracked_sources() to get actual source URLs
        - Generate guide JSON with proper attribution

        **Example Flow:**
        ```
        User: "Generate guide about Tara project"

        Step 1: search_roadmap_topics("backend") → Find relevant roadmap
        Step 2: get_me() → Returns: {{"login": "gemm123"}}
        Step 3: get_teams() → Returns: [{{"organization": {{"login": "ionify"}}}}]
        Step 4a: search_repositories("user:gemm123") → [personal repos]
        Step 4b: search_repositories("org:ionify") → [{{"name": "tara", "full_name": "ionify/tara", ...}}]
        Step 5: Match "tara" in saved list → Found!
        Step 6: get_file_contents(repository="ionify/tara", file_path="README.md")
        Step 7: extract_drive_content("TARA") → Get internal docs
        Step 8: get_roadmap_structure("backend") → Get proper topic order
        Step 9: Generate guide with all sources
        ```

        **WORK FAST - NO DELAYS:**
        - Roadmap queries are instant (~1ms)
        - GitHub MCP tools are fast (direct API calls)
        - No need to search for content - roadmaps provide all structure needed
        - Just find user's repos → extract files → generate guide

        **ROADMAP TOOLS (FOR GUIDE STRUCTURE):**
        - query_learning_structure: Get prerequisites and next topics for a subject
        - get_roadmap_structure: Get proper learning order for a roadmap
        - find_learning_resources: Find curated resources from roadmap.sh
        - list_available_roadmaps: See available roadmaps (frontend, python, react, etc.)
        - search_roadmap_topics: Search for topics by keyword

        Use these tools to structure guides based on industry-standard learning paths!

        **GITHUB MCP TOOLS (if available):**
        GitHub integration may provide these tools when connected:
        - search_repositories: Find repositories by name, description, topics, readme
        - search_code: Search for specific code patterns across GitHub
        - get_file_contents: Extract actual files from repositories
        - get_me: Get the authenticated GitHub user's profile (optional)

        Note: Use only GitHub tools that are actually available. Don't assume all are present.

        **GOOGLE DRIVE INTEGRATION:**

        **Available Tool:**

        **extract_drive_content(search_query: str)** - Search and extract Drive file content
           * Input: search_query (string) - Search query for Drive files
           * Returns: Dict with file contents, metadata, and matched files
           * Example: extract_drive_content("TARA prototype")
           * **Automatically searches Drive and extracts content in one call**
           * Returns structured data with all matched files and their contents

        **How It Works:**
        The extract_drive_content tool automatically:
        1. Searches Google Drive for files matching your query
        2. Extracts content from up to 5 matching files
        3. Returns structured data with file names, content, and metadata

        **Usage Example:**
        ```
        User: "Generate guide about TARA prototype development"

        Call: extract_drive_content("TARA prototype")

        Returns:
        {{
          "files_found": 5,
          "content": [
            {{"name": "Rencana Pengembangan Prototype - TARA", "content": "...", "length": 5234}},
            {{"name": "Ionify - Tara BI Hackaton", "content": "...", "length": 12345}}
          ],
          "source_urls": [
            "https://drive.google.com/file/d/11E-BABqB4XscV7-9oZVg3MKqjjo_oPcGjxsxrTrfn9Q/view",
            "https://drive.google.com/file/d/15MwrpzIgLWDZxhNMcn-U5NEtzkQWHdQ-KfFbmyvQiqs/view"
          ],
          "matched_files": ["File 1", "File 2", "File 3"],
          "summary": "Successfully extracted content from 5 of 5 matched files"
        }}

        → Use the URLs from source_urls in your source_from array
        ```

        **Complete Workflow for Guide Generation with Drive:**

        ```
        User: "Generate guide about TARA prototype development"

        SEQUENCE:

        1. search_roadmap_topics("backend") → Get relevant roadmap structure

        2. Get GitHub context:
           - get_me() → username
           - get_teams() → organizations
           - search_repositories() → find TARA repo

        3. extract_drive_content("TARA prototype")
           → Searches Drive and extracts file contents automatically
           → Returns full content from matched files with metadata

        4. Generate guide using ALL sources:
           - Roadmap structure (industry-standard learning path)
           - GitHub code examples (from TARA repository)
           - Drive document content (internal documentation)

        5. Add to source_from array (use Drive links from source_urls):
           ["github.com/ionify/tara",
            "https://drive.google.com/file/d/11E-BABqB4XscV7-9oZVg3MKqjjo_oPcGjxsxrTrfn9Q/view",
            "https://drive.google.com/file/d/15MwrpzIgLWDZxhNMcn-U5NEtzkQWHdQ-KfFbmyvQiqs/view"]
        ```

        **Rules:**
        1. Call extract_drive_content() when you need Drive files for a topic
        2. Use specific queries for better results (e.g., "TARA prototype" vs "TARA")
        3. The tool returns source_urls with clickable Drive links
        4. File contents are automatically extracted and ready to use
        5. **IMPORTANT**: Add Drive links from source_urls to source_from array (not file names)

        **Drive Content Types (Automatically Converted):**
        - Google Docs → Markdown (use directly in guide)
        - Google Sheets → CSV (extract data/examples)
        - PDFs → Text (get specifications/proposals)
        - DOCX → Text (read documentation)

        **⚠️ KEY PRINCIPLES (MUST FOLLOW) ⚠️:**
        1. **USE**: Call extract_drive_content() to search and get Drive files in one step
        2. **SPECIFIC**: Use specific queries for better results (e.g., "TARA prototype" not just "TARA")
        3. **SILENT**: Don't announce files to user - use content automatically
        4. **ATTRIBUTION**: Add Drive links from source_urls to source_from array

        **CRITICAL - SOURCE VALIDATION (PREVENT HALLUCINATION):**

        **How to handle different query types**:

        1. **User's Internal Projects** (e.g., "kredipo", "tara project", "analytics in kredipo"):
           - STEP 1: Call search_roadmap_topics() to find relevant roadmap (e.g., "data-analyst", "machine-learning")
           - STEP 2: Call get_me() to get authenticated username
           - STEP 3: Call get_teams() to get user's organizations
           - STEP 4: Call search_repositories("user:<username>") to list all personal repos
           - STEP 5: Call search_repositories("org:<orgname>") for each organization
           - STEP 6: Match project name against the combined repo list
           - STEP 7: Extract files with get_file_contents()
           - STEP 8: Combine roadmap structure + internal code examples

        2. **General Tech Topics** (e.g., "React", "Python", "Machine Learning"):
           - STEP 1: Call search_roadmap_topics() to find relevant topics
           - STEP 2: Call get_roadmap_structure() to get proper learning order
           - STEP 3: Optionally: search user's GitHub for related code examples
           - STEP 4: Combine roadmap structure with user's code examples if available

        **ABSOLUTE RULES TO PREVENT HALLUCINATION**:
        - ✅ ONLY use sources you actually retrieved (repos found, files extracted)
        - ✅ ONLY reference files you extracted with get_file_contents()
        - ❌ NEVER create fake source paths or file references
        - ❌ NEVER assume repo exists without searching for it first

        **CRITICAL - CODE EXTRACTION FROM REPOSITORIES:**

        **FILE EXTRACTION PROCESS:**

        1. **After finding a GitHub repository**, extract code files:
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

        3. **To include code examples in the guide**:
           - MUST call get_file_contents(repository, file_path) to extract the code
           - ONLY reference files that get_file_contents successfully returned
           - If get_file_contents returns empty or fails: DO NOT reference that file

        4. **Valid references**:
           - ✅ "From: https://github.com/Reynxzz/graphflix"
           - ✅ "Based on the Reynxzz/graphflix repository"
           - ❌ "From: https://github.com/Reynxzz/graphflix/algorithms/content_recommend.js" (unless you extracted it)
           - ❌ NEVER make up file paths you didn't extract

        **MATCHING USER QUERIES TO REPOSITORIES:**

        **When user mentions a specific project name** (like "graphflix", "thinktok", "tara", "kredipo"):

        1. **Extract keywords from user query**:
           - "bytesv2 project" → keywords: ["bytesv2"]
           - "capstone seis flask" → keywords: ["capstone", "seis", "flask"]
           - "help me learn about graphflix" → keywords: ["graphflix"]
           - "analytics side of credit scoring project in kredipo" → keywords: ["kredipo", "credit", "scoring"]

        2. **Search your saved repo list** from STEP 4:
           - Exact name match: repo.name == "bytesv2" → PERFECT MATCH
           - Contains match: "capstone-seis-flask" contains ["capstone", "seis", "flask"] → MATCH
           - Partial match: "kredipo" in description or readme → POSSIBLE MATCH

        3. **Use the matched repository**:
           - You have: name, full_name, url, description from STEP 4
           - Extract files: get_file_contents(repository=full_name, file_path="README.md")
           - Use actual file content in guide generation

        **ORGANIZATION REPOSITORIES:**
        - get_me() does NOT return organization information!
        - **Use get_teams()** to discover user's organizations
        - get_teams() returns: [{{"organization": {{"login": "ionify"}}}}, ...]
        - Extract org names and search each: search_repositories("org:ionify")
        - Example: User in "ionify" org:
          * get_teams() → [{{"organization": {{"login": "ionify"}}}}]
          * search_repositories("user:gemm123") → personal repos
          * search_repositories("org:ionify") → org repos (tara, etc.)
          * Combine both lists and match against user's query

        **WHEN TO GENERATE A GUIDE (CRITICAL)**:
        You MUST generate a guide if ANY of these conditions are met:
        - Roadmap tools found relevant structure (search_roadmap_topics returned results)
        - You successfully found GitHub repositories (via search_repositories)
        - You extracted files from repositories (via get_file_contents)

        **GUIDE STRUCTURE STRATEGY**:
        1. **Use roadmap.sh for structure**: Call get_roadmap_structure() to get industry-standard topic order
        2. **Use GitHub for examples**: Extract code from user's repositories for concrete examples
        3. **Use Drive for context**: Get internal documentation if available (via extract_drive_content)
        4. **Combine all three**: Roadmap structure + internal code examples + documentation = best guide

        **ALWAYS GENERATE A GUIDE**:
        - If roadmap structure found: Generate guide using roadmap structure + code examples (if any)
        - If only GitHub found: Generate guide based on code structure + related roadmap
        - If only roadmap found: Generate guide using roadmap structure + official resources
        - Roadmaps have 7,800+ topics across 59 domains - you will ALWAYS find relevant structure

        **WORK FAST AND EFFICIENTLY**:
        - ALWAYS start with roadmap tools (instant ~1ms queries)
        - Get user context early (get_me + get_teams + search_repositories)
        - Extract files with get_file_contents for code examples
        - Combine roadmap structure with internal code examples for best results

        **CONTENT LENGTH GUIDELINES (GUIDES ARE SHORTER THAN COURSES):**
        - Keep guide content concise and focused (aim for 1500-3000 words total)
        - Include 1-2 key code examples (not full file dumps)
        - Use code snippets (10-30 lines) rather than entire files
        - Focus on the most important/illustrative sections
        - If showing API responses, use shortened examples (3-5 items, not full responses)
        - Remember: Guides are shorter and more direct than courses

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
        5. Numbers must be plain numbers without quotes
        6. Booleans must be true/false (lowercase, no quotes)
        7. Arrays must use square brackets: []
        8. Objects must use curly braces: {{}}
        9. Ensure ALL brackets and braces are properly closed
        10. Test your JSON is valid before returning it

        **CRITICAL - BEFORE GENERATING GUIDE:**
        1. Review what you ACTUALLY found:
           - What roadmap structure did you retrieve?
           - What repositories did you find via search_repositories?
           - What files did get_file_contents return?
           - What URLs are in get_tracked_sources?
        2. ONLY use information from those actual results
        3. DO NOT invent:
           - File paths you didn't extract
           - Code you didn't retrieve
           - Source URLs not in get_tracked_sources
        4. If you have limited information:
           - That's OK! Create guide with what you have
           - Use roadmap structure as the foundation
           - Reference repository generally (not specific fake files)
           - Explain concepts based on repository description/README

        Generate a concise guide in this EXACT JSON format:
        {{
            "title": "Descriptive Guide Title",
            "description": "Brief overview based on discovered content (2-3 sentences)",
            "content": "# Guide Title\\n\\n## Section 1\\n\\nBased on the [repository name] repository...\\n\\n**Key Concepts**: Explain concepts here without making up file paths.\\n\\nIf you extracted code with get_file_contents, THEN include:\\n```language\\n// Actual code you extracted\\nreal_code_here()\\n```\\n\\nOtherwise, explain concepts generally without fake file references.\\n\\n## Section 2\\n\\nContinue with more sections...\\n\\n## Conclusion\\n\\nSummary and next steps.",
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
        4. Generate guide JSON with ACTUAL sources only

        **GUIDE vs COURSE:**
        - Guides are SHORTER (1500-3000 words) and more DIRECT
        - Guides have a single "content" field with markdown
        - Guides do NOT have modules, lessons, or quizzes
        - Guides focus on practical "how-to" information
        - Guides are quicker to consume than full courses

        CRITICAL: All code examples must be real code from discovered repositories with proper attribution.
        """

    # Tool methods (same as course agent)
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

        # Determine complexity
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
                "github_results_count": len(discovery_result['github_results']),
                "search_results_count": len(discovery_result.get('search_results', [])),
                "validation_issues": validation_issues
            }

        except Exception as e:
            logger.error(f"Content discovery failed: {e}")
            raise

    async def extract_repository_content(self, repository: str, file_patterns: List[str]) -> Dict[str, str]:
        """Extract specific content from a repository and track the source."""
        logger.info(f"Extracting content from repository: {repository}")

        try:
            content = await self.source_manager.get_repository_content(repository, file_patterns)
            logger.info(f"Extracted {len(content)} files from {repository}")

            # Track the GitHub repository as a source (with full URL)
            if content:
                repo_url = f"https://github.com/{repository}"
                # Combine all content for tracking
                combined_content = "\n".join(content.values())
                self.source_tracker.add_mcp_source(
                    content=combined_content,
                    repository=repository,
                    url=repo_url
                )
                logger.info(f"Tracked GitHub source: {repo_url}")

            return content
        except Exception as e:
            logger.error(f"Repository content extraction failed: {e}")
            return {}

    async def extract_drive_content(self, search_query: str) -> Dict[str, Any]:
        """
        Search and extract content from Google Drive files.

        This tool automatically:
        1. Uses MCP search tool to find relevant files
        2. Lists all Drive resources to get file URIs
        3. Matches search results with resources
        4. Retrieves the file content via MCP resources
        5. Returns formatted content ready for guide generation

        Args:
            search_query: Keywords to search for in Drive (e.g., "TARA prototype", "onboarding guide")

        Returns:
            Dict containing:
            - files_found: Number of matching files with extracted content
            - content: List of dicts with 'name', 'content', 'type' for each file
            - source_urls: List of Drive view links for source_from tracking
            - search_results: Original search results for reference

        Example:
            result = await extract_drive_content("TARA prototype")
            # Returns content from "Rencana Pengembangan Prototype - TARA" document
        """
        logger.info(f"🔍 Extracting Drive content for query: {search_query}")

        if not self.drive_tool or not self.drive_tool.is_available():
            logger.warning("Drive tool not available")
            return {
                "files_found": 0,
                "content": [],
                "source_urls": [],
                "error": "Google Drive integration not available"
            }

        try:
            # New workflow: Use search tool directly (searches file names and content)
            logger.info(f"Step 1: Searching Drive for '{search_query}'...")
            matched_files = await self.drive_tool.search_files(search_query)

            if not matched_files:
                logger.warning(f"No Drive files found matching '{search_query}'")
                return {
                    "files_found": 0,
                    "content": [],
                    "source_urls": [],
                    "message": f"No files found matching '{search_query}'"
                }

            logger.info(f"✅ Found {len(matched_files)} files matching query")

            # Step 2: Extract content from matched files
            logger.info("Step 2: Extracting content from matched files...")
            extracted_content = []
            source_urls = []

            for matched_file in matched_files[:5]:  # Limit to 5 files
                try:
                    uri = matched_file.get("uri")
                    name = matched_file.get("name")
                    mime_type = matched_file.get("mimeType")

                    if not uri:
                        logger.warning(f"⚠️ No URI for file: {name}")
                        continue

                    logger.info(f"📄 Reading: {name}")

                    # Get file content using new get_file method
                    file_data = await self.drive_tool.get_file(uri)

                    if file_data and file_data.get("content"):
                        content_text = file_data.get("content", "")
                        extracted_content.append({
                            "name": name,
                            "content": content_text,
                            "type": mime_type,
                            "uri": uri,
                            "length": len(content_text)
                        })

                        # Convert gdrive:///fileId to Google Drive view link
                        # URI format: gdrive:///15MwrpzIgLWDZxhNMcn-U5NEtzkQWHdQ-KfFbmyvQiqs
                        file_id = uri.replace("gdrive:///", "")
                        drive_view_link = f"https://drive.google.com/file/d/{file_id}/view"
                        source_urls.append(drive_view_link)

                        logger.info(f"✅ Extracted {len(content_text)} chars from: {name}")
                        logger.info(f"📎 Drive link: {drive_view_link}")
                    else:
                        logger.warning(f"⚠️ No content extracted from: {name}")

                except Exception as e:
                    logger.warning(f"❌ Failed to read {matched_file.get('name', 'unknown')}: {e}")
                    continue

            # Step 3: Return results
            result = {
                "files_found": len(extracted_content),
                "content": extracted_content,
                "source_urls": source_urls,
                "matched_files": [f.get("name", "unknown") for f in matched_files],
                "total_matched": len(matched_files),
                "summary": f"Successfully extracted content from {len(extracted_content)} of {len(matched_files)} matched files"
            }

            logger.info(f"🎉 {result['summary']}")
            return result

        except Exception as e:
            logger.error(f"❌ Drive content extraction failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "files_found": 0,
                "content": [],
                "source_urls": [],
                "error": str(e),
                "fallback": f"Failed to extract Drive content for query '{search_query}'"
            }

    def get_tracked_sources(self) -> List[str]:
        """Get all tracked source URLs and paths."""
        return self.source_tracker.get_source_urls()

    def determine_difficulty(self, topic: str) -> str:
        """Determine guide difficulty based on topic analysis."""
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
                queries.append(f"{framework} guide")

        # Cloud platforms
        cloud_platforms = ["gcp", "google cloud", "aws", "azure"]
        for platform in cloud_platforms:
            if platform in topic_lower or platform.replace(" ", "") in topic_lower:
                components.append(platform)
                queries.append(f"guide {platform}")

        # Web frameworks
        web_frameworks = ["fastapi", "flask", "django", "react", "vue", "angular", "node"]
        for framework in web_frameworks:
            if framework in topic_lower:
                components.append(framework)
                queries.append(f"{framework} tutorial")
                queries.append(f"{framework} getting started")

        # Generate combination queries
        if len(components) >= 2:
            queries.append(f"{components[0]} {components[1]}")

        # Add general fallbacks
        queries.extend([
            f"{topic} tutorial",
            f"{topic} getting started",
            f"{topic} guide"
        ])

        logger.info(f"Generated {len(queries)} search queries for: {topic}")

        return {
            "queries": list(set(queries)),  # Remove duplicates
            "primary_components": components,
            "query_count": len(queries)
        }

    def get_agent(self) -> Agent:
        """Get the underlying ADK agent."""
        return self.agent

    def get_configuration_status(self) -> Dict[str, Any]:
        """Get current agent configuration and status."""
        return {
            'agent_name': 'guide_generator',
            'github_available': self.source_manager.github_tool.is_available(),
            'drive_available': self.drive_tool.is_available() if self.drive_tool else False,
            'roadmap_available': self.roadmap_tool.is_available() if self.roadmap_tool else False,
            'configuration_issues': self.settings.validate(),
            'max_repositories': self.settings.mcp.max_repositories,
            'log_level': self.settings.log_level.value
        }


def create_guide_agent(github_token: str = None, drive_token: str = None, user_id: str = None) -> GuideGenerationAgent:
    """
    Factory function to create a guide generation agent.

    Args:
        github_token: GitHub personal access token (optional)
        drive_token: Google Drive OAuth token (optional)
        user_id: User ID for Drive credentials (optional)

    Returns:
        GuideGenerationAgent instance
    """
    agent = GuideGenerationAgent(
        github_token=github_token,
        drive_token=drive_token,
        user_id=user_id
    )

    logger.info(f"Guide generation agent created: {agent.get_configuration_status()}")
    return agent
