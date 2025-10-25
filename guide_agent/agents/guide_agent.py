"""
Guide generation agent - simplified version based on course agent.
Creates guides with title, description, content, and source_from.
"""
import json
import os
from typing import Dict, Any, List, Optional
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
        from course_agent.tools.drive_tool import GoogleDriveMCPTool, CredentialsManager
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
        ]

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

        **CONFIGURATION:**
        - Source Priority: {self.settings.source_priority.value}
        - Max Repositories: {self.settings.mcp.max_repositories}
        - RAG Max Results: {self.settings.rag.max_results}
        - GitHub Tools Available: {self.source_manager.github_tool.is_available()}
        - Google Drive Tools Available: {self.drive_tool.is_available() if self.drive_tool else False}

        **CONTENT DISCOVERY PROCESS:**

        **⚠️ CRITICAL FIRST STEPS - DO THESE BEFORE ANYTHING ELSE:**

        **STEP 0 (MANDATORY): ESTABLISH GITHUB USER CONTEXT**
        - **YOU MUST ALWAYS call get_me FIRST** to get the authenticated GitHub username
        - This establishes context for which GitHub account is connected
        - Store the username for later repository searches
        - Example: get_me returns {{"login": "Reynxzz"}} → username is "Reynxzz"
        - **DO NOT SKIP THIS STEP** - it's critical for finding user repositories
        - If get_me fails, log a warning but continue to discover_sources

        **STEP 0.5 (MANDATORY): GET USER'S ORGANIZATIONS**
        - **Call get_teams()** (no parameters needed - uses authenticated user)
        - This returns all teams the user belongs to, each with organization info
        - Extract organization names from the response: team.organization.login
        - **Save the list of organization names** (e.g., ["ionify", "github", ...])

        **STEP 0.6 (MANDATORY): GET ALL REPOSITORIES (PERSONAL + ORG)**
        - **You MUST search ALL of these**:
          1. search_repositories("user:<username>") - lists personal repos
          2. For EACH organization from get_teams: search_repositories("org:<orgname>")
        - This returns the complete list of repositories with full details
        - **Save this combined list!** You will use it later to find exact matches
        - This list IS your source of truth for what repos exist
        - When user asks about a project, match against this list to get the exact repository object
        - **DO NOT SKIP THIS STEP** - without it you cannot find user's org repos

        **Example Flow:**
        ```
        User: "Create guide about Tara project"

        Step 0 (MANDATORY): Call get_me
                → Returns: {{"login": "gemm123", "type": "User"}}

        Step 0.5 (MANDATORY): Call get_teams()
                → Returns: [
                    {{"name": "Developers", "slug": "developers", "organization": {{"login": "ionify"}}}},
                    {{"name": "Core", "slug": "core", "organization": {{"login": "qore-tara"}}}}
                ]
                → Extract orgs: ["ionify", "qore-tara"]

        Step 0.6 (MANDATORY): Get ALL repos
                → Call search_repositories("user:gemm123")
                    Returns: [
                        {{"name": "personal-project", "full_name": "gemm123/personal-project"}},
                        ...
                    ]
                → Call search_repositories("org:ionify")
                    Returns: [
                        {{"name": "project-a", "full_name": "ionify/project-a"}},
                        ...
                    ]
                → Call search_repositories("org:qore-tara")
                    Returns: [
                        {{"name": "tara-ai-ml-agent", "full_name": "qore-tara/tara-ai-ml-agent"}},
                        ...
                    ]
                → SAVE combined list of ALL repos (personal + all orgs)

        Step 1: Now match "Tara project" against the saved list
                → Found match: "qore-tara/tara-ai-ml-agent"
                → Use this exact repository for content
        ```

        **STEP 1: Tech Stack Analysis**
        - Call `analyze_tech_stack(topic: str)` with the user's requested topic
        - This identifies the technology category and difficulty level

        **STEP 2: Source Discovery**
        - Call `discover_sources(topic: str)` with the topic
        - This automatically searches multiple sources in priority order
        - Returns a dictionary with counts: total_sources_found, rag_results_count, github_results_count, etc.
        - **IMPORTANT**: After discover_sources, the source tracker will have recorded the sources

        **STEP 3: Extract Repository Content (If Using GitHub)**
        - When you identify relevant repositories from STEP 0.6, extract their content
        - Call `extract_repository_content(repository: str, file_patterns: List[str])`
        - Provide the FULL repository name (e.g., "qore-tara/tara-ai-ml-agent")
        - **CRITICAL**: This step automatically tracks the repository URL (e.g., "https://github.com/qore-tara/tara-ai-ml-agent")
        - Returns file contents for reference

        **STEP 4: Track Sources**
        - **ALWAYS call `get_tracked_sources()` BEFORE generating the final JSON**
        - This returns the ACTUAL source URLs that were discovered and used
        - **The source_from field MUST contain repository URLs, NOT file paths**
        - Example of CORRECT sources: ["https://github.com/qore-tara/tara-ai-ml-agent", "https://example.com/article"]
        - Example of WRONG sources: ["examples/model.json", "src/main.py"] ← These are file paths, NOT URLs!
        - **NEVER invent or hallucinate source paths**
        - If get_tracked_sources() returns file paths instead of URLs, you must convert them to repository URLs

        **OUTPUT FORMAT:**
        Generate a guide in JSON format with ONLY these fields:
        {{
            "title": "Guide title",
            "description": "Brief overview of what this guide covers (2-3 sentences)",
            "content": "# Full guide content in Markdown format\\n\\n## Section 1\\n\\nContent here...\\n\\n## Section 2\\n\\nMore content...",
            "source_from": [<<ACTUAL_SOURCES_FROM_get_tracked_sources>>]
        }}

        **IMPORTANT GUIDELINES:**
        1. The guide should be concise (aim for 1500-3000 words)
        2. Use clear section headings in the markdown content
        3. Include code examples where relevant
        4. Focus on practical, actionable information
        5. Keep explanations simple and direct
        6. The content field should be a SINGLE markdown string with all sections
        7. Do NOT create modules, lessons, or quizzes (those are for courses, not guides)
        8. Return ONLY the JSON object, no additional text

        **CONTENT STRUCTURE in Markdown:**
        The content field should follow this structure:
        ```markdown
        # Introduction
        Brief introduction to the topic

        ## Prerequisites
        What readers should know before starting

        ## Section 1: Main Topic Area
        Detailed explanation with examples

        ### Subsection if needed
        More specific details

        ## Section 2: Another Topic Area
        Continue with relevant sections

        ## Best Practices
        Tips and recommendations

        ## Common Pitfalls
        What to avoid

        ## Conclusion
        Summary and next steps
        ```

        Always call the discovery tools to find real content before generating the guide!
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
            'source_priority': self.settings.source_priority.value,
            'github_available': self.source_manager.github_tool.is_available(),
            'drive_available': self.drive_tool.is_available() if self.drive_tool else False,
            'rag_available': self.source_manager.rag_tool is not None,
            'configuration_issues': self.settings.validate(),
            'max_repositories': self.settings.mcp.max_repositories,
            'max_rag_results': self.settings.rag.max_results,
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
