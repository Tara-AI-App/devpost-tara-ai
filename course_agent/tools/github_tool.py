"""
GitHub MCP tool implementation.
"""
import json
import os
from typing import Dict, Any, List, Optional, Union
from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams

# Import and apply JSON encoder patch
from ..utils.json_encoder import CustomJSONEncoder

# Monkey patch json.dumps to use our custom encoder by default
_original_dumps = json.dumps
def patched_dumps(obj, **kwargs):
    if 'cls' not in kwargs:
        kwargs['cls'] = CustomJSONEncoder
    return _original_dumps(obj, **kwargs)

json.dumps = patched_dumps

from .base import RepositoryTool, SourceResult, SourceType
from ..config.settings import settings
from ..utils.logger import logger

# Temporarily comment out serializable wrapper to debug
# from .serializable_mcp_wrapper import create_serializable_mcp_wrapper




class GitHubMCPTool(RepositoryTool):
    """GitHub MCP tool implementation."""

    def __init__(self):
        self._mcp_tools: Optional[McpToolset] = None
        self._serializable_wrapper = None
        self._initialize_mcp()


    def _initialize_mcp(self):
        """Initialize MCP tools if token is available."""
        logger.info("Starting GitHub MCP initialization...")

        # Check token from settings first
        settings_token = settings.mcp.github_token
        env_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

        logger.info(f"Settings token available: {bool(settings_token)}")
        logger.info(f"Environment token available: {bool(env_token)}")

        if not settings_token and not env_token:
            logger.warning("No GitHub token found in settings or environment - MCP tools disabled")
            return

        try:
            # Use environment variable directly like in the official example
            github_token = env_token or settings_token
            if not github_token:
                logger.warning("GitHub token is None after fallback check")
                return

            logger.info("Creating MCP toolset...")

            connect_timeout = float(os.getenv("MCP_CONNECT_TIMEOUT", "60.0"))
            read_timeout = float(os.getenv("MCP_READ_TIMEOUT", "300.0"))
            logger.info(f"MCP timeout configuration: connect={connect_timeout}s, read={read_timeout}s")
            logger.info("Note: Timeouts are managed at client request level, not httpx level")

            # Use exact pattern from official example to avoid serialization issues
            self._mcp_tools = McpToolset(
                connection_params=StreamableHTTPConnectionParams(
                    url="https://api.githubcopilot.com/mcp/",
                    headers={
                        "Authorization": "Bearer " + github_token,
                    },
                ),
                # Read only tools to match official example pattern
                tool_filter=[
                    "search_repositories",
                    "get_file_contents",
                    "search_code",
                    "get_me",
                    "get_teams"  # Get user's teams/organizations
                ],
            )

            logger.info(f"MCP toolset created: {self._mcp_tools}")
            logger.info(f"MCP toolset type: {type(self._mcp_tools)}")

            # Temporarily disable serializable wrapper for debugging
            self._serializable_wrapper = None
            logger.info("Skipping serializable wrapper creation for debugging")

            logger.info("GitHub MCP tools initialized successfully with serialization wrapper")
        except Exception as e:
            logger.error(f"Failed to initialize GitHub MCP tools: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self._mcp_tools = None
            self._serializable_wrapper = None

    def is_available(self) -> bool:
        """Check if MCP tools are available."""
        return self._mcp_tools is not None

    def get_serializable_toolset(self):
        """Get the serializable MCP toolset for agent integration."""
        return self._serializable_wrapper if self._serializable_wrapper else self._mcp_tools

    async def search_repositories(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for repositories using MCP."""
        if not self.is_available():
            logger.warning("GitHub MCP tools not available")
            return []

        try:
            logger.info(f"Repository search for: {query}")
            # Note: MCP tools are called directly by the agent framework
            # This method serves as a placeholder that will be overridden by the agent
            return []
        except Exception as e:
            logger.error(f"Repository search failed: {e}")
            return []

    async def get_file_contents(self, repository: str, file_path: str) -> str:
        """Get contents of a specific file from repository using MCP."""
        if not self.is_available():
            logger.warning("GitHub MCP tools not available")
            return ""

        try:
            logger.info(f"File contents request for: {repository}/{file_path}")
            # Note: MCP tools are called directly by the agent framework
            # This method serves as a placeholder that will be overridden by the agent
            return ""
        except Exception as e:
            logger.error(f"Get file contents failed: {e}")
            return ""

    async def search_code(self, query: str, repository: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for code patterns using MCP."""
        if not self.is_available():
            logger.warning("GitHub MCP tools not available")
            return []

        try:
            logger.info(f"Code search for: {query}")
            # Note: MCP tools are called directly by the agent framework
            # This method serves as a placeholder that will be overridden by the agent
            return []
        except Exception as e:
            logger.error(f"Code search failed: {e}")
            return []

    async def get_me(self) -> Dict[str, Any]:
        """Get the authenticated user's profile using MCP."""
        if not self.is_available():
            logger.warning("GitHub MCP tools not available")
            return {}

        try:
            logger.info("Fetching authenticated user profile")
            # Note: MCP tools are called directly by the agent framework
            # This method serves as a placeholder that will be overridden by the agent
            return {}
        except Exception as e:
            logger.error(f"Get user profile failed: {e}")
            return {}

    def extract_source_results(self, repositories: List[Dict[str, Any]]) -> List[SourceResult]:
        """Convert repository data to standardized SourceResult format."""
        results = []
        for repo in repositories:
            result = SourceResult(
                content=repo.get('description', ''),
                source_type=SourceType.GITHUB,
                url=repo.get('url', ''),
                repository=repo.get('name', ''),
                metadata={
                    'stars': repo.get('stars', 0),
                    'language': repo.get('language', ''),
                    'updated_at': repo.get('updated_at', ''),
                }
            )
            results.append(result)
        return results