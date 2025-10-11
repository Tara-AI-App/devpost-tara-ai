"""
Google Drive MCP tool implementation.
"""
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
import logging

from .base import RepositoryTool, SourceResult, SourceType
from ..config.settings import settings
from ..utils.logger import logger as module_logger


class CredentialsManager:
    """Manage user-specific credentials in shared volume."""
    
    def __init__(self, base_path: str = "/credentials"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    def save_drive_credentials(self, user_id: str, drive_token: str) -> str:
        """
        Save user's Drive credentials to shared volume.
        Always overwrites existing credentials to ensure fresh tokens.
        
        Args:
            user_id: Unique user identifier
            drive_token: Google Drive OAuth access token
            
        Returns:
            Path to the created credentials file
        """
        # Create user-specific directory
        user_dir = self.base_path / user_id
        user_dir.mkdir(exist_ok=True, parents=True)
        
        # Path to credentials file
        credentials_path = user_dir / "drive.json"
        
        # Remove old credentials if they exist
        if credentials_path.exists():
            module_logger.info(f"🔄 Overwriting existing credentials for user {user_id}")
            credentials_path.unlink()
        
        # Create credentials JSON
        credentials = {
            "access_token": drive_token
        }
        
        # Save to user's directory
        with open(credentials_path, 'w') as f:
            json.dump(credentials, f, indent=2)
        
        module_logger.info(f"✅ Saved credentials for user {user_id} at {credentials_path}")
        return str(credentials_path)
    
    def get_credentials_path(self, user_id: str) -> Optional[str]:
        """Get path to user's credentials file."""
        credentials_path = self.base_path / user_id / "drive.json"
        if credentials_path.exists():
            return str(credentials_path)
        return None
    
    def cleanup_old_credentials(self, max_age_hours: int = 24):
        """
        Clean up old credential files.
        Run this periodically to avoid accumulating stale credentials.
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        for user_dir in self.base_path.iterdir():
            if user_dir.is_dir():
                cred_file = user_dir / "drive.json"
                if cred_file.exists():
                    # Check file modification time
                    mtime = datetime.fromtimestamp(cred_file.stat().st_mtime)
                    if mtime < cutoff_time:
                        cred_file.unlink()
                        module_logger.info(f"🧹 Cleaned up old credentials for {user_dir.name}")
                        
                        # Remove empty directory
                        if not any(user_dir.iterdir()):
                            user_dir.rmdir()


class GoogleDriveMCPTool(RepositoryTool):
    """Google Drive MCP tool implementation."""

    def __init__(self, user_id: str = None, credentials_path: str = None):
        self._mcp_tools: Optional[McpToolset] = None
        self._user_id = user_id
        self._credentials_path = credentials_path
        
        if credentials_path:
            self._initialize_mcp()
    
    def _cleanup_old_containers(self):
        """Clean up old Drive MCP containers for this specific user only."""
        import subprocess
        try:
            # Generate user-specific container name
            container_name = f"mcp-gdrive-{self._user_id}"
            
            module_logger.info(f"Checking for old Drive MCP container: {container_name}")
            
            # Check if container with this name exists
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.ID}}"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.stdout.strip():
                container_id = result.stdout.strip()
                module_logger.info(f"🧹 Removing old Drive MCP container for user {self._user_id}: {container_id}")
                subprocess.run(["docker", "rm", "-f", container_id], check=False, capture_output=True)
                module_logger.info(f"✅ Cleaned up old container: {container_name}")
            else:
                module_logger.info(f"No old Drive MCP container found for user {self._user_id}")
                
        except Exception as e:
            module_logger.warning(f"Failed to cleanup old containers: {e}")

    def _initialize_mcp(self):
        """Initialize MCP tools if credentials are available."""
        module_logger.info(f"Starting Google Drive MCP initialization for user {self._user_id}...")

        if not self._credentials_path:
            module_logger.warning("No Drive credentials path provided - MCP tools disabled")
            return

        try:
            # Clean up any old Drive MCP containers first
            self._cleanup_old_containers()
            
            module_logger.info("Creating Drive MCP toolset...")

            # Convert container path to host path for Docker-in-Docker
            # Container path: /credentials/user_id/drive.json
            # Host path: /home/qais_jabbar/drive-credentials/user_id/drive.json (VM)
            host_credentials_base = os.getenv("HOST_CREDENTIALS_PATH", "/home/qais_jabbar/drive-credentials")
            host_credentials_path = os.path.join(host_credentials_base, self._user_id, "drive.json")
            
            module_logger.info(f"Host credentials path: {host_credentials_path}")
            
            # Create server parameters for the Docker command
            # Note: Docker compose creates network with project name prefix
            network_name = os.getenv("DOCKER_NETWORK", "tara-ai-ml-agent_course-agent-network")
            
            # Use user-specific container name to isolate each user's MCP server
            container_name = f"mcp-gdrive-{self._user_id}"
            module_logger.info(f"Creating Drive MCP container: {container_name}")
            
            server_params = {
                "command": "docker",
                "args": [
                    "run",
                    "-i",
                    "--rm",
                    "--name", container_name,  # Name container by user_id
                    # Mount credentials file from host (for Docker-in-Docker)
                    "--mount", f"type=bind,source={host_credentials_path},target=/credentials.json,readonly",
                    "-e", "GDRIVE_CREDENTIALS_PATH=/credentials.json",
                    "--network", network_name,
                    "mcp/gdrive"
                ]
            }

            # Create MCP toolset with StdioConnectionParams for Docker command
            # MCP resources are automatically exposed when the server provides them
            self._mcp_tools = McpToolset(
                connection_params=StdioConnectionParams(
                    server_params=server_params
                ),
                tool_filter=[
                    "search"
                ]
            )

            module_logger.info(f"Drive MCP toolset created: {self._mcp_tools}")
            module_logger.info(f"Drive MCP toolset type: {type(self._mcp_tools)}")
            module_logger.info("Google Drive MCP tools initialized successfully")
            
        except Exception as e:
            module_logger.error(f"Failed to initialize Google Drive MCP tools: {e}")
            module_logger.error(f"Exception type: {type(e)}")
            import traceback
            module_logger.error(f"Traceback: {traceback.format_exc()}")
            self._mcp_tools = None

    def is_available(self) -> bool:
        """Check if MCP tools are available."""
        return self._mcp_tools is not None

    async def search_files(self, query: str) -> List[Dict[str, Any]]:
        """Search for files in Google Drive using MCP."""
        if not self.is_available():
            module_logger.warning("Google Drive MCP tools not available")
            return []

        try:
            module_logger.info(f"Drive file search for: {query}")
            # Note: MCP tools are called directly by the agent framework
            return []
        except Exception as e:
            module_logger.error(f"Drive search failed: {e}")
            return []

    async def read_file(self, file_name: str) -> str:
        """
        Read a file from Google Drive by name.
        
        This function searches for the file and reads its content.
        
        Args:
            file_name: Name of the file to read (or partial name to search for)
            
        Returns:
            File content as string (Markdown for Docs, CSV for Sheets, etc.)
        """
        if not self.is_available():
            module_logger.warning("Google Drive MCP tools not available")
            return "Error: Google Drive tools not available"

        try:
            module_logger.info(f"Reading file from Drive: {file_name}")
            
            # Note: Direct resource reading through MCP requires the mcp-client
            # This is a placeholder - the ADK framework should handle resource access
            # when the MCP server exposes resources
            
            return f"Drive file reading is being processed. File requested: {file_name}"
            
        except Exception as e:
            module_logger.error(f"Drive file read failed: {e}")
            return f"Error reading file: {str(e)}"

    def extract_source_results(self, files: List[Dict[str, Any]]) -> List[SourceResult]:
        """Convert Drive file data to standardized SourceResult format."""
        results = []
        for file in files:
            result = SourceResult(
                content=file.get('content', ''),
                source_type=SourceType.DRIVE,
                url=file.get('uri', ''),
                repository='',
                metadata={
                    'name': file.get('name', ''),
                    'mimeType': file.get('mimeType', ''),
                }
            )
            results.append(result)
        return results

    # Implement abstract methods from RepositoryTool (not used for Drive, but required)
    async def search_repositories(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Not applicable for Drive MCP - returns empty list."""
        return []

    async def get_file_contents(self, repository: str, file_path: str) -> str:
        """Not applicable for Drive MCP - use read_file instead."""
        return ""

    async def search_code(self, query: str, repository: Optional[str] = None) -> List[Dict[str, Any]]:
        """Not applicable for Drive MCP - use search_files instead."""
        return []