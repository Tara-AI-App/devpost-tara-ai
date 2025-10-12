"""
Google Drive MCP tool implementation.
"""
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
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
        self._mcp_tools: Optional[bool] = None  # Flag indicating if MCP tools are available
        self._user_id = user_id
        self._credentials_path = credentials_path
        
        if credentials_path:
            self._initialize_mcp()
    
    def _initialize_mcp(self):
        """
        Initialize MCP tools if credentials are available.
        
        Note: The new qais004/tara-mcp-drive uses stdio transport with docker run -i,
        so we don't need to start a persistent container. We just verify credentials exist.
        """
        module_logger.info(f"Starting Google Drive MCP initialization for user {self._user_id}...")

        if not self._credentials_path:
            module_logger.warning("No Drive credentials path provided - MCP tools disabled")
            self._mcp_tools = None
            return

        try:
            # Verify credentials file exists
            if os.path.exists(self._credentials_path):
                module_logger.info(f"✅ Drive credentials found at: {self._credentials_path}")
                # Mark as available by setting a simple flag
                self._mcp_tools = True  # Just a flag to indicate availability
                module_logger.info("Google Drive MCP tools initialized successfully")
            else:
                module_logger.warning(f"Drive credentials file not found: {self._credentials_path}")
                self._mcp_tools = None
            
        except Exception as e:
            module_logger.error(f"Failed to initialize Google Drive MCP tools: {e}")
            import traceback
            module_logger.error(f"Traceback: {traceback.format_exc()}")
            self._mcp_tools = None

    def is_available(self) -> bool:
        """Check if MCP tools are available (credentials exist)."""
        return self._mcp_tools is not None

    async def search_files(self, query: str) -> List[Dict[str, str]]:
        """
        Search for files in Google Drive using MCP search tool.
        
        Args:
            query: Search query string
            
        Returns:
            List of dicts with 'name' and 'mimeType' for each file found
        """
        if not self.is_available():
            module_logger.warning("Google Drive MCP tools not available")
            return []

        try:
            import subprocess
            import json
            
            module_logger.info(f"Searching Drive for: {query}")
            
            # Convert container path to host path for Docker-in-Docker
            host_credentials_base = os.getenv("HOST_CREDENTIALS_PATH", "/home/qais_jabbar/drive-credentials")
            module_logger.info(f"DEBUG: host_credentials_base = {host_credentials_base}")
            module_logger.info(f"DEBUG: self._user_id = {self._user_id}")
            module_logger.info(f"DEBUG: self._credentials_path = {self._credentials_path}")
            
            host_credentials_path = os.path.join(host_credentials_base, self._user_id, "drive.json")
            module_logger.info(f"DEBUG: host_credentials_path = {host_credentials_path}")
            
            network_name = os.getenv("DOCKER_NETWORK", "tara-ai-ml-agent_course-agent-network")
            
            # Call search tool via JSON-RPC using docker run -i (stdio transport)
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "search",
                    "arguments": {"query": query}
                }
            }
            
            request_json = json.dumps(request) + "\n"
            
            # Run MCP server container with stdio transport
            result = subprocess.run(
                [
                    "docker", "run", "-i", "--rm",
                    "--mount", f"type=bind,source={host_credentials_path},target=/credentials.json,readonly",
                    "-e", "GDRIVE_CREDENTIALS_PATH=/credentials.json",
                    "--network", network_name,
                    "qais004/tara-mcp-drive"
                ],
                input=request_json,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                module_logger.error(f"Failed to search Drive: {result.stderr}")
                return []
            
            # Parse JSON-RPC response from stdout
            output = result.stdout
            
            if not output or not output.strip():
                module_logger.error("Empty output from MCP server")
                module_logger.error(f"stderr: {result.stderr[:500]}")
                return []
            
            module_logger.debug(f"Raw output: {output[:500]}...")
            
            # Find JSON-RPC response by looking for lines starting with {"result" or {"jsonrpc"
            response = None
            lines = output.split('\n')
            
            for line in lines:
                line = line.strip()
                if line.startswith('{"result') or line.startswith('{"jsonrpc') or line.startswith('{"error'):
                    try:
                        response = json.loads(line)
                        module_logger.info(f"✅ Found and parsed JSON-RPC response")
                        break
                    except json.JSONDecodeError as e:
                        module_logger.debug(f"Failed to parse potential JSON line: {e}")
                        continue
            
            if not response:
                module_logger.error(f"❌ No valid JSON-RPC response found in output")
                module_logger.error(f"Output lines: {[line[:100] for line in lines[:5] if line.strip()]}")
                return []
            
            if "result" not in response:
                module_logger.warning(f"No result in search response: {response}")
                return []
            
            # Extract text content from result
            content = response["result"].get("content", [])
            if not content:
                module_logger.warning("No content in search result")
                return []
            
            # The new MCP server returns structured JSON string in the text field
            text = content[0].get("text", "")
            module_logger.info(f"Search result text: {text[:200]}...")
            
            # Parse the structured JSON response
            try:
                search_data = json.loads(text)
                files = search_data.get("files", [])
                
                # Convert to the format expected by the rest of the code
                result_files = []
                for file in files:
                    result_files.append({
                        "uri": file.get("uri", ""),
                        "id": file.get("id", ""),
                        "name": file.get("name", ""),
                        "mimeType": file.get("mimeType", ""),
                        "modifiedTime": file.get("modifiedTime", ""),
                        "size": file.get("size", ""),
                        "webViewLink": file.get("webViewLink", "")
                    })
                
                module_logger.info(f"✅ Found {len(result_files)} files matching '{query}'")
                return result_files
                
            except json.JSONDecodeError as e:
                module_logger.error(f"Failed to parse search result JSON: {e}")
                module_logger.error(f"Text content: {text}")
                return []
            
        except Exception as e:
            module_logger.error(f"Drive search failed: {e}")
            import traceback
            module_logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    async def get_file(self, uri: str) -> Dict[str, Any]:
        """
        Get file content from Google Drive using MCP get_file tool.
        
        Args:
            uri: File URI from search results (e.g., "gdrive:///fileId")
            
        Returns:
            Dict with file metadata and content
        """
        if not self.is_available():
            module_logger.warning("Google Drive MCP tools not available")
            return {}

        try:
            import subprocess
            import json
            
            module_logger.info(f"Getting file content for: {uri}")
            
            # Convert container path to host path for Docker-in-Docker
            host_credentials_base = os.getenv("HOST_CREDENTIALS_PATH", "/home/qais_jabbar/drive-credentials")
            module_logger.info(f"DEBUG: host_credentials_base = {host_credentials_base}")
            module_logger.info(f"DEBUG: self._user_id = {self._user_id}")
            
            host_credentials_path = os.path.join(host_credentials_base, self._user_id, "drive.json")
            module_logger.info(f"DEBUG: host_credentials_path = {host_credentials_path}")
            
            network_name = os.getenv("DOCKER_NETWORK", "tara-ai-ml-agent_course-agent-network")
            
            # Call get_file tool via JSON-RPC using docker run -i (stdio transport)
            request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "get_file",
                    "arguments": {"uri": uri}
                }
            }
            
            request_json = json.dumps(request) + "\n"
            
            # Run MCP server container with stdio transport
            result = subprocess.run(
                [
                    "docker", "run", "-i", "--rm",
                    "--mount", f"type=bind,source={host_credentials_path},target=/credentials.json,readonly",
                    "-e", "GDRIVE_CREDENTIALS_PATH=/credentials.json",
                    "--network", network_name,
                    "qais004/tara-mcp-drive"
                ],
                input=request_json,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                module_logger.error(f"Failed to get file: {result.stderr}")
                return {}
            
            # Parse JSON-RPC response from stdout
            output = result.stdout
            
            if not output or not output.strip():
                module_logger.error("Empty output from MCP server")
                module_logger.error(f"stderr: {result.stderr[:500]}")
                return {}
            
            module_logger.debug(f"Raw output: {output[:500]}...")
            
            # Find JSON-RPC response
            response = None
            lines = output.split('\n')
            
            for line in lines:
                line = line.strip()
                if line.startswith('{"result') or line.startswith('{"jsonrpc') or line.startswith('{"error'):
                    try:
                        response = json.loads(line)
                        module_logger.info(f"✅ Found and parsed JSON-RPC response")
                        break
                    except json.JSONDecodeError as e:
                        module_logger.debug(f"Failed to parse potential JSON line: {e}")
                        continue
            
            if not response:
                module_logger.error(f"❌ No valid JSON-RPC response found in output")
                return {}
            
            if "result" not in response:
                module_logger.warning(f"No result in get_file response: {response}")
                return {}
            
            # Extract text content from result
            content = response["result"].get("content", [])
            if not content:
                module_logger.warning("No content in get_file result")
                return {}
            
            # The new MCP server returns structured JSON string in the text field
            text = content[0].get("text", "")
            
            # Parse the structured JSON response
            try:
                file_data = json.loads(text)
                module_logger.info(f"✅ Retrieved file: {file_data.get('name')} ({file_data.get('contentType')})")
                return file_data
                
            except json.JSONDecodeError as e:
                module_logger.error(f"Failed to parse get_file result JSON: {e}")
                module_logger.error(f"Text content: {text[:200]}")
                return {}
            
        except Exception as e:
            module_logger.error(f"Drive get_file failed: {e}")
            import traceback
            module_logger.error(f"Traceback: {traceback.format_exc()}")
            return {}

    async def list_resources(self) -> List[Dict[str, Any]]:
        """
        List all available Drive resources (files).
        
        DEPRECATED: The new qais004/tara-mcp-drive server doesn't support resources/list.
        Use search_files() instead to find files.
        
        Returns:
            Empty list (deprecated)
        """
        module_logger.warning("list_resources is deprecated. Use search_files() instead.")
        return []
    
    async def read_resource(self, uri: str) -> str:
        """
        Read a specific Drive resource by URI.
        
        DEPRECATED: The new qais004/tara-mcp-drive server doesn't support resources/read.
        Use get_file() instead to retrieve file content.
        
        Args:
            uri: Resource URI (e.g., "gdrive:///file_id")
            
        Returns:
            Empty string (deprecated)
        """
        module_logger.warning("read_resource is deprecated. Use get_file() instead.")
        return ""

    async def read_file(self, file_name: str) -> str:
        """
        Read a file from Google Drive by name.
        
        This function searches for the file and reads its content using the new workflow:
        1. Search for files matching the name
        2. Get the URI of the first match
        3. Use get_file to retrieve the content
        
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
            
            # Search for files matching the name
            files = await self.search_files(file_name)
            
            if not files:
                return f"File not found: {file_name}"
            
            # Get the first matching file's URI
            first_file = files[0]
            uri = first_file.get("uri")
            
            if not uri:
                return f"No URI found for file: {file_name}"
            
            # Get file content using get_file
            file_data = await self.get_file(uri)
            
            if not file_data:
                return f"Failed to retrieve file content: {file_name}"
            
            content = file_data.get("content", "")
            module_logger.info(f"✅ Successfully read file: {first_file.get('name')} ({len(content)} chars)")
            
            return content
            
        except Exception as e:
            module_logger.error(f"Drive file read failed: {e}")
            import traceback
            module_logger.error(f"Traceback: {traceback.format_exc()}")
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