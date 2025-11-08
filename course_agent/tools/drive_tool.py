"""
Google Drive MCP tool implementation - HTTP mode for Cloud Run sidecar.
"""
import os
import json
import requests
from typing import Optional, Dict, Any, List
import logging

from .base import RepositoryTool, SourceResult, SourceType
from ..config.settings import settings
from ..utils.logger import logger as module_logger


class GoogleDriveMCPTool(RepositoryTool):
    """Google Drive MCP tool implementation - HTTP mode with direct token passing."""

    def __init__(self, user_id: str = None, access_token: str = None, mcp_url: str = None):
        self._mcp_tools: Optional[bool] = None  # Flag indicating if MCP tools are available
        self._user_id = user_id
        self._access_token = access_token
        self._mcp_url = mcp_url or os.getenv("MCP_DRIVE_URL", "http://localhost:9000")
        
        if access_token:
            self._initialize_mcp()
    
    def _initialize_mcp(self):
        """
        Initialize MCP tools if access token is available.
        
        Note: In HTTP mode, we just verify the MCP server is reachable.
        """
        module_logger.info(f"Starting Google Drive MCP initialization for user {self._user_id}...")

        if not self._access_token:
            module_logger.warning("No Drive access token provided - MCP tools disabled")
            self._mcp_tools = None
            return

        try:
            # Test MCP server connection
            try:
                response = requests.post(
                    self._mcp_url,
                    json={
                        "jsonrpc": "2.0",
                        "id": 0,
                        "method": "tools/list"
                    },
                    timeout=5
                )
                if response.status_code == 200:
                    module_logger.info(f"✅ MCP Drive server reachable at {self._mcp_url}")
                    self._mcp_tools = True
                else:
                    module_logger.warning(f"MCP server returned status {response.status_code}")
                    self._mcp_tools = None
            except requests.exceptions.RequestException as e:
                module_logger.warning(f"MCP server not reachable at {self._mcp_url}: {e}")
                module_logger.warning("Drive tools will be disabled")
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
        Search for files in Google Drive using MCP search tool via HTTP.
        
        Args:
            query: Search query string
            
        Returns:
            List of dicts with file metadata for each file found
        """
        if not self.is_available():
            module_logger.warning("Google Drive MCP tools not available")
            return []

        try:
            module_logger.info(f"Searching Drive for: {query}")
            
            if not self._access_token:
                module_logger.error("No access_token available")
                return []
            
            # Call MCP server via HTTP with the access token
            request_data = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "search",
                    "arguments": {
                        "query": query,
                        "access_token": self._access_token
                    }
                }
            }
            
            module_logger.debug(f"Sending HTTP request to {self._mcp_url}")
            
            response = requests.post(
                self._mcp_url,
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=90
            )
            
            if response.status_code != 200:
                module_logger.error(f"Failed to search Drive: HTTP {response.status_code}")
                module_logger.error(f"Response: {response.text}")
                return []
            
            # Parse JSON-RPC response
            json_response = response.json()
            
            if "error" in json_response:
                module_logger.error(f"MCP server returned error: {json_response['error']}")
                return []
            
            if "result" not in json_response:
                module_logger.warning(f"No result in search response: {json_response}")
                return []
            
            # Extract text content from result
            content = json_response["result"].get("content", [])
            if not content:
                module_logger.warning("No content in search result")
                return []
            
            # The MCP server returns structured JSON string in the text field
            text = content[0].get("text", "")
            module_logger.info(f"Search result text: {text[:200]}...")
            
            # Parse the structured JSON response
            try:
                search_data = json.loads(text)
                files = search_data.get("files", [])
                
                module_logger.info(f"✅ Found {len(files)} files matching '{query}'")
                return files
                
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
        Get file content from Google Drive using MCP get_file tool via HTTP.
        
        Args:
            uri: File URI from search results (e.g., "gdrive:///fileId")
            
        Returns:
            Dict with file metadata and content
        """
        if not self.is_available():
            module_logger.warning("Google Drive MCP tools not available")
            return {}

        try:
            module_logger.info(f"Getting file content for: {uri}")
            
            if not self._access_token:
                module_logger.error("No access_token available")
                return {}
            
            # Call MCP server via HTTP with the access token
            request_data = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "get_file",
                    "arguments": {
                        "uri": uri,
                        "access_token": self._access_token
                    }
                }
            }
            
            module_logger.debug(f"Sending HTTP request to {self._mcp_url}")
            
            response = requests.post(
                self._mcp_url,
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=180
            )
            
            if response.status_code != 200:
                module_logger.error(f"Failed to get file: HTTP {response.status_code}")
                module_logger.error(f"Response: {response.text}")
                return {}
            
            # Parse JSON-RPC response
            json_response = response.json()
            
            if "error" in json_response:
                module_logger.error(f"MCP server returned error: {json_response['error']}")
                return {}
            
            if "result" not in json_response:
                module_logger.warning(f"No result in get_file response: {json_response}")
                return {}
            
            # Extract text content from result
            content = json_response["result"].get("content", [])
            if not content:
                module_logger.warning("No content in get_file result")
                return {}
            
            # The MCP server returns structured JSON string in the text field
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