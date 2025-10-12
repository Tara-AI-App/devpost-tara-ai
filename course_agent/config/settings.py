"""
Enhanced configuration system for course generation agent.
"""
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class SourcePriority(str, Enum):
    RAG_FIRST = "rag_first"
    GITHUB_FIRST = "github_first"
    BALANCED = "balanced"


@dataclass
class MCPConfig:
    """Configuration for MCP tools."""
    github_token: Optional[str] = field(default_factory=lambda: os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN'))
    github_api_url: str = "https://api.githubcopilot.com/mcp/"
    enabled_tools: list[str] = field(default_factory=lambda: [
        "search_repositories",
        "get_file_contents",
        "search_code",
        "list_projects",
        "get_me"
    ])
    max_repositories: int = 2  # Reduced from 5 to 2 for faster performance
    max_code_results: int = 10
    quality_threshold: int = 100  # Minimum stars for repository consideration

    # Google Drive MCP toggle
    enable_drive_mcp: bool = field(default_factory=lambda: os.getenv('ENABLE_DRIVE_MCP', 'false').lower() == 'true')
    drive_token: Optional[str] = field(default_factory=lambda: os.getenv('GOOGLE_DRIVE_TOKEN'))


@dataclass
class RAGConfig:
    """Configuration for RAG processing."""
    max_results: int = 2  # Reduced from 5 to avoid token limit
    relevance_threshold: float = 0.7
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass
class CourseConfig:
    """Configuration for course generation."""
    default_difficulty: str = "Intermediate"
    default_duration: str = "8-12 hours"
    max_modules: int = 6
    max_lessons_per_module: int = 4
    include_code_examples: bool = True
    include_repository_links: bool = True


@dataclass
class SourceTrackingConfig:
    """Configuration for source tracking."""
    max_references_per_type: int = 5
    track_content_preview: bool = True
    preview_length: int = 200


@dataclass
class AgentConfig:
    """Main agent configuration."""
    name: str = "course_generator"
    description: str = "Technical course generator with dynamic source discovery"
    model_name: str = "gemini-2.5-flash"
    source_priority: SourcePriority = SourcePriority.RAG_FIRST
    log_level: LogLevel = LogLevel.INFO

    # Model generation parameters for deterministic behavior
    temperature: float = 0.1  # Lower temperature for more predictable outputs
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 50000  # Increased for complete course JSON generation

    # Sub-configurations
    mcp: MCPConfig = field(default_factory=MCPConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    course: CourseConfig = field(default_factory=CourseConfig)
    source_tracking: SourceTrackingConfig = field(default_factory=SourceTrackingConfig)

    # Environment-specific settings
    project_id: Optional[str] = field(default_factory=lambda: os.getenv('GOOGLE_CLOUD_PROJECT'))
    location: str = field(default_factory=lambda: os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1'))

    def validate(self) -> Dict[str, Any]:
        """Validate configuration and return any issues."""
        issues = {}

        if self.mcp.github_token is None:
            issues['github_token'] = "GitHub token not set - MCP GitHub tools will be unavailable"

        if self.project_id is None:
            issues['project_id'] = "Google Cloud project ID not set"

        if self.mcp.max_repositories <= 0:
            issues['max_repositories'] = "Max repositories must be positive"

        if self.rag.max_results <= 0:
            issues['max_results'] = "Max RAG results must be positive"

        return issues

    def is_mcp_enabled(self) -> bool:
        """Check if MCP tools can be enabled."""
        return self.mcp.github_token is not None

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for agent initialization."""
        return {
            "model_name": self.model_name,
            "project_id": self.project_id,
            "location": self.location,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens
        }


def load_config() -> AgentConfig:
    """Load configuration from environment variables and defaults."""
    config = AgentConfig()

    # Override with environment variables if available
    if os.getenv('COURSE_AGENT_LOG_LEVEL'):
        config.log_level = LogLevel(os.getenv('COURSE_AGENT_LOG_LEVEL'))

    if os.getenv('COURSE_AGENT_SOURCE_PRIORITY'):
        config.source_priority = SourcePriority(os.getenv('COURSE_AGENT_SOURCE_PRIORITY'))

    if os.getenv('COURSE_AGENT_MAX_REPOSITORIES'):
        config.mcp.max_repositories = int(os.getenv('COURSE_AGENT_MAX_REPOSITORIES'))

    if os.getenv('COURSE_AGENT_RAG_MAX_RESULTS'):
        config.rag.max_results = int(os.getenv('COURSE_AGENT_RAG_MAX_RESULTS'))

    return config


# Global configuration instance
settings = load_config()