from .agents.guide_agent import create_guide_agent, GuideGenerationAgent
from course_agent.config.settings import settings
from course_agent.utils.logger import logger
import os

# Create the main agent instance
guide_agent_instance = create_guide_agent(
    github_token=os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN'),
    drive_token=os.getenv('GOOGLE_DRIVE_TOKEN'),
    user_id=os.getenv('USER_ID', 'default-user')
)

# Export the ADK agent
root_agent = guide_agent_instance.get_agent()

# Export main functions for direct use
analyze_tech_stack = guide_agent_instance.analyze_tech_stack
discover_sources = guide_agent_instance.discover_sources
extract_repository_content = guide_agent_instance.extract_repository_content
get_tracked_sources = guide_agent_instance.get_tracked_sources
generate_search_queries = guide_agent_instance.generate_search_queries

# Export configuration and status functions
def get_agent_status():
    """Get comprehensive agent status and configuration."""
    return guide_agent_instance.get_configuration_status()

def reload_configuration():
    """Reload configuration and recreate agent."""
    global guide_agent_instance, root_agent
    guide_agent_instance = create_guide_agent(
        github_token=os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN'),
        drive_token=os.getenv('GOOGLE_DRIVE_TOKEN'),
        user_id=os.getenv('USER_ID')
    )
    root_agent = guide_agent_instance.get_agent()
    logger.info("Guide agent configuration reloaded")

# Log initialization
logger.info("Guide Agent v1.0 initialized")
logger.info(f"Configuration status: {get_agent_status()}")

# Backward compatibility exports
__all__ = [
    'root_agent',
    'guide_agent_instance',
    'analyze_tech_stack',
    'discover_sources',
    'extract_repository_content',
    'get_tracked_sources',
    'generate_search_queries',
    'get_agent_status',
    'reload_configuration',
    'GuideGenerationAgent',
    'settings',
    'logger'
]
