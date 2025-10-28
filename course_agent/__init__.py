"""
Course Generator Agent Package with Source Tracking - Version 2.0
"""

# Import from the refactored agent
from .agent import root_agent, course_agent_instance, get_agent_status

# Import new modular agent
from .agents.course_agent import CourseGenerationAgent, create_course_agent

# Import new configuration system
from .config.settings import settings
from .core.enhanced_source_tracker import EnhancedSourceTracker

# Import old config for backward compatibility (if needed)
try:
    import importlib.util
    import os
    config_file_path = os.path.join(os.path.dirname(__file__), 'config.py')
    spec = importlib.util.spec_from_file_location("legacy_config", config_file_path)
    legacy_config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_config_module)
    config = legacy_config_module.config
except Exception:
    # Fallback if old config.py doesn't exist or has issues
    config = None

# Backward compatibility aliases
course_generator = root_agent
SourceTracker = EnhancedSourceTracker  # Alias old name to new implementation

__all__ = [
    "course_generator",
    "root_agent",
    "course_agent_instance",
    "get_agent_status",
    "CourseGenerationAgent",
    "create_course_agent",
    "config",
    "settings",
    "SourceTracker",
    "EnhancedSourceTracker"
]