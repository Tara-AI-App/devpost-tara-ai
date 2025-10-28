"""
Roadmap Tool - Provides learning path structure from roadmap.sh data
"""
import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Add learning_path to Python path to import roadmap_graph
_learning_path_dir = Path(__file__).parent.parent.parent / "learning_path"
if str(_learning_path_dir) not in sys.path:
    sys.path.insert(0, str(_learning_path_dir))

try:
    from roadmap_graph import RoadmapGraph
except ImportError as e:
    # Fallback: try direct import if module structure is different
    import importlib.util
    spec = importlib.util.spec_from_file_location("roadmap_graph", _learning_path_dir / "roadmap_graph.py")
    if spec and spec.loader:
        roadmap_graph_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(roadmap_graph_module)
        RoadmapGraph = roadmap_graph_module.RoadmapGraph
    else:
        raise ImportError(f"Could not import RoadmapGraph: {e}")

from ..utils.logger import logger


class RoadmapTool:
    """
    Tool for querying roadmap learning paths and structure.

    Uses NetworkX graph to provide intelligent learning path recommendations,
    prerequisite detection, and proper topic ordering based on roadmap.sh data.
    """

    def __init__(self, rag_dir: Optional[str] = None, cache_path: Optional[str] = None):
        """
        Initialize roadmap tool.

        Args:
            rag_dir: Directory containing roadmap RAG data (default: learning_path/rag_data)
            cache_path: Optional cache path for faster loading
        """
        # Determine paths
        if not rag_dir:
            base_dir = Path(__file__).parent.parent.parent
            rag_dir = str(base_dir / "learning_path" / "rag_data")

        if not cache_path:
            base_dir = Path(__file__).parent.parent.parent
            cache_path = str(base_dir / "learning_path" / "cache" / "roadmap_graphs.pkl")

        logger.info(f"Initializing Roadmap Tool with rag_dir={rag_dir}")

        try:
            self.graph = RoadmapGraph(rag_dir, cache_path)
            logger.info(f"Roadmap Tool initialized with {len(self.graph.graphs)} roadmaps")
        except Exception as e:
            logger.error(f"Failed to initialize Roadmap Tool: {e}")
            self.graph = None

    def is_available(self) -> bool:
        """Check if roadmap tool is available."""
        return self.graph is not None and len(self.graph.graphs) > 0

    def query_learning_structure(
        self,
        topic: str,
        roadmap: Optional[str] = None,
        include_prerequisites: bool = True,
        include_next_topics: bool = True
    ) -> str:
        """
        Query learning structure for a topic from roadmap.sh data.

        This tool helps structure courses by providing:
        - Prerequisites that should be covered first
        - Next topics to cover after the current one
        - Proper learning order based on industry-standard roadmaps

        Args:
            topic: The topic to query (e.g., "React Hooks", "Python Async")
            roadmap: Optional roadmap to search in (e.g., "frontend", "python")
            include_prerequisites: Whether to include prerequisite topics
            include_next_topics: Whether to include next topics to cover

        Returns:
            JSON string with topic structure, prerequisites, and next topics

        Example:
            >>> query_learning_structure("React Hooks", roadmap="react")
            {
              "topic": "React Hooks",
              "roadmap": "react",
              "found": true,
              "prerequisites": ["React Components", "JavaScript Basics"],
              "next_topics": ["State Management", "Context API"],
              "recommended_order": ["React Components", "React Hooks", "State Management"]
            }
        """
        if not self.is_available():
            return json.dumps({"error": "Roadmap tool not available"})

        logger.info(f"Querying learning structure for: {topic} (roadmap: {roadmap})")

        try:
            result = {
                "topic": topic,
                "roadmap": roadmap,
                "found": False
            }

            # Get prerequisites
            if include_prerequisites:
                prerequisites = self.graph.get_prerequisites(topic, roadmap)
                if prerequisites:
                    result["found"] = True
                    result["prerequisites"] = [
                        {
                            "name": p["topic_name"],
                            "path": p["full_path"],
                            "resources_count": len(p["resources"])
                        }
                        for p in prerequisites
                    ]

            # Get next topics
            if include_next_topics:
                next_topics = self.graph.get_next_topics(topic, roadmap)
                if next_topics:
                    result["found"] = True
                    result["next_topics"] = [
                        {
                            "name": n["topic_name"],
                            "path": n["full_path"],
                            "resources_count": len(n["resources"])
                        }
                        for n in next_topics
                    ]

            # If topic found, get recommended order
            if result["found"] and include_prerequisites and include_next_topics:
                # Build recommended order: prerequisites → topic → next topics
                order = []
                if "prerequisites" in result:
                    order.extend([p["name"] for p in result["prerequisites"]])
                order.append(topic)
                if "next_topics" in result:
                    order.extend([n["name"] for n in result["next_topics"][:3]])  # Limit to 3

                result["recommended_order"] = order

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error querying learning structure: {e}")
            return json.dumps({"error": str(e)})

    def get_roadmap_structure(
        self,
        roadmap_id: str,
        start_topics: Optional[List[str]] = None
    ) -> str:
        """
        Get the complete learning order for a roadmap.

        Use this to understand the proper sequence of topics to cover
        in a course based on industry-standard learning paths.

        Args:
            roadmap_id: The roadmap to get structure for (e.g., "frontend", "python", "react")
            start_topics: Optional list of starting topics to filter from

        Returns:
            JSON string with ordered list of topics in proper learning sequence

        Example:
            >>> get_roadmap_structure("frontend")
            {
              "roadmap": "frontend",
              "total_topics": 120,
              "learning_order": [
                {"name": "HTML Basics", "type": "topic", "path": "Frontend / HTML / Basics"},
                {"name": "CSS Basics", "type": "topic", "path": "Frontend / CSS / Basics"},
                {"name": "JavaScript Basics", "type": "topic", "path": "Frontend / JavaScript / Basics"}
              ]
            }
        """
        if not self.is_available():
            return json.dumps({"error": "Roadmap tool not available"})

        logger.info(f"Getting roadmap structure for: {roadmap_id}")

        try:
            # Get learning order
            learning_order = self.graph.get_learning_order(roadmap_id, start_topics)

            result = {
                "roadmap": roadmap_id,
                "total_topics": len(learning_order),
                "learning_order": [
                    {
                        "name": topic["topic_name"],
                        "type": topic["topic_type"],
                        "path": topic["full_path"],
                        "resources_count": len(topic["resources"])
                    }
                    for topic in learning_order
                ]
            }

            # Add stats
            stats = self.graph.get_roadmap_stats(roadmap_id)
            if stats:
                result["stats"] = stats

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error getting roadmap structure: {e}")
            return json.dumps({"error": str(e)})

    def find_learning_resources(
        self,
        topic: str,
        roadmap: Optional[str] = None
    ) -> str:
        """
        Find curated learning resources for a specific topic.

        Returns official resources from roadmap.sh for the topic.

        Args:
            topic: The topic to find resources for
            roadmap: Optional roadmap filter

        Returns:
            JSON string with learning resources from roadmap.sh

        Example:
            >>> find_learning_resources("React Hooks")
            {
              "topic": "React Hooks",
              "resources_found": 5,
              "resources": [
                {"title": "React Docs - Hooks", "url": "https://react.dev/reference/react"},
                {"title": "Hooks at a Glance", "url": "https://react.dev/learn"}
              ]
            }
        """
        if not self.is_available():
            return json.dumps({"error": "Roadmap tool not available"})

        logger.info(f"Finding resources for: {topic}")

        try:
            # Search for the topic
            results = self.graph.search_topics(topic, roadmap)

            if not results:
                return json.dumps({
                    "topic": topic,
                    "resources_found": 0,
                    "message": f"No resources found for '{topic}'"
                })

            # Collect all resources from matching topics
            all_resources = []
            for result in results:
                for resource in result.get("resources", []):
                    if resource not in all_resources:
                        all_resources.append(resource)

            response = {
                "topic": topic,
                "resources_found": len(all_resources),
                "resources": all_resources,
                "matching_topics": [
                    {
                        "name": r["topic_name"],
                        "roadmap": r["roadmap"],
                        "path": r["full_path"]
                    }
                    for r in results
                ]
            }

            return json.dumps(response, indent=2)

        except Exception as e:
            logger.error(f"Error finding resources: {e}")
            return json.dumps({"error": str(e)})

    def list_available_roadmaps(self) -> str:
        """
        List all available roadmaps with statistics.

        Use this to see what roadmaps are available and choose
        the appropriate one for course structure.

        Returns:
            JSON string with list of available roadmaps and their statistics

        Example:
            >>> list_available_roadmaps()
            {
              "total_roadmaps": 4,
              "roadmaps": [
                {"id": "frontend", "title": "Frontend Development", "topics": 120},
                {"id": "python", "title": "Python", "topics": 99},
                {"id": "react", "title": "React", "topics": 91}
              ]
            }
        """
        if not self.is_available():
            return json.dumps({"error": "Roadmap tool not available"})

        try:
            roadmaps = self.graph.list_roadmaps()

            response = {
                "total_roadmaps": len(roadmaps),
                "roadmaps": [
                    {
                        "id": rm["roadmap_id"],
                        "title": rm["title"],
                        "topics": rm["stats"]["total_topics"],
                        "connections": rm["stats"]["total_connections"],
                        "topics_with_resources": rm["stats"]["topics_with_resources"]
                    }
                    for rm in roadmaps
                ]
            }

            return json.dumps(response, indent=2)

        except Exception as e:
            logger.error(f"Error listing roadmaps: {e}")
            return json.dumps({"error": str(e)})

    def search_roadmap_topics(
        self,
        keyword: str,
        roadmap: Optional[str] = None
    ) -> str:
        """
        Search for topics matching a keyword across roadmaps.

        Use this to find relevant topics when planning course structure.

        Args:
            keyword: Keyword to search for
            roadmap: Optional roadmap to search in

        Returns:
            JSON string with matching topics

        Example:
            >>> search_roadmap_topics("hooks", roadmap="react")
            {
              "keyword": "hooks",
              "matches_found": 2,
              "topics": [
                {"name": "React Hooks", "roadmap": "react", "path": "React / Hooks"}
              ]
            }
        """
        if not self.is_available():
            return json.dumps({"error": "Roadmap tool not available"})

        logger.info(f"Searching for keyword: {keyword}")

        try:
            results = self.graph.search_topics(keyword, roadmap)

            response = {
                "keyword": keyword,
                "matches_found": len(results),
                "topics": [
                    {
                        "name": r["topic_name"],
                        "roadmap": r["roadmap"],
                        "path": r["full_path"],
                        "type": r["topic_type"],
                        "resources_count": len(r["resources"])
                    }
                    for r in results[:10]  # Limit to 10 results
                ]
            }

            return json.dumps(response, indent=2)

        except Exception as e:
            logger.error(f"Error searching topics: {e}")
            return json.dumps({"error": str(e)})


# Test
if __name__ == "__main__":
    tool = RoadmapTool()

    if tool.is_available():
        print("✓ Roadmap Tool initialized")
        print("\nAvailable roadmaps:")
        print(tool.list_available_roadmaps())

        print("\n\nQuerying 'React Hooks':")
        print(tool.query_learning_structure("React Hooks", roadmap="react"))
    else:
        print("❌ Roadmap Tool not available")
