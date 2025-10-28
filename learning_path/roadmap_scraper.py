#!/usr/bin/env python3
"""
Roadmap.sh Data Scraper
Scrapes learning path data from the roadmap.sh GitHub repository
Optimized for RAG (Retrieval-Augmented Generation) storage
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set
import requests
from dataclasses import dataclass, asdict, field
import yaml


@dataclass
class TopicContent:
    """Represents detailed content for a roadmap topic"""
    node_id: str
    topic_name: str
    node_type: str  # 'topic', 'subtopic', 'paragraph', etc.
    content: str  # Full markdown content
    resources: List[Dict[str, str]]  # List of {title, url}
    parent_nodes: List[str] = field(default_factory=list)
    child_nodes: List[str] = field(default_factory=list)
    position: Optional[Dict] = None  # {x, y} for context
    metadata: Dict = field(default_factory=dict)


@dataclass
class Roadmap:
    """Represents a complete roadmap optimized for RAG"""
    roadmap_id: str  # e.g., 'frontend'
    title: str
    description: str
    overview: str  # High-level description from .md file
    topics: List[TopicContent]
    node_relationships: Dict[str, List[str]] = field(default_factory=dict)  # node_id -> [child_ids]
    skills_required: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class RoadmapScraper:
    """Scrapes roadmap data from the GitHub repository optimized for RAG"""

    # List of all available roadmaps (as of early 2025)
    AVAILABLE_ROADMAPS = [
        'ai-agents', 'ai-data-scientist', 'ai-engineer', 'ai-red-teaming', 'android',
        'angular', 'api-design', 'aspnet-core', 'aws', 'backend', 'bi-analyst',
        'blockchain', 'cloudflare', 'code-review', 'computer-science', 'cpp', 'css',
        'cyber-security', 'data-analyst', 'data-engineer', 'datastructures-and-algorithms',
        'design-system', 'devops', 'devrel', 'docker', 'engineering-manager', 'flutter',
        'frontend', 'full-stack', 'game-developer', 'git-github', 'golang', 'graphql',
        'html', 'ios', 'java', 'javascript', 'kotlin', 'kubernetes', 'linux',
        'machine-learning', 'mlops', 'mongodb', 'nextjs', 'nodejs', 'php',
        'postgresql-dba', 'product-manager', 'prompt-engineering', 'python', 'qa',
        'react-native', 'react', 'redis', 'rust', 'server-side-game-developer',
        'software-architect', 'software-design-architecture', 'spring-boot', 'sql',
        'system-design', 'technical-writer', 'terraform', 'typescript', 'ux-design', 'vue'
    ]

    def __init__(self, github_token: Optional[str] = None, branch: str = 'master'):
        self.base_url = "https://api.github.com/repos/kamranahmedse/developer-roadmap"
        self.raw_content_url = f"https://raw.githubusercontent.com/kamranahmedse/developer-roadmap/{branch}"
        self.branch = branch
        self.headers = {
            "Accept": "application/vnd.github.v3+json"
        }
        if github_token:
            self.headers["Authorization"] = f"token {github_token}"

        # Cache for avoiding redundant API calls
        self._repo_tree = None
    
    def get_repository_structure(self) -> Dict:
        """Get the repository tree structure (cached)"""
        if self._repo_tree is not None:
            return self._repo_tree

        url = f"{self.base_url}/git/trees/{self.branch}?recursive=1"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        self._repo_tree = response.json()
        return self._repo_tree

    def get_file_content(self, file_path: str) -> str:
        """Get content of a specific file from GitHub"""
        url = f"{self.raw_content_url}/{file_path}"
        response = requests.get(url)
        response.raise_for_status()
        return response.text

    def find_roadmap_content_files(self, roadmap_name: str) -> List[str]:
        """Find all content markdown files for a specific roadmap"""
        tree = self.get_repository_structure()
        content_files = []

        content_dir = f"src/data/roadmaps/{roadmap_name}/content"

        for item in tree.get('tree', []):
            path = item['path']
            if path.startswith(content_dir) and path.endswith('.md'):
                content_files.append(path)

        return content_files

    def extract_node_id_from_filename(self, filename: str) -> Optional[str]:
        """Extract node ID from content filename (e.g., 'accessibility@iJIqi7ngpGHWAqtgdjgxB.md')"""
        match = re.search(r'@([a-zA-Z0-9_-]+)\.md$', filename)
        if match:
            return match.group(1)
        return None

    def extract_topic_name_from_filename(self, filename: str) -> str:
        """Extract topic name from content filename"""
        basename = os.path.basename(filename)
        # Remove .md extension and node ID
        name = re.sub(r'@[a-zA-Z0-9_-]+\.md$', '', basename)
        return name.replace('-', ' ').replace('_', ' ').title()
    
    def parse_roadmap_overview(self, content: str) -> Dict:
        """Parse the main roadmap .md file with YAML frontmatter"""
        # Split YAML frontmatter from content
        parts = content.split('---')

        metadata = {}
        overview = ""
        skills = []

        if len(parts) >= 3:
            # Parse YAML frontmatter
            try:
                yaml_content = parts[1]
                metadata = yaml.safe_load(yaml_content) or {}
            except Exception as e:
                print(f"Warning: Could not parse YAML frontmatter: {e}")

            # Get markdown content after frontmatter
            overview = '---'.join(parts[2:]).strip()
        else:
            overview = content.strip()

        # Extract skills from overview text
        # Look for patterns like "Required skills:" or "Skills:" followed by a list
        skills_match = re.search(r'(?:Required\s+)?Skills?:?\s*\n((?:[-*]\s+.+\n?)+)', overview, re.IGNORECASE)
        if skills_match:
            skills_text = skills_match.group(1)
            skills = [line.strip('- *\n') for line in skills_text.split('\n') if line.strip()]

        return {
            'metadata': metadata,
            'overview': overview,
            'skills': skills,
            'title': metadata.get('briefTitle') or metadata.get('title', ''),
            'description': metadata.get('briefDescription') or metadata.get('description', '')
        }

    def parse_json_nodes(self, json_content: str) -> Dict[str, Dict]:
        """Parse JSON roadmap file and extract all nodes with their relationships"""
        data = json.loads(json_content)
        nodes = {}

        # Handle different JSON structures
        if isinstance(data, dict) and 'nodes' in data:
            node_list = data['nodes']
        elif isinstance(data, list):
            node_list = data
        else:
            return nodes

        for node in node_list:
            node_id = node.get('id')
            if not node_id:
                continue

            node_data = node.get('data', {})

            nodes[node_id] = {
                'id': node_id,
                'type': node.get('type', 'unknown'),
                'label': node_data.get('label', ''),
                'position': node.get('position', {}),
                'width': node.get('width'),
                'height': node.get('height'),
                'style': node.get('style', {})
            }

        # Extract edges/relationships if present
        relationships = {}
        if isinstance(data, dict) and 'edges' in data:
            for edge in data.get('edges', []):
                source = edge.get('source')
                target = edge.get('target')
                if source and target:
                    if source not in relationships:
                        relationships[source] = []
                    relationships[source].append(target)

        return {'nodes': nodes, 'relationships': relationships}

    def parse_content_markdown(self, content: str) -> Dict:
        """Parse a content markdown file to extract information and resources"""
        # Extract all links/resources
        resources = []
        links = re.findall(r'\[([^\]]+)\]\(([^\)]+)\)', content)
        for text, url in links:
            resources.append({'title': text, 'url': url})

        # Remove markdown links for clean content
        clean_content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)

        return {
            'content': clean_content.strip(),
            'resources': resources
        }
    
    def scrape_roadmap(self, roadmap_name: str, include_content: bool = True) -> Optional[Roadmap]:
        """
        Scrape a complete roadmap with all its content

        Args:
            roadmap_name: Name of the roadmap (e.g., 'frontend', 'backend')
            include_content: Whether to fetch detailed content files (slower but more complete)

        Returns:
            Roadmap object with all topics and content
        """
        try:
            print(f"Scraping roadmap: {roadmap_name}")

            # 1. Get the main JSON file with node structure
            json_path = f"src/data/roadmaps/{roadmap_name}/{roadmap_name}.json"
            print(f"  - Fetching JSON structure...")
            json_content = self.get_file_content(json_path)
            parsed_json = self.parse_json_nodes(json_content)
            nodes = parsed_json['nodes']
            relationships = parsed_json['relationships']

            print(f"  - Found {len(nodes)} nodes")

            # 2. Get the overview markdown file
            overview_path = f"src/data/roadmaps/{roadmap_name}/{roadmap_name}.md"
            print(f"  - Fetching overview...")
            try:
                overview_content = self.get_file_content(overview_path)
                overview_data = self.parse_roadmap_overview(overview_content)
            except Exception as e:
                print(f"    Warning: Could not fetch overview: {e}")
                overview_data = {
                    'metadata': {},
                    'overview': '',
                    'skills': [],
                    'title': roadmap_name.replace('-', ' ').title(),
                    'description': ''
                }

            # 3. Get all content files if requested
            topics = []
            if include_content:
                print(f"  - Fetching content files...")
                content_files = self.find_roadmap_content_files(roadmap_name)
                print(f"  - Found {len(content_files)} content files")

                # Build a mapping of node_id -> content
                content_map = {}
                for file_path in content_files:
                    node_id = self.extract_node_id_from_filename(file_path)
                    if node_id:
                        try:
                            content = self.get_file_content(file_path)
                            parsed_content = self.parse_content_markdown(content)
                            content_map[node_id] = {
                                'file_path': file_path,
                                'topic_name': self.extract_topic_name_from_filename(file_path),
                                **parsed_content
                            }
                        except Exception as e:
                            print(f"    Warning: Could not fetch {file_path}: {e}")

                print(f"  - Successfully fetched {len(content_map)} content files")

                # 4. Combine node structure with content
                for node_id, node_info in nodes.items():
                    # Skip non-topic nodes (like visual elements, titles, etc.)
                    node_type = node_info.get('type', '')
                    if node_type in ['paragraph', 'button', 'vertical', 'horizontal']:
                        continue

                    # Get content if available
                    content_data = content_map.get(node_id, {})

                    # Determine topic name
                    topic_name = content_data.get('topic_name') or node_info.get('label', f'Node {node_id}')

                    # Get parent and child relationships
                    child_nodes = relationships.get(node_id, [])
                    parent_nodes = [pid for pid, children in relationships.items() if node_id in children]

                    topic = TopicContent(
                        node_id=node_id,
                        topic_name=topic_name,
                        node_type=node_type,
                        content=content_data.get('content', ''),
                        resources=content_data.get('resources', []),
                        parent_nodes=parent_nodes,
                        child_nodes=child_nodes,
                        position=node_info.get('position'),
                        metadata={
                            'label': node_info.get('label', ''),
                            'file_path': content_data.get('file_path', '')
                        }
                    )
                    topics.append(topic)

            # 5. Create the complete Roadmap object
            roadmap = Roadmap(
                roadmap_id=roadmap_name,
                title=overview_data['title'] or roadmap_name.replace('-', ' ').title(),
                description=overview_data['description'],
                overview=overview_data['overview'],
                topics=topics,
                node_relationships=relationships,
                skills_required=overview_data['skills'],
                metadata=overview_data['metadata']
            )

            print(f"  ✓ Successfully scraped '{roadmap_name}' with {len(topics)} topics")
            return roadmap

        except Exception as e:
            print(f"  ✗ Error scraping roadmap '{roadmap_name}': {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def scrape_all_roadmaps(self, include_content: bool = True, roadmap_filter: Optional[List[str]] = None) -> List[Roadmap]:
        """
        Scrape all available roadmaps

        Args:
            include_content: Whether to fetch detailed content (slower but more complete)
            roadmap_filter: Optional list of specific roadmaps to scrape (e.g., ['frontend', 'backend'])

        Returns:
            List of Roadmap objects
        """
        roadmaps = []

        # Use filter if provided, otherwise use all available roadmaps
        roadmap_names = roadmap_filter if roadmap_filter else self.AVAILABLE_ROADMAPS

        print(f"\n{'='*60}")
        print(f"Scraping {len(roadmap_names)} roadmaps...")
        print(f"{'='*60}\n")

        for i, name in enumerate(roadmap_names, 1):
            print(f"[{i}/{len(roadmap_names)}] ", end='')
            roadmap = self.scrape_roadmap(name, include_content=include_content)
            if roadmap:
                roadmaps.append(roadmap)
            print()  # Empty line for readability

        print(f"\n{'='*60}")
        print(f"✓ Successfully scraped {len(roadmaps)}/{len(roadmap_names)} roadmaps")
        print(f"{'='*60}\n")

        return roadmaps

    def save_to_json(self, roadmaps: List[Roadmap], output_file: str, pretty: bool = True):
        """
        Save scraped roadmaps to JSON file

        Args:
            roadmaps: List of Roadmap objects
            output_file: Path to output JSON file
            pretty: Whether to format JSON with indentation
        """
        data = []
        for roadmap in roadmaps:
            if isinstance(roadmap, Roadmap):
                data.append(asdict(roadmap))
            else:
                data.append(roadmap)

        with open(output_file, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)

        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"✓ Data saved to {output_file} ({file_size:.2f} MB)")

    def save_for_rag(self, roadmaps: List[Roadmap], output_dir: str):
        """
        Save roadmaps in RAG-optimized format

        Creates separate files optimized for RAG ingestion:
        - One file per roadmap with flattened, searchable content
        - Metadata for each topic
        - Hierarchical context preserved

        Args:
            roadmaps: List of Roadmap objects
            output_dir: Directory to save RAG-optimized files
        """
        os.makedirs(output_dir, exist_ok=True)

        for roadmap in roadmaps:
            # Create one JSON file per roadmap
            output_file = os.path.join(output_dir, f"{roadmap.roadmap_id}_rag.json")

            rag_data = {
                'roadmap_id': roadmap.roadmap_id,
                'roadmap_title': roadmap.title,
                'roadmap_description': roadmap.description,
                'roadmap_overview': roadmap.overview,
                'skills_required': roadmap.skills_required,
                'metadata': roadmap.metadata,
                'total_topics': len(roadmap.topics),
                'topics': []
            }

            # Process each topic for RAG
            for topic in roadmap.topics:
                # Create a RAG-friendly document for each topic
                topic_doc = {
                    # Identifiers
                    'id': topic.node_id,
                    'roadmap': roadmap.roadmap_id,

                    # Main content
                    'topic_name': topic.topic_name,
                    'topic_type': topic.node_type,
                    'content': topic.content,

                    # Context for better retrieval
                    'full_path': f"{roadmap.title} > {topic.topic_name}",

                    # Relationships for traversal
                    'parent_topics': topic.parent_nodes,
                    'child_topics': topic.child_nodes,

                    # Resources
                    'resources': topic.resources,

                    # Additional metadata
                    'metadata': topic.metadata,

                    # Searchable combined text (for embeddings)
                    'searchable_text': self._create_searchable_text(roadmap, topic)
                }

                rag_data['topics'].append(topic_doc)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(rag_data, f, indent=2, ensure_ascii=False)

            print(f"  ✓ Saved RAG data for '{roadmap.roadmap_id}' ({len(roadmap.topics)} topics)")

        print(f"\n✓ All RAG files saved to {output_dir}/")

    def _create_searchable_text(self, roadmap: Roadmap, topic: TopicContent) -> str:
        """Create a combined searchable text for embeddings"""
        parts = [
            f"Roadmap: {roadmap.title}",
            f"Topic: {topic.topic_name}",
            f"Type: {topic.node_type}",
        ]

        if topic.content:
            parts.append(f"Content: {topic.content}")

        if topic.resources:
            resource_names = [r['title'] for r in topic.resources if 'title' in r]
            if resource_names:
                parts.append(f"Resources: {', '.join(resource_names)}")

        return " | ".join(parts)


def main():
    """Main function to demonstrate usage"""
    # Optional: Set your GitHub token for higher API rate limits
    # Get from: https://github.com/settings/tokens
    github_token = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')

    scraper = RoadmapScraper(github_token)

    print("""
╔══════════════════════════════════════════════════════════════╗
║         Roadmap.sh Data Scraper - RAG Optimized             ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Scrape ALL roadmaps (WARNING: This takes 30-60 minutes!)
    print("\n⚠️  About to scrape ALL 66 roadmaps...")
    print("    - Estimated time: 30-60 minutes")
    print("    - API calls: ~5000+ requests")
    print("    - Requires: GitHub token for rate limits")
    print()

    if not github_token:
        print("❌ WARNING: No GITHUB_PERSONAL_ACCESS_TOKEN found!")
        print("   You may hit rate limits (60 requests/hour without auth)")
        print("   Get a token from: https://github.com/settings/tokens")
        print()
        response = input("Continue anyway? (yes/no): ").strip().lower()
        if response != 'yes':
            print("Aborted. Set GITHUB_PERSONAL_ACCESS_TOKEN and try again.")
            return
    else:
        print("✓ GitHub token found - using authenticated requests (5000/hour limit)")
        print()

    print("Starting in 3 seconds... (Ctrl+C to cancel)")
    import time
    time.sleep(3)

    all_roadmaps = scraper.scrape_all_roadmaps(include_content=True)

    # Save in both formats
    scraper.save_to_json(all_roadmaps, 'all_roadmaps_complete.json')
    scraper.save_for_rag(all_roadmaps, 'rag_data')

    # Print summary
    print(f"\nFinal Summary:")
    print(f"  - Total roadmaps scraped: {len(all_roadmaps)}")
    print(f"  - Total topics across all roadmaps: {sum(len(r.topics) for r in all_roadmaps)}")
    print(f"  - Topics with content: {sum(sum(1 for t in r.topics if t.content) for r in all_roadmaps)}")

    # Calculate total resources
    total_resources = sum(sum(len(t.resources) for t in r.topics) for r in all_roadmaps)
    print(f"  - Total learning resources: {total_resources}")
    print(f"  - Average topics per roadmap: {sum(len(r.topics) for r in all_roadmaps) // len(all_roadmaps)}")

    print("\n" + "="*60)
    print("✓ Done! Check the output files:")
    print("  - all_roadmaps_complete.json (all roadmaps, standard format)")
    print("  - rag_data/ (RAG-optimized format, one file per roadmap)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()