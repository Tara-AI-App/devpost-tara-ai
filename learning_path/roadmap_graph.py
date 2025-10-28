#!/usr/bin/env python3
"""
Roadmap Graph Implementation using NetworkX
Provides graph-based learning path navigation for roadmap.sh data
"""

import json
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pickle


class RoadmapGraph:
    """
    Graph-based learning path manager for roadmap.sh data.

    Uses NetworkX to build directed acyclic graphs (DAGs) from roadmap data,
    enabling efficient prerequisite tracking and learning path discovery.
    """

    def __init__(self, rag_dir: str = "rag_data", cache_path: Optional[str] = None):
        """
        Initialize roadmap graphs from RAG data.

        Args:
            rag_dir: Directory containing *_rag.json files
            cache_path: Optional path to cached graphs (pickle file)
        """
        self.rag_dir = Path(rag_dir)
        self.cache_path = Path(cache_path) if cache_path else None

        # Dictionary of graphs: {roadmap_id: NetworkX DiGraph}
        self.graphs: Dict[str, nx.DiGraph] = {}

        # Dictionary of topic lookup: {roadmap_id: {topic_name_lower: node_id}}
        self.topic_index: Dict[str, Dict[str, str]] = {}

        # Load graphs
        self._load_graphs()

    def _load_graphs(self):
        """Load graphs from cache or build from RAG data."""
        # Try loading from cache first
        if self.cache_path and self.cache_path.exists():
            try:
                self._load_from_cache()
                print(f"✓ Loaded {len(self.graphs)} roadmap graphs from cache")
                return
            except Exception as e:
                print(f"⚠️  Cache load failed: {e}, rebuilding from RAG data...")

        # Build from RAG data
        self._build_from_rag_data()

        # Save to cache if path provided
        if self.cache_path:
            try:
                self._save_to_cache()
                print(f"✓ Saved graphs to cache: {self.cache_path}")
            except Exception as e:
                print(f"⚠️  Cache save failed: {e}")

    def _build_from_rag_data(self):
        """Build graphs from RAG JSON files."""
        if not self.rag_dir.exists():
            raise FileNotFoundError(f"RAG directory not found: {self.rag_dir}")

        rag_files = list(self.rag_dir.glob("*_rag.json"))
        if not rag_files:
            raise FileNotFoundError(f"No RAG files found in {self.rag_dir}")

        print(f"Building graphs from {len(rag_files)} roadmap files...")

        for file_path in rag_files:
            roadmap_id = file_path.stem.replace('_rag', '')

            with open(file_path, 'r', encoding='utf-8') as f:
                roadmap_data = json.load(f)

            # Create directed graph
            G = nx.DiGraph()
            topic_lookup = {}

            # Add all topics as nodes
            for topic in roadmap_data['topics']:
                node_id = f"{roadmap_id}:{topic['id']}"

                # Add node with attributes
                G.add_node(
                    node_id,
                    topic_id=topic['id'],
                    topic_name=topic['topic_name'],
                    topic_type=topic['topic_type'],
                    content=topic['content'],
                    resources=topic['resources'],
                    full_path=topic['full_path'],
                    roadmap=roadmap_id
                )

                # Build topic name index (lowercase for matching)
                topic_name_lower = topic['topic_name'].lower()
                topic_lookup[topic_name_lower] = node_id

            # Add edges (parent → child represents prerequisite relationship)
            for topic in roadmap_data['topics']:
                node_id = f"{roadmap_id}:{topic['id']}"

                # Add edges from parents to this node
                for parent_id in topic['parent_topics']:
                    parent_node = f"{roadmap_id}:{parent_id}"
                    if G.has_node(parent_node):
                        G.add_edge(parent_node, node_id)

            # Store graph and index
            self.graphs[roadmap_id] = G
            self.topic_index[roadmap_id] = topic_lookup

            print(f"  ✓ {roadmap_id}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        print(f"✓ Built {len(self.graphs)} roadmap graphs")

    def _save_to_cache(self):
        """Save graphs to pickle cache."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        cache_data = {
            'graphs': self.graphs,
            'topic_index': self.topic_index
        }

        with open(self.cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

    def _load_from_cache(self):
        """Load graphs from pickle cache."""
        with open(self.cache_path, 'rb') as f:
            cache_data = pickle.load(f)

        self.graphs = cache_data['graphs']
        self.topic_index = cache_data['topic_index']

    def _find_node(self, topic_name: str, roadmap_id: Optional[str] = None) -> Optional[str]:
        """
        Find a node by topic name.

        Args:
            topic_name: Name of the topic (case-insensitive)
            roadmap_id: Optional roadmap to search in (searches all if None)

        Returns:
            Node ID if found, None otherwise
        """
        topic_lower = topic_name.lower()

        if roadmap_id:
            # Search in specific roadmap
            if roadmap_id in self.topic_index:
                return self.topic_index[roadmap_id].get(topic_lower)
            return None
        else:
            # Search all roadmaps (return first match)
            for rm_id, lookup in self.topic_index.items():
                if topic_lower in lookup:
                    return lookup[topic_lower]
            return None

    def get_prerequisites(self, topic_name: str, roadmap_id: Optional[str] = None) -> List[Dict]:
        """
        Get ALL prerequisites for a topic (all ancestor nodes).

        Args:
            topic_name: Name of the topic
            roadmap_id: Optional roadmap filter

        Returns:
            List of prerequisite topics with details
        """
        node_id = self._find_node(topic_name, roadmap_id)
        if not node_id:
            return []

        roadmap = node_id.split(':')[0]
        graph = self.graphs[roadmap]

        # Get all ancestors (prerequisites)
        ancestors = nx.ancestors(graph, node_id)

        prerequisites = []
        for ancestor_id in ancestors:
            node_data = graph.nodes[ancestor_id]
            prerequisites.append({
                'topic_name': node_data['topic_name'],
                'topic_id': node_data['topic_id'],
                'roadmap': node_data['roadmap'],
                'full_path': node_data['full_path'],
                'resources': node_data['resources']
            })

        return prerequisites

    def get_next_topics(self, topic_name: str, roadmap_id: Optional[str] = None) -> List[Dict]:
        """
        Get immediate next topics (direct children).

        Args:
            topic_name: Name of the current topic
            roadmap_id: Optional roadmap filter

        Returns:
            List of next topics with details
        """
        node_id = self._find_node(topic_name, roadmap_id)
        if not node_id:
            return []

        roadmap = node_id.split(':')[0]
        graph = self.graphs[roadmap]

        # Get direct successors (next topics)
        successors = list(graph.successors(node_id))

        next_topics = []
        for successor_id in successors:
            node_data = graph.nodes[successor_id]
            next_topics.append({
                'topic_name': node_data['topic_name'],
                'topic_id': node_data['topic_id'],
                'roadmap': node_data['roadmap'],
                'full_path': node_data['full_path'],
                'resources': node_data['resources']
            })

        return next_topics

    def get_learning_path(self, from_topic: str, to_topic: str, roadmap_id: Optional[str] = None) -> List[Dict]:
        """
        Find optimal learning path between two topics.

        Args:
            from_topic: Starting topic name
            to_topic: Target topic name
            roadmap_id: Optional roadmap filter (both topics must be in same roadmap)

        Returns:
            List of topics in order from start to end
        """
        from_node = self._find_node(from_topic, roadmap_id)
        to_node = self._find_node(to_topic, roadmap_id)

        if not from_node or not to_node:
            return []

        # Must be in same roadmap
        from_roadmap = from_node.split(':')[0]
        to_roadmap = to_node.split(':')[0]

        if from_roadmap != to_roadmap:
            return []

        graph = self.graphs[from_roadmap]

        try:
            # Find shortest path
            path = nx.shortest_path(graph, from_node, to_node)

            learning_path = []
            for node_id in path:
                node_data = graph.nodes[node_id]
                learning_path.append({
                    'topic_name': node_data['topic_name'],
                    'topic_id': node_data['topic_id'],
                    'roadmap': node_data['roadmap'],
                    'full_path': node_data['full_path'],
                    'resources': node_data['resources']
                })

            return learning_path
        except nx.NetworkXNoPath:
            # No path exists
            return []

    def get_learning_order(self, roadmap_id: str, start_topics: Optional[List[str]] = None) -> List[Dict]:
        """
        Get proper learning order for a roadmap using topological sort.

        Args:
            roadmap_id: Roadmap to get order for
            start_topics: Optional list of starting topics (filters to descendants only)

        Returns:
            List of topics in proper learning order
        """
        if roadmap_id not in self.graphs:
            return []

        graph = self.graphs[roadmap_id]

        # If start topics provided, filter to only descendants
        if start_topics:
            # Find nodes for start topics
            start_nodes = set()
            for topic_name in start_topics:
                node_id = self._find_node(topic_name, roadmap_id)
                if node_id:
                    start_nodes.add(node_id)

            # Get all descendants (topics that come after start topics)
            relevant_nodes = set()
            for start_node in start_nodes:
                relevant_nodes.add(start_node)
                relevant_nodes.update(nx.descendants(graph, start_node))

            # Create subgraph
            subgraph = graph.subgraph(relevant_nodes)
        else:
            subgraph = graph

        # Topological sort
        try:
            ordered_nodes = list(nx.topological_sort(subgraph))
        except nx.NetworkXError:
            # Graph has cycles (shouldn't happen with DAG)
            ordered_nodes = list(subgraph.nodes())

        learning_order = []
        for node_id in ordered_nodes:
            node_data = graph.nodes[node_id]
            learning_order.append({
                'topic_name': node_data['topic_name'],
                'topic_id': node_data['topic_id'],
                'roadmap': node_data['roadmap'],
                'full_path': node_data['full_path'],
                'topic_type': node_data['topic_type'],
                'resources': node_data['resources']
            })

        return learning_order

    def find_common_prerequisites(self, topics: List[str], roadmap_id: Optional[str] = None) -> List[Dict]:
        """
        Find common prerequisites for multiple topics.

        Args:
            topics: List of topic names
            roadmap_id: Optional roadmap filter

        Returns:
            List of common prerequisite topics
        """
        if not topics:
            return []

        # Get prerequisites for each topic
        all_prereqs = []
        for topic in topics:
            prereqs = self.get_prerequisites(topic, roadmap_id)
            prereq_names = {p['topic_name'] for p in prereqs}
            all_prereqs.append(prereq_names)

        # Find intersection (common prerequisites)
        if not all_prereqs:
            return []

        common_names = set.intersection(*all_prereqs)

        # Get full details for common prerequisites
        common_prereqs = []
        for topic_name in common_names:
            node_id = self._find_node(topic_name, roadmap_id)
            if node_id:
                roadmap = node_id.split(':')[0]
                graph = self.graphs[roadmap]
                node_data = graph.nodes[node_id]
                common_prereqs.append({
                    'topic_name': node_data['topic_name'],
                    'topic_id': node_data['topic_id'],
                    'roadmap': node_data['roadmap'],
                    'full_path': node_data['full_path'],
                    'resources': node_data['resources']
                })

        return common_prereqs

    def get_roadmap_stats(self, roadmap_id: str) -> Dict:
        """
        Get statistics for a roadmap.

        Args:
            roadmap_id: Roadmap to analyze

        Returns:
            Dictionary with statistics
        """
        if roadmap_id not in self.graphs:
            return {}

        graph = self.graphs[roadmap_id]

        # Find root nodes (no predecessors - starting points)
        root_nodes = [n for n in graph.nodes() if graph.in_degree(n) == 0]

        # Find leaf nodes (no successors - end points)
        leaf_nodes = [n for n in graph.nodes() if graph.out_degree(n) == 0]

        # Calculate average path length
        try:
            avg_path_length = nx.average_shortest_path_length(graph)
        except:
            avg_path_length = 0

        # Count topics with resources
        topics_with_resources = sum(
            1 for node_id in graph.nodes()
            if len(graph.nodes[node_id]['resources']) > 0
        )

        return {
            'roadmap_id': roadmap_id,
            'total_topics': graph.number_of_nodes(),
            'total_connections': graph.number_of_edges(),
            'root_topics': len(root_nodes),
            'leaf_topics': len(leaf_nodes),
            'average_path_length': round(avg_path_length, 2),
            'topics_with_resources': topics_with_resources,
            'is_dag': nx.is_directed_acyclic_graph(graph)
        }

    def search_topics(self, keyword: str, roadmap_id: Optional[str] = None) -> List[Dict]:
        """
        Search for topics by keyword in name or content.

        Args:
            keyword: Keyword to search for
            roadmap_id: Optional roadmap filter

        Returns:
            List of matching topics with details
        """
        keyword_lower = keyword.lower()
        matching_topics = []

        roadmaps_to_search = [roadmap_id] if roadmap_id else list(self.graphs.keys())

        for rm_id in roadmaps_to_search:
            if rm_id not in self.graphs:
                continue

            graph = self.graphs[rm_id]

            for node_id in graph.nodes():
                node_data = graph.nodes[node_id]

                # Search in topic name and content
                if (keyword_lower in node_data['topic_name'].lower() or
                    keyword_lower in node_data.get('content', '').lower()):

                    matching_topics.append({
                        'topic_name': node_data['topic_name'],
                        'topic_id': node_data['topic_id'],
                        'roadmap': node_data['roadmap'],
                        'full_path': node_data['full_path'],
                        'topic_type': node_data['topic_type'],
                        'resources': node_data['resources'],
                        'content_preview': node_data.get('content', '')[:200]
                    })

        return matching_topics

    def list_roadmaps(self) -> List[Dict]:
        """
        List all available roadmaps with statistics.

        Returns:
            List of roadmap information
        """
        roadmaps = []
        for roadmap_id in self.graphs.keys():
            stats = self.get_roadmap_stats(roadmap_id)
            roadmaps.append({
                'roadmap_id': roadmap_id,
                'title': roadmap_id.replace('-', ' ').title(),
                'stats': stats
            })

        return roadmaps


# Quick test
if __name__ == "__main__":
    import os

    # Determine correct path
    script_dir = Path(__file__).parent
    rag_dir = script_dir / "rag_data"
    cache_dir = script_dir / "cache"
    cache_path = cache_dir / "roadmap_graphs.pkl"

    print("Initializing Roadmap Graph...")
    print(f"RAG dir: {rag_dir}")
    print(f"Cache path: {cache_path}")

    # Initialize graph
    rg = RoadmapGraph(str(rag_dir), str(cache_path))

    # Show available roadmaps
    print("\n" + "="*70)
    print("Available Roadmaps:")
    print("="*70)
    roadmaps = rg.list_roadmaps()
    for rm in roadmaps:
        print(f"\n{rm['roadmap_id']}")
        stats = rm['stats']
        print(f"  Topics: {stats['total_topics']}")
        print(f"  Connections: {stats['total_connections']}")
        print(f"  Starting points: {stats['root_topics']}")

    # Test search
    print("\n" + "="*70)
    print("Search Test: 'react'")
    print("="*70)
    results = rg.search_topics('react')
    for i, topic in enumerate(results[:5], 1):
        print(f"{i}. [{topic['roadmap']}] {topic['topic_name']}")

    print("\n✓ Roadmap Graph test complete!")
