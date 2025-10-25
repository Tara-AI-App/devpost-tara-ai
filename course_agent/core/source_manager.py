"""
Source manager for orchestrating content discovery across different sources.
"""
import asyncio
from typing import List, Dict, Any, Optional
from ..tools import SourceResult, SearchQuery, SourceType
from ..tools.rag_tool import RAGTool
from ..tools.github_tool import GitHubMCPTool
from ..tools.search_tool import GoogleSearchTool
from ..config.settings import settings, SourcePriority
from ..utils.logger import logger


class SourceManager:
    """Manages content discovery across different sources."""

    def __init__(self):
        self.rag_tool = RAGTool()
        self.github_tool = GitHubMCPTool()
        self.search_tool = GoogleSearchTool()
        self._source_priority = settings.source_priority

    async def discover_content(self, topic: str) -> Dict[str, List[SourceResult]]:
        """
        Discover content for a topic using the configured priority strategy.

        Returns:
            Dict with keys: 'rag_results', 'github_results', 'used_sources'
        """
        logger.info("=" * 80)
        logger.info(f"STARTING CONTENT DISCOVERY")
        logger.info(f"Topic: {topic}")
        logger.info(f"Source Priority: {self._source_priority.value}")
        logger.info("=" * 80)

        # Initialize results
        rag_results: List[SourceResult] = []
        github_results: List[SourceResult] = []
        search_results: List[SourceResult] = []
        used_sources: List[str] = []

        if self._source_priority == SourcePriority.RAG_FIRST:
            rag_results, github_results, used_sources = await self._rag_first_strategy(topic)
        elif self._source_priority == SourcePriority.GITHUB_FIRST:
            github_results, rag_results, used_sources = await self._github_first_strategy(topic)
        else:  # BALANCED
            rag_results, github_results, used_sources = await self._balanced_strategy(topic)

        # Fallback to Google Search if insufficient results from primary sources
        total_primary_results = len(rag_results) + len(github_results)
        if total_primary_results < 3:  # Minimum threshold for sufficient content
            logger.info("Insufficient results from primary sources, falling back to Google Search")
            search_results = await self._search_web(topic)
            if search_results:
                used_sources.append("Google Search")

        # Log final summary
        logger.info("=" * 80)
        logger.info(f"CONTENT DISCOVERY COMPLETED")
        logger.info(f"Sources used: {used_sources}")
        logger.info(f"RAG results: {len(rag_results)}")
        logger.info(f"GitHub results: {len(github_results)}")
        logger.info(f"Search results: {len(search_results)}")
        logger.info(f"Total results: {len(rag_results) + len(github_results) + len(search_results)}")
        logger.info("=" * 80)

        return {
            'rag_results': rag_results,
            'github_results': github_results,
            'search_results': search_results,
            'used_sources': used_sources,
            'total_results': len(rag_results) + len(github_results) + len(search_results)
        }

    async def _rag_first_strategy(self, topic: str) -> tuple[List[SourceResult], List[SourceResult], List[str]]:
        """RAG-first content discovery strategy with parallel execution."""
        rag_results = []
        github_results = []
        used_sources = []

        # Run RAG and GitHub searches in parallel for faster discovery
        tasks = []

        if self.rag_tool.is_available():
            tasks.append(self._search_rag_async(topic))

        # Always search GitHub in parallel (don't wait for RAG sufficiency check)
        if self.github_tool.is_available():
            tasks.append(self._search_github(topic))

        # Execute in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Search task {i} failed: {result}")
                    continue

                # First result is RAG (if available)
                if i == 0 and self.rag_tool.is_available():
                    rag_results = result if isinstance(result, list) else []
                    if rag_results:
                        used_sources.append("RAG")
                # Second result is GitHub (if both available) or first if only GitHub
                elif (i == 1 and self.rag_tool.is_available()) or (i == 0 and not self.rag_tool.is_available()):
                    github_results = result if isinstance(result, list) else []
                    if github_results:
                        used_sources.append("GitHub")

        return rag_results, github_results, used_sources

    async def _search_rag_async(self, topic: str) -> List[SourceResult]:
        """Async wrapper for RAG search."""
        try:
            query = SearchQuery(query=topic, max_results=settings.rag.max_results)
            return await self.rag_tool.search(query)
        except Exception as e:
            logger.warning(f"RAG search failed: {e}")
            return []

    async def _github_first_strategy(self, topic: str) -> tuple[List[SourceResult], List[SourceResult], List[str]]:
        """GitHub-first content discovery strategy with parallel execution."""
        github_results = []
        rag_results = []
        used_sources = []

        # Run both in parallel
        tasks = []

        if self.github_tool.is_available():
            tasks.append(self._search_github(topic))

        if self.rag_tool.is_available():
            tasks.append(self._search_rag_async(topic))

        # Execute in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Search task {i} failed: {result}")
                    continue

                # First result is GitHub (if available)
                if i == 0 and self.github_tool.is_available():
                    github_results = result if isinstance(result, list) else []
                    if github_results:
                        used_sources.append("GitHub")
                # Second result is RAG (if both available) or first if only RAG
                elif (i == 1 and self.github_tool.is_available()) or (i == 0 and not self.github_tool.is_available()):
                    rag_results = result if isinstance(result, list) else []
                    if rag_results:
                        used_sources.append("RAG")

        return rag_results, github_results, used_sources

    async def _balanced_strategy(self, topic: str) -> tuple[List[SourceResult], List[SourceResult], List[str]]:
        """Balanced content discovery strategy with parallel execution."""
        rag_results = []
        github_results = []
        used_sources = []

        # Search both sources concurrently in parallel
        tasks = []

        if self.rag_tool.is_available():
            async def search_rag_balanced():
                try:
                    query = SearchQuery(query=topic, max_results=settings.rag.max_results // 2)
                    return await self.rag_tool.search(query)
                except Exception as e:
                    logger.warning(f"RAG search failed: {e}")
                    return []

            tasks.append(search_rag_balanced())

        if self.github_tool.is_available():
            tasks.append(self._search_github(topic))

        # Execute in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Search task {i} failed: {result}")
                    continue

                # First result is RAG (if available)
                if i == 0 and self.rag_tool.is_available():
                    rag_results = result if isinstance(result, list) else []
                    if rag_results:
                        used_sources.append("RAG")
                # Second result is GitHub (if both available) or first if only GitHub
                elif (i == 1 and self.rag_tool.is_available()) or (i == 0 and not self.rag_tool.is_available()):
                    github_results = result if isinstance(result, list) else []
                    if github_results:
                        used_sources.append("GitHub")

        return rag_results, github_results, used_sources

    async def _search_github(self, topic: str) -> List[SourceResult]:
        """Search GitHub repositories for the topic, prioritizing the authenticated user's repositories."""
        logger.info("-" * 80)
        logger.info(f"GITHUB SEARCH STARTING")
        logger.info(f"Topic: {topic}")

        if not self.github_tool.is_available():
            logger.warning("GitHub tools not available")
            return []

        try:
            # Try to get authenticated user and their repositories for context
            # The agent can call get_me + search_repositories manually later
            username = None
            user_repos = []

            logger.info("ℹ️  Note: Agent should call get_me + search_repositories('user:username') directly")
            logger.info("ℹ️  Automatic get_me call removed - agent will handle it via MCP tools")
            # The agent instructions now MANDATE calling get_me and search_repositories
            # This programmatic approach was unreliable, so we rely on the agent doing it

            # Extract potential repository name from topic with multiple strategies
            logger.info(f"Extracting repository name from topic...")
            topic_lower = topic.lower()
            words = topic_lower.split()

            # Common words to ignore when extracting repo name
            ignore_words = {'about', 'project', 'repository', 'repo', 'make', 'create', 'generate',
                           'course', 'the', 'a', 'an', 'for', 'on', 'in', 'of', 'my', 'your', 'want',
                           'know', 'to', 'me', 'can', 'you', 'i', 'help', 'learn', 'from'}

            # Extract potential repository name (words that aren't common filler words)
            potential_repo_names = [word for word in words if word not in ignore_words and len(word) > 2]
            logger.info(f"Potential repo names extracted: {potential_repo_names}")

            # Generate multiple search variations
            search_variations = []
            if len(potential_repo_names) > 0:
                # For single word (like "bytesv2"), use it directly first
                if len(potential_repo_names) == 1:
                    single_word = potential_repo_names[0]
                    search_variations.append(single_word)
                    # Also add common variations for single words
                    search_variations.append(single_word + '-api')
                    search_variations.append(single_word + '-app')
                    search_variations.append(single_word + '-project')
                else:
                    # For multiple words, generate variations
                    # Variation 1: Join with hyphens (capstone-seis-flask)
                    hyphenated = '-'.join(potential_repo_names)
                    search_variations.append(hyphenated)

                    # Variation 2: Join with underscores (capstone_seis_flask)
                    underscored = '_'.join(potential_repo_names)
                    search_variations.append(underscored)

                    # Variation 3: Concatenated (capstoneseis flask)
                    concatenated = ''.join(potential_repo_names)
                    search_variations.append(concatenated)

                    # Variation 4: Space-separated (capstone seis flask)
                    space_separated = ' '.join(potential_repo_names)
                    search_variations.append(space_separated)

            logger.info(f"Generated search variations: {search_variations[:5]}")  # Show first 5

            # Simplified search: Don't do automatic searching here
            # The agent MUST call get_me + search_repositories directly as per instructions
            # This automatic search is kept minimal as a fallback only

            logger.info("⚠️  Skipping automatic GitHub search")
            logger.info("→ Agent MUST call get_me + search_repositories('user:username') per instructions")
            logger.info("→ Then agent will match user query against the repo list")

            # Return empty - agent will handle the search properly via MCP tools
            github_results = []

            logger.info(f"✓ Search completed: Found {len(github_results)} repositories")
            if len(github_results) == 0:
                logger.warning("⚠ 0 repositories found - Agent should call get_me + search_repositories manually")
            else:
                for i, result in enumerate(github_results, 1):
                    logger.info(f"  {i}. {result.repository}")
            logger.info("-" * 80)

            return github_results

        except Exception as e:
            logger.error(f"✗ GitHub search failed: {e}")
            logger.info("-" * 80)
            return []

    async def get_repository_content(self, repository: str, file_patterns: List[str]) -> Dict[str, str]:
        """Get specific file contents from a repository in parallel."""
        if not self.github_tool.is_available():
            return {}

        # Fetch all files in parallel for faster extraction
        async def fetch_file(pattern: str) -> tuple[str, str]:
            try:
                file_content = await self.github_tool.get_file_contents(repository, pattern)
                return (pattern, file_content)
            except Exception as e:
                logger.warning(f"Failed to get {pattern} from {repository}: {e}")
                return (pattern, "")

        tasks = [fetch_file(pattern) for pattern in file_patterns]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        content = {}
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"File fetch error: {result}")
                continue
            if isinstance(result, tuple) and len(result) == 2:
                pattern, file_content = result
                if file_content:  # Only add non-empty content
                    content[pattern] = file_content

        return content

    async def search_code_in_repositories(self, query: str, repositories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search for specific code patterns across repositories in parallel."""
        if not self.github_tool.is_available():
            return []

        all_results = []

        if repositories:
            # Search all repositories in parallel
            async def search_in_repo(repo: str) -> List[Dict[str, Any]]:
                try:
                    return await self.github_tool.search_code(query, repo)
                except Exception as e:
                    logger.warning(f"Code search failed in {repo}: {e}")
                    return []

            tasks = [search_in_repo(repo) for repo in repositories]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            for results in results_list:
                if isinstance(results, Exception):
                    logger.warning(f"Code search error: {results}")
                    continue
                if isinstance(results, list):
                    all_results.extend(results)
        else:
            try:
                results = await self.github_tool.search_code(query)
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Global code search failed: {e}")

        return all_results

    async def _search_web(self, topic: str) -> List[SourceResult]:
        """Search web for the topic as a fallback."""
        if not self.search_tool.is_available():
            logger.warning("Google Search tools not available")
            return []

        try:
            # Search for web content
            query = SearchQuery(query=topic, max_results=settings.mcp.max_repositories)
            search_results = await self.search_tool.search(query)

            logger.info(f"Found {len(search_results)} web search results for topic: {topic}")
            return search_results

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []