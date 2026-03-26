"""
=============================================================================
 GitHub Profile Analyzer Module
=============================================================================
 Role in the pipeline:
   This is the SECOND stage. It connects to the GitHub REST API v3 to fetch
   a user's public repositories, analyzes the programming languages used,
   and maps them to skills from the master skills list.

 This provides "demonstrated" skills — things the candidate has actually
 coded, as opposed to "claimed" skills from the resume.

 API Reference: https://docs.github.com/en/rest
=============================================================================
"""

import asyncio
import base64
import math
from typing import Dict, List, Optional, Tuple

import httpx
from loguru import logger

from modules.resume_parser import get_compiled_patterns
from modules.code_quality_analyzer import (
    is_code_file, should_skip_path, detect_language, analyze_code_quality,
)

# Semaphore to limit concurrent GitHub API requests
_github_semaphore = asyncio.Semaphore(10)


class GitHubAnalyzer:
    """Analyzes a GitHub user's public profile to extract demonstrated skills."""

    # Base URL for the GitHub REST API v3
    API_BASE = "https://api.github.com"

    def __init__(self, github_token: Optional[str] = None) -> None:
        """
        Initialize the analyzer with optional authentication.

        Args:
            github_token: A GitHub personal access token for higher rate limits.
                          Without token: 60 requests/hour.
                          With token:  5,000 requests/hour.
        """
        self.headers: Dict[str, str] = {
            "Accept": "application/vnd.github.v3+json",
        }

        # Add authorization header if a token is provided
        if github_token:
            self.headers["Authorization"] = f"token {github_token}"
            logger.info("[GitHubAnalyzer] Initialized with authentication token.")
        else:
            logger.info("[GitHubAnalyzer] Initialized WITHOUT token (rate-limited to 60 req/hr).")

        # Shared async HTTP client with connection pooling
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the shared async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers=self.headers,
                timeout=httpx.Timeout(15.0, connect=10.0),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._client

    async def close(self):
        """Close the async HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # -----------------------------------------------------------------
    #  Fetch User Repositories
    # -----------------------------------------------------------------
    async def get_user_repos(self, username: str) -> Tuple[List[Dict], str]:
        """Fetch all public repositories for a given GitHub username."""
        url = f"{self.API_BASE}/users/{username}/repos"
        params = {
            "per_page": 100,
            "sort": "updated",
            "type": "owner",
        }

        logger.info(f"[GitHubAnalyzer] Fetching repos for user: {username}")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                client = await self._get_client()
                response = await client.get(url, params=params)

                if response.status_code == 404:
                    logger.warning(f"[GitHubAnalyzer] User '{username}' not found (404).")
                    return [], f"GitHub user '{username}' not found."

                if response.status_code == 403:
                    logger.error("[GitHubAnalyzer] Rate limit exceeded (403).")
                    return [], "GitHub API rate limit exceeded. Try again later or add a token."

                if response.status_code != 200:
                    logger.error(f"[GitHubAnalyzer] HTTP {response.status_code}")
                    return [], f"GitHub API error: HTTP {response.status_code}"

                repos_data = response.json()

                repos = []
                for repo in repos_data:
                    repos.append({
                        "name": repo.get("name", ""),
                        "language": repo.get("language"),
                        "description": repo.get("description", ""),
                        "stargazers_count": repo.get("stargazers_count", 0),
                        "fork": repo.get("fork", False),
                        "topics": repo.get("topics", []),
                    })

                logger.info(f"[GitHubAnalyzer] Found {len(repos)} repositories.")
                return repos, ""

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                wait = 2 ** attempt
                logger.warning(f"[GitHubAnalyzer] Retry {attempt+1}/{max_retries} after {e.__class__.__name__}, waiting {wait}s...")
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait)
                else:
                    return [], f"GitHub API failed after {max_retries} retries: {e}"

    # -----------------------------------------------------------------
    #  Fetch Repository Languages (async)
    # -----------------------------------------------------------------
    async def get_repo_languages(self, username: str, repo_name: str) -> Dict[str, int]:
        """Get the byte count of each programming language in a repository."""
        url = f"{self.API_BASE}/repos/{username}/{repo_name}/languages"
        try:
            async with _github_semaphore:
                client = await self._get_client()
                response = await client.get(url)
                if response.status_code == 200:
                    return response.json()
                return {}
        except (httpx.TimeoutException, httpx.ConnectError):
            return {}

    # -----------------------------------------------------------------
    #  Fetch Repository Topics (async)
    # -----------------------------------------------------------------
    async def get_repo_topics(self, username: str, repo_name: str) -> List[str]:
        """Get the topic tags assigned to a repository."""
        url = f"{self.API_BASE}/repos/{username}/{repo_name}/topics"
        topic_headers = {"Accept": "application/vnd.github.mercy-preview+json"}
        try:
            async with _github_semaphore:
                client = await self._get_client()
                response = await client.get(url, headers=topic_headers)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("names", [])
                return []
        except (httpx.TimeoutException, httpx.ConnectError):
            return []

    # -----------------------------------------------------------------
    #  Fetch repo data concurrently (languages + topics per repo)
    # -----------------------------------------------------------------
    async def _fetch_repo_data(self, username: str, repo: Dict) -> Tuple[Dict[str, int], List[str]]:
        """Fetch languages and topics for a single repo concurrently."""
        langs, topics = await asyncio.gather(
            self.get_repo_languages(username, repo["name"]),
            self.get_repo_topics(username, repo["name"]),
        )
        return langs, topics

    # -----------------------------------------------------------------
    #  Full Profile Analysis (Main Entry Point) — now async
    # -----------------------------------------------------------------
    async def analyze_github_profile(
        self,
        username: str,
        skills_master: Dict[str, List[str]],
        analyze_code: bool = False,
    ) -> Dict:
        """
        Perform a full analysis of a GitHub user's public profile.
        Now uses async httpx with concurrent requests for massive speedup.
        """
        logger.info(f"[GitHubAnalyzer] Analyzing profile: {username}")

        # Step 1: Fetch all repos
        repos, error = await self.get_user_repos(username)

        if error:
            return {
                "repos_analyzed": 0,
                "demonstrated_skills": [],
                "skill_proficiency": {},
                "raw_languages": {},
                "raw_topics": [],
                "error": error,
            }

        # Step 2: Filter out forked repos
        original_repos = [r for r in repos if not r["fork"]]
        logger.info(f"[GitHubAnalyzer] Analyzing {len(original_repos)} original repos (skipped forks).")

        # Step 3: Fetch languages and topics for ALL repos concurrently
        repo_data_tasks = [
            self._fetch_repo_data(username, repo) for repo in original_repos
        ]
        repo_data_results = await asyncio.gather(*repo_data_tasks)

        language_bytes: Dict[str, int] = {}
        all_topics: List[str] = []

        for repo, (repo_langs, repo_topics) in zip(original_repos, repo_data_results):
            for lang, byte_count in repo_langs.items():
                language_bytes[lang] = language_bytes.get(lang, 0) + byte_count
            all_topics.extend(repo_topics)
            if repo["language"]:
                lang = repo["language"]
                if lang not in language_bytes:
                    language_bytes[lang] = 0

        all_topics = list(set(all_topics))

        logger.debug(f"[GitHubAnalyzer] Languages found: {list(language_bytes.keys())}")
        logger.debug(f"[GitHubAnalyzer] Topics found: {all_topics}")

        # Step 4: Map languages and topics to master skills
        demonstrated_skills = set()

        all_master_skills = []
        for category, skills in skills_master.items():
            all_master_skills.extend(skills)

        # Build lookup maps once (O(n) instead of O(n*m))
        skill_lower_map = {}
        skill_clean_map = {}
        for skill in all_master_skills:
            skill_lower_map[skill.lower()] = skill
            cleaned = skill.lower().replace("-", " ").replace("_", " ")
            skill_clean_map[cleaned] = skill

        # Match GitHub languages to master skills via dict lookup
        for lang in language_bytes.keys():
            match = skill_lower_map.get(lang.lower())
            if match:
                demonstrated_skills.add(match)

        # Match GitHub topics to master skills (exact cleaned match only — no substring)
        for topic in all_topics:
            topic_clean = topic.lower().replace("-", " ").replace("_", " ")
            exact = skill_clean_map.get(topic_clean)
            if exact:
                demonstrated_skills.add(exact)
            # Also try direct lowercase match
            direct = skill_lower_map.get(topic.lower())
            if direct:
                demonstrated_skills.add(direct)

        # Step 5: Calculate proficiency scores
        skill_proficiency: Dict[str, float] = {}
        if language_bytes:
            max_bytes = max(language_bytes.values(), default=1)
            if max_bytes <= 0:
                max_bytes = 1
            lang_lower_bytes = {lang.lower(): bc for lang, bc in language_bytes.items()}
            for skill in demonstrated_skills:
                skill_bytes = lang_lower_bytes.get(skill.lower(), 0)
                if skill_bytes > 0:
                    score = math.log(skill_bytes + 1) / math.log(max_bytes + 1)
                    skill_proficiency[skill] = round(score, 3)
                else:
                    skill_proficiency[skill] = 0.3

        # Step 6: Deep repo analysis — concurrent file fetches
        logger.info("[GitHubAnalyzer] Running deep analysis on top repos...")
        dep_skills = await self._deep_analyze_repos(username, original_repos[:10], all_master_skills)
        demonstrated_skills.update(dep_skills)

        # Step 7: Get commit activity stats (concurrent with nothing, but async)
        commit_stats = await self._get_commit_activity(username)

        # Step 8: Code quality analysis (optional — analyzes actual source files)
        code_quality = {}
        if analyze_code:
            try:
                code_quality = await self.analyze_code_quality_from_repos(
                    username, original_repos[:5]
                )
            except Exception as e:
                logger.warning(f"[GitHubAnalyzer] Code quality analysis failed (non-critical): {e}")

        demonstrated_list = sorted(demonstrated_skills)
        logger.info(f"[GitHubAnalyzer] Demonstrated skills: {demonstrated_list}")

        return {
            "repos_analyzed": len(original_repos),
            "demonstrated_skills": demonstrated_list,
            "skill_proficiency": skill_proficiency,
            "raw_languages": language_bytes,
            "raw_topics": all_topics,
            "commit_activity": commit_stats,
            "code_quality": code_quality,
            "total_repos": len(repos),
            "original_repos": len(original_repos),
            "error": "",
        }

    # -----------------------------------------------------------------
    #  Deep Repo Analysis — Parse dependency files (async + concurrent)
    # -----------------------------------------------------------------
    DEP_FILES = {
        "requirements.txt": "_parse_requirements_txt",
        "Pipfile": "_parse_pipfile",
        "package.json": "_parse_package_json",
        "pom.xml": "_parse_pom_xml",
        "Cargo.toml": "_parse_cargo_toml",
        "go.mod": "_parse_go_mod",
        "Gemfile": "_parse_gemfile",
        "build.gradle": "_parse_gradle",
        "docker-compose.yml": "_parse_docker_compose",
        "Dockerfile": "_parse_dockerfile",
    }

    DEP_TO_SKILL = {
        # Python
        "flask": "Flask", "django": "Django", "fastapi": "FastAPI",
        "tensorflow": "TensorFlow", "torch": "PyTorch", "pytorch": "PyTorch",
        "keras": "Keras", "scikit-learn": "Scikit-learn", "sklearn": "Scikit-learn",
        "pandas": "Pandas", "numpy": "NumPy", "scipy": "Statistics",
        "matplotlib": "Matplotlib", "plotly": "Plotly", "seaborn": "Seaborn",
        "sqlalchemy": "SQLAlchemy", "psycopg2": "PostgreSQL", "pymongo": "MongoDB",
        "redis": "Redis", "celery": "Celery", "pytest": "PyTest",
        "selenium": "Selenium", "spacy": "NLP", "nltk": "NLP",
        "transformers": "Hugging Face", "opencv": "Computer Vision",
        "mlflow": "MLflow", "airflow": "Airflow", "prefect": "Prefect",
        "dagster": "Dagster", "dbt": "dbt",
        "langchain": "LangChain", "llamaindex": "LlamaIndex",
        "chromadb": "ChromaDB", "pinecone": "Pinecone", "weaviate": "Weaviate",
        "polars": "Polars", "duckdb": "DuckDB", "dask": "Dask", "ray": "Ray",
        "uvicorn": "FastAPI",
        "great_expectations": "Great Expectations",
        # JavaScript/Node
        "react": "React", "react-dom": "React", "next": "Next.js",
        "vue": "Vue", "nuxt": "Nuxt.js", "svelte": "Svelte",
        "angular": "Angular", "@angular/core": "Angular",
        "express": "Express", "nestjs": "Nest.js", "@nestjs/core": "Nest.js",
        "typescript": "TypeScript",
        "webpack": "Webpack", "vite": "Vite", "esbuild": "esbuild",
        "jest": "Jest", "mocha": "Mocha", "cypress": "Cypress", "playwright": "Playwright",
        "mongoose": "Mongoose", "sequelize": "Sequelize", "prisma": "Prisma",
        "drizzle-orm": "Drizzle", "typeorm": "TypeORM",
        "graphql": "GraphQL", "@apollo/client": "GraphQL",
        "tailwindcss": "Tailwind CSS", "redux": "Redux",
        "@mui/material": "Material UI", "@chakra-ui/react": "Chakra UI",
        "styled-components": "Styled Components", "storybook": "Storybook",
        "three": "Three.js", "d3": "D3.js", "chart.js": "Chart.js", "recharts": "Recharts",
        "react-native": "React Native", "expo": "React Native",
        "electron": "Electron", "tauri": "Tauri",
        "remix": "Remix", "astro": "Astro", "gatsby": "Gatsby",
        # Docker/DevOps
        "docker": "Docker", "kubernetes": "Kubernetes", "k8s": "Kubernetes",
        "terraform": "Terraform", "ansible": "Ansible", "pulumi": "Pulumi",
        "jenkins": "Jenkins", "nginx": "Nginx",
        "helm": "Helm", "istio": "Istio", "argocd": "ArgoCD",
        "prometheus": "Prometheus", "grafana": "Grafana",
        "datadog": "Datadog", "sentry": "Sentry",
        # Cloud
        "boto3": "AWS", "aws-sdk": "AWS", "aws-cdk": "AWS",
        "azure": "Azure", "@azure": "Azure",
        "google-cloud": "GCP", "@google-cloud": "GCP",
        "firebase": "Firebase", "supabase": "Supabase",
        # Data/Messaging
        "kafka": "Kafka", "rabbitmq": "RabbitMQ", "amqplib": "RabbitMQ",
        "elasticsearch": "Elasticsearch",
        "snowflake-connector-python": "Snowflake",
        "clickhouse-driver": "ClickHouse",
        # Go
        "gin": "Gin", "fiber": "Fiber", "echo": "Echo",
        # Rust
        "actix": "Actix", "rocket": "Rocket",
        # Mobile
        "flutter": "Flutter", "ionic": "Ionic",
        # Blockchain
        "ethers": "Solidity", "web3": "Web3", "hardhat": "Smart Contracts",
    }

    async def _deep_analyze_repos(
        self, username: str, repos: List[Dict], master_skills: List[str]
    ) -> set:
        """Analyze top repos for dependency files and READMEs — all fetches concurrent."""
        found_skills = set()

        # Build all fetch tasks across all repos
        fetch_tasks = []
        task_metadata = []

        for repo in repos:
            repo_name = repo["name"]
            for dep_file in self.DEP_FILES:
                fetch_tasks.append(self._get_file_content(username, repo_name, dep_file))
                task_metadata.append(("dep", dep_file, repo_name))
            for readme_name in ["README.md", "readme.md", "README.rst"]:
                fetch_tasks.append(self._get_file_content(username, repo_name, readme_name))
                task_metadata.append(("readme", readme_name, repo_name))

        # Execute all fetches concurrently
        results = await asyncio.gather(*fetch_tasks)

        # Process results
        seen_readmes = set()
        for (file_type, filename, repo_name), content in zip(task_metadata, results):
            if not content:
                continue
            if file_type == "dep":
                skills = self._extract_skills_from_deps(content, filename)
                found_skills.update(skills)
            elif file_type == "readme" and repo_name not in seen_readmes:
                seen_readmes.add(repo_name)
                skills = self._extract_skills_from_readme(content, master_skills)
                found_skills.update(skills)

        logger.info(f"[GitHubAnalyzer] Deep analysis found {len(found_skills)} additional skills")
        return found_skills

    async def _get_file_content(self, username: str, repo_name: str, file_path: str) -> str:
        """Fetch raw file content from a GitHub repo (async)."""
        url = f"{self.API_BASE}/repos/{username}/{repo_name}/contents/{file_path}"
        try:
            async with _github_semaphore:
                client = await self._get_client()
                response = await client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("encoding") == "base64":
                        return base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
                return ""
        except Exception:
            return ""

    def _extract_skills_from_deps(self, content: str, filename: str) -> set:
        """Extract skills from dependency file content."""
        found = set()
        content_lower = content.lower()

        for dep_name, skill_name in self.DEP_TO_SKILL.items():
            if dep_name.lower() in content_lower:
                found.add(skill_name)

        if filename == "Dockerfile":
            found.add("Docker")
            if "python" in content_lower:
                found.add("Python")
            if "node" in content_lower:
                found.add("Node.js")
            if "nginx" in content_lower:
                found.add("Linux")

        if filename == "docker-compose.yml":
            found.add("Docker")
            if "postgres" in content_lower:
                found.add("PostgreSQL")
            if "redis" in content_lower:
                found.add("Redis")
            if "mongo" in content_lower:
                found.add("MongoDB")

        return found

    def _extract_skills_from_readme(
        self, readme: str, master_skills: List[str]
    ) -> set:
        """Extract skill mentions from README content."""
        import re
        found = set()
        readme_lower = readme.lower()

        patterns = get_compiled_patterns()
        if patterns:
            for skill in master_skills:
                pat = patterns.get(skill)
                if pat and pat.search(readme_lower):
                    found.add(skill)
        else:
            for skill in master_skills:
                pattern = r"\b" + re.escape(skill.lower()) + r"\b"
                if skill in ("C++", "C#"):
                    pattern = re.escape(skill.lower())
                if re.search(pattern, readme_lower):
                    found.add(skill)

        return found

    # -----------------------------------------------------------------
    #  Code Quality Analysis — Fetch and analyze actual source files
    # -----------------------------------------------------------------
    async def get_repo_tree(self, username: str, repo_name: str) -> List[Dict]:
        """Fetch the recursive file tree for a repository."""
        url = f"{self.API_BASE}/repos/{username}/{repo_name}/git/trees/HEAD"
        try:
            async with _github_semaphore:
                client = await self._get_client()
                response = await client.get(url, params={"recursive": "1"})
                if response.status_code == 200:
                    data = response.json()
                    return data.get("tree", [])
                return []
        except (httpx.TimeoutException, httpx.ConnectError):
            return []

    def _select_analysis_files(self, tree: List[Dict], max_files: int = 3) -> List[str]:
        """Select the best source code files from a repo tree for quality analysis."""
        candidates = []
        for item in tree:
            if item.get("type") != "blob":
                continue
            path = item.get("path", "")
            size = item.get("size", 0)
            if not is_code_file(path):
                continue
            if should_skip_path(path):
                continue
            if size < 200 or size > 50000:  # skip tiny or huge files
                continue
            # Prefer files in src/app/lib directories
            priority = 0
            path_lower = path.lower()
            if any(d in path_lower for d in ["src/", "app/", "lib/", "core/", "api/"]):
                priority += 2
            if size > 1000:
                priority += 1
            if size > 3000:
                priority += 1
            candidates.append((path, size, priority))

        # Sort by priority desc, then size desc
        candidates.sort(key=lambda x: (-x[2], -x[1]))
        return [c[0] for c in candidates[:max_files]]

    async def analyze_code_quality_from_repos(
        self, username: str, repos: List[Dict], max_repos: int = 5, max_files_per_repo: int = 3,
    ) -> Dict:
        """
        Fetch and analyze actual source code files from top repos.
        Returns aggregated code quality scores.
        """
        logger.info(f"[GitHubAnalyzer] Starting code quality analysis for {username}")

        all_file_results = []
        total_scores = {"speed": 0, "complexity": 0, "flexibility": 0,
                        "code_quality": 0, "best_practices": 0}
        valid_count = 0

        # Fetch trees for top repos concurrently
        selected_repos = repos[:max_repos]
        tree_tasks = [self.get_repo_tree(username, r["name"]) for r in selected_repos]
        trees = await asyncio.gather(*tree_tasks)

        # For each repo, select files and fetch content
        fetch_tasks = []
        fetch_meta = []  # (repo_name, file_path)
        for repo, tree in zip(selected_repos, trees):
            if not tree:
                continue
            files = self._select_analysis_files(tree, max_files=max_files_per_repo)
            for file_path in files:
                fetch_tasks.append(self._get_file_content(username, repo["name"], file_path))
                fetch_meta.append((repo["name"], file_path))

        if not fetch_tasks:
            logger.info("[GitHubAnalyzer] No analyzable source files found")
            return {}

        # Fetch all file contents concurrently (cap at 15)
        fetch_tasks = fetch_tasks[:15]
        fetch_meta = fetch_meta[:15]
        file_contents = await asyncio.gather(*fetch_tasks)

        # Analyze each file with LLM
        for (repo_name, file_path), content in zip(fetch_meta, file_contents):
            if not content or len(content.strip()) < 100:
                continue
            language = detect_language(file_path)
            result = analyze_code_quality(
                code=content,
                language=language,
                context="github",
                filename=f"{repo_name}/{file_path}",
            )
            if result:
                all_file_results.append({
                    "repo": repo_name,
                    "file": file_path,
                    "language": language,
                    "scores": result,
                })
                for dim in total_scores:
                    total_scores[dim] += result.get(dim, {}).get("score", 0)
                valid_count += 1

        if valid_count == 0:
            logger.info("[GitHubAnalyzer] No files could be analyzed for code quality")
            return {}

        # Compute aggregate scores
        aggregate = {}
        for dim in total_scores:
            avg = round(total_scores[dim] / valid_count, 1)
            aggregate[dim] = {"score": avg, "notes": f"Average across {valid_count} files"}

        dim_scores = [aggregate[d]["score"] for d in total_scores]
        aggregate["overall_score"] = round(sum(dim_scores) / len(dim_scores), 1)
        aggregate["files_analyzed"] = valid_count
        aggregate["per_file"] = all_file_results

        logger.info(f"[GitHubAnalyzer] Code quality analysis complete — "
                    f"{valid_count} files, overall: {aggregate['overall_score']}/10")

        return aggregate

    # -----------------------------------------------------------------
    #  Commit Activity Analysis (async)
    # -----------------------------------------------------------------
    async def _get_commit_activity(self, username: str) -> Dict:
        """Get commit activity stats for the user."""
        url = f"{self.API_BASE}/users/{username}/events/public"
        try:
            client = await self._get_client()
            response = await client.get(url, params={"per_page": 100})
            if response.status_code != 200:
                return {"total_recent_events": 0, "push_events": 0, "active_days": 0}

            events = response.json()
            push_events = [e for e in events if e.get("type") == "PushEvent"]
            unique_dates = set()
            total_commits = 0
            for event in push_events:
                date = event.get("created_at", "")[:10]
                if date:
                    unique_dates.add(date)
                commits = event.get("payload", {}).get("commits", [])
                total_commits += len(commits)

            return {
                "total_recent_events": len(events),
                "push_events": len(push_events),
                "total_commits": total_commits,
                "active_days": len(unique_dates),
            }
        except Exception:
            return {"total_recent_events": 0, "push_events": 0, "active_days": 0}
