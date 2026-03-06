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

import math
from typing import Dict, List, Optional, Tuple

import requests


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
            print("   [GitHubAnalyzer] Initialized with authentication token.")
        else:
            print("   [GitHubAnalyzer] Initialized WITHOUT token (rate-limited to 60 req/hr).")

    # -----------------------------------------------------------------
    #  Fetch User Repositories
    # -----------------------------------------------------------------
    def get_user_repos(self, username: str) -> Tuple[List[Dict], str]:
        """
        Fetch all public repositories for a given GitHub username.

        Args:
            username: The GitHub username to look up.

        Returns:
            A tuple of (repo_list, error_message).
            repo_list contains dicts with: name, language, description,
            stargazers_count, fork, topics.
            error_message is empty string on success.
        """
        url = f"{self.API_BASE}/users/{username}/repos"
        params = {
            "per_page": 100,   # Max allowed per page
            "sort": "updated",  # Most recently updated first
            "type": "owner",    # Only repos owned by the user (not forks by default)
        }

        print(f"   [GitHubAnalyzer] Fetching repos for user: {username}")

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=15)

            # --- Handle HTTP error codes ---
            if response.status_code == 404:
                print(f"   [GitHubAnalyzer] ERROR: User '{username}' not found (404).")
                return [], f"GitHub user '{username}' not found."

            if response.status_code == 403:
                print("   [GitHubAnalyzer] ERROR: Rate limit exceeded (403).")
                return [], "GitHub API rate limit exceeded. Try again later or add a token."

            if response.status_code != 200:
                print(f"   [GitHubAnalyzer] ERROR: HTTP {response.status_code}")
                return [], f"GitHub API error: HTTP {response.status_code}"

            repos_data = response.json()

            # Extract the fields we care about from each repo
            repos = []
            for repo in repos_data:
                repos.append({
                    "name": repo.get("name", ""),
                    "language": repo.get("language"),         # Primary language
                    "description": repo.get("description", ""),
                    "stargazers_count": repo.get("stargazers_count", 0),
                    "fork": repo.get("fork", False),
                    "topics": repo.get("topics", []),
                })

            print(f"   [GitHubAnalyzer] Found {len(repos)} repositories.")
            return repos, ""

        except requests.exceptions.Timeout:
            print("   [GitHubAnalyzer] ERROR: Request timed out.")
            return [], "GitHub API request timed out."
        except requests.exceptions.ConnectionError:
            print("   [GitHubAnalyzer] ERROR: Connection failed.")
            return [], "Failed to connect to GitHub API."

    # -----------------------------------------------------------------
    #  Fetch Repository Languages
    # -----------------------------------------------------------------
    def get_repo_languages(self, username: str, repo_name: str) -> Dict[str, int]:
        """
        Get the byte count of each programming language in a repository.

        GitHub returns a dict like {"Python": 12400, "JavaScript": 3200}
        where values are bytes of code in that language.

        Args:
            username:  The GitHub username.
            repo_name: The repository name.

        Returns:
            A dict mapping language names to byte counts.
        """
        url = f"{self.API_BASE}/repos/{username}/{repo_name}/languages"

        try:
            response = requests.get(url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                return response.json()
            else:
                return {}

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            return {}

    # -----------------------------------------------------------------
    #  Fetch Repository Topics
    # -----------------------------------------------------------------
    def get_repo_topics(self, username: str, repo_name: str) -> List[str]:
        """
        Get the topic tags assigned to a repository.

        Topics are user-defined tags like "machine-learning", "react", etc.
        Requires the 'mercy-preview' Accept header.

        Args:
            username:  The GitHub username.
            repo_name: The repository name.

        Returns:
            A list of topic strings.
        """
        url = f"{self.API_BASE}/repos/{username}/{repo_name}/topics"

        # Topics API requires a special Accept header
        topic_headers = {
            **self.headers,
            "Accept": "application/vnd.github.mercy-preview+json",
        }

        try:
            response = requests.get(url, headers=topic_headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return data.get("names", [])
            else:
                return []

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            return []

    # -----------------------------------------------------------------
    #  Full Profile Analysis (Main Entry Point)
    # -----------------------------------------------------------------
    def analyze_github_profile(
        self,
        username: str,
        skills_master: Dict[str, List[str]],
    ) -> Dict:
        """
        Perform a full analysis of a GitHub user's public profile.

        Pipeline:
          1. Fetch all repos (skip forks — we want original work only)
          2. For each repo, fetch languages and topics
          3. Aggregate language byte counts and topic lists
          4. Map languages and topics to the master skills list
          5. Calculate a proficiency score per skill using log-normalization

        Args:
            username:      The GitHub username to analyze.
            skills_master: The master skills dict keyed by category.

        Returns:
            A dict containing:
              - repos_analyzed:      Number of non-fork repos processed
              - demonstrated_skills: List of skills found on GitHub
              - skill_proficiency:   Dict of {skill: score 0-1}
              - raw_languages:       Dict of {language: total_bytes}
              - raw_topics:          List of all topics found
              - error:               Error message, if any
        """
        print(f"\n{'='*60}")
        print(f"   [GitHubAnalyzer] Analyzing profile: {username}")
        print(f"{'='*60}")

        # Step 1: Fetch all repos
        repos, error = self.get_user_repos(username)

        if error:
            # Return a partial result with the error message
            return {
                "repos_analyzed": 0,
                "demonstrated_skills": [],
                "skill_proficiency": {},
                "raw_languages": {},
                "raw_topics": [],
                "error": error,
            }

        # Step 2: Filter out forked repos — we only want original work
        original_repos = [r for r in repos if not r["fork"]]
        print(f"   [GitHubAnalyzer] Analyzing {len(original_repos)} original repos (skipped forks).")

        # Step 3: Aggregate languages and topics across all repos
        language_bytes: Dict[str, int] = {}
        all_topics: List[str] = []

        for repo in original_repos:
            # Get detailed language breakdown for this repo
            repo_langs = self.get_repo_languages(username, repo["name"])
            for lang, byte_count in repo_langs.items():
                language_bytes[lang] = language_bytes.get(lang, 0) + byte_count

            # Get topics for this repo
            repo_topics = self.get_repo_topics(username, repo["name"])
            all_topics.extend(repo_topics)

            # Also include the primary language from the repo metadata
            if repo["language"]:
                lang = repo["language"]
                if lang not in language_bytes:
                    language_bytes[lang] = 0

        # Deduplicate topics
        all_topics = list(set(all_topics))

        print(f"   [GitHubAnalyzer] Languages found: {list(language_bytes.keys())}")
        print(f"   [GitHubAnalyzer] Topics found: {all_topics}")

        # Step 4: Map languages and topics to master skills
        demonstrated_skills = set()

        # Flatten the master skills list for matching
        all_master_skills = []
        for category, skills in skills_master.items():
            all_master_skills.extend(skills)

        # Match GitHub languages to master skills (case-insensitive)
        for lang in language_bytes.keys():
            for skill in all_master_skills:
                if lang.lower() == skill.lower():
                    demonstrated_skills.add(skill)

        # Match GitHub topics to master skills (case-insensitive, with hyphen handling)
        for topic in all_topics:
            topic_clean = topic.lower().replace("-", " ").replace("_", " ")
            for skill in all_master_skills:
                skill_clean = skill.lower().replace("-", " ").replace("_", " ")
                if topic_clean == skill_clean or topic_clean in skill_clean or skill_clean in topic_clean:
                    demonstrated_skills.add(skill)

        # Step 5: Calculate proficiency score for each demonstrated skill
        # Use log-normalization: score = log(bytes + 1) / log(max_bytes + 1)
        # This gives a 0-1 scale where more code = higher score
        skill_proficiency: Dict[str, float] = {}

        if language_bytes:
            max_bytes = max(language_bytes.values()) if language_bytes else 1

            for skill in demonstrated_skills:
                # Find if this skill corresponds to a language with byte counts
                skill_bytes = 0
                for lang, byte_count in language_bytes.items():
                    if lang.lower() == skill.lower():
                        skill_bytes = byte_count
                        break

                if skill_bytes > 0:
                    # Log-normalize: higher byte count = higher proficiency
                    score = math.log(skill_bytes + 1) / math.log(max_bytes + 1)
                    skill_proficiency[skill] = round(score, 3)
                else:
                    # Skill found via topics but no byte count — give a base score
                    skill_proficiency[skill] = 0.3

        # Step 6: Deep repo analysis — parse dependency files and READMEs
        print(f"   [GitHubAnalyzer] Running deep analysis on top repos...")
        dep_skills = self._deep_analyze_repos(username, original_repos[:10], all_master_skills)
        demonstrated_skills.update(dep_skills)

        # Step 7: Get commit activity stats
        commit_stats = self._get_commit_activity(username)

        demonstrated_list = sorted(demonstrated_skills)
        print(f"   [GitHubAnalyzer] Demonstrated skills: {demonstrated_list}")

        return {
            "repos_analyzed": len(original_repos),
            "demonstrated_skills": demonstrated_list,
            "skill_proficiency": skill_proficiency,
            "raw_languages": language_bytes,
            "raw_topics": all_topics,
            "commit_activity": commit_stats,
            "total_repos": len(repos),
            "original_repos": len(original_repos),
            "error": "",
        }

    # -----------------------------------------------------------------
    #  Deep Repo Analysis — Parse dependency files
    # -----------------------------------------------------------------
    # Maps dependency file names to parsers
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

    # Maps common dependency names to skill names
    DEP_TO_SKILL = {
        # Python
        "flask": "Flask", "django": "Django", "fastapi": "FastAPI",
        "tensorflow": "TensorFlow", "torch": "PyTorch", "pytorch": "PyTorch",
        "keras": "Keras", "scikit-learn": "Scikit-learn", "sklearn": "Scikit-learn",
        "pandas": "Pandas", "numpy": "NumPy", "scipy": "Statistics",
        "matplotlib": "Matplotlib", "plotly": "Data Visualization",
        "sqlalchemy": "SQL", "psycopg2": "PostgreSQL", "pymongo": "MongoDB",
        "redis": "Redis", "celery": "Celery", "pytest": "Testing",
        "selenium": "Testing", "spacy": "NLP", "nltk": "NLP",
        "transformers": "NLP", "opencv": "Computer Vision",
        "mlflow": "MLflow", "airflow": "CI/CD",
        # JavaScript/Node
        "react": "React", "react-dom": "React", "next": "Next.js",
        "vue": "Vue", "angular": "Angular", "@angular/core": "Angular",
        "express": "Express", "typescript": "TypeScript",
        "webpack": "Webpack", "jest": "Testing", "mocha": "Testing",
        "mongoose": "MongoDB", "sequelize": "SQL", "graphql": "GraphQL",
        "tailwindcss": "Tailwind CSS", "redux": "React",
        # Docker/DevOps
        "docker": "Docker", "kubernetes": "Kubernetes", "k8s": "Kubernetes",
        "terraform": "Terraform", "ansible": "Ansible",
        "jenkins": "Jenkins", "nginx": "Linux",
        # Cloud
        "boto3": "AWS", "aws-sdk": "AWS",
        "azure": "Azure", "@azure": "Azure",
        "google-cloud": "GCP", "@google-cloud": "GCP",
        "firebase": "Firebase",
    }

    def _deep_analyze_repos(
        self, username: str, repos: List[Dict], master_skills: List[str]
    ) -> set:
        """Analyze top repos for dependency files and READMEs to find more skills."""
        found_skills = set()

        for repo in repos:
            repo_name = repo["name"]

            # Try to fetch each dependency file
            for dep_file in self.DEP_FILES:
                content = self._get_file_content(username, repo_name, dep_file)
                if content:
                    skills = self._extract_skills_from_deps(content, dep_file)
                    found_skills.update(skills)

            # Parse README for skill mentions
            for readme_name in ["README.md", "readme.md", "README.rst"]:
                readme = self._get_file_content(username, repo_name, readme_name)
                if readme:
                    skills = self._extract_skills_from_readme(readme, master_skills)
                    found_skills.update(skills)
                    break

        print(f"   [GitHubAnalyzer] Deep analysis found {len(found_skills)} additional skills")
        return found_skills

    def _get_file_content(self, username: str, repo_name: str, file_path: str) -> str:
        """Fetch raw file content from a GitHub repo."""
        url = f"{self.API_BASE}/repos/{username}/{repo_name}/contents/{file_path}"
        try:
            response = requests.get(url, headers=self.headers, timeout=8)
            if response.status_code == 200:
                data = response.json()
                if data.get("encoding") == "base64":
                    import base64
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

        # Dockerfile-specific
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

        for skill in master_skills:
            pattern = r"\b" + re.escape(skill.lower()) + r"\b"
            if skill in ("C++", "C#"):
                pattern = re.escape(skill.lower())
            if re.search(pattern, readme_lower):
                found.add(skill)

        return found

    # -----------------------------------------------------------------
    #  Commit Activity Analysis
    # -----------------------------------------------------------------
    def _get_commit_activity(self, username: str) -> Dict:
        """Get commit activity stats for the user."""
        url = f"{self.API_BASE}/users/{username}/events/public"
        try:
            response = requests.get(
                url, headers=self.headers, params={"per_page": 100}, timeout=10
            )
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
