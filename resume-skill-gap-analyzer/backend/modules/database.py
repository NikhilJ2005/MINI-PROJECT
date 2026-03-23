"""
=============================================================================
 Database Module — SQLite-based Candidate & Analysis Storage
=============================================================================
 Provides persistent storage for:
   - Candidates (name, email, resume text, GitHub URL)
   - Analyses (per-candidate, per-role analysis results)
   - Batch jobs (multi-resume analysis sessions)

 Uses SQLite — zero config, single file, handles thousands of candidates.
=============================================================================
"""

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

from loguru import logger

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "recruiting.db")


class Database:
    """SQLite database for candidate and analysis persistence."""

    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Run PRAGMAs once at init, not per-connection
        init_conn = sqlite3.connect(db_path)
        init_conn.execute("PRAGMA journal_mode=WAL")
        init_conn.execute("PRAGMA foreign_keys=ON")
        init_conn.execute("PRAGMA synchronous=NORMAL")
        init_conn.close()

        self._init_tables()
        logger.info(f"[Database] Initialized at {db_path}")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_tables(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT DEFAULT '',
                    email TEXT DEFAULT '',
                    phone TEXT DEFAULT '',
                    education TEXT DEFAULT '',
                    github_username TEXT DEFAULT '',
                    github_url TEXT DEFAULT '',
                    linkedin_url TEXT DEFAULT '',
                    resume_text TEXT DEFAULT '',
                    resume_filename TEXT DEFAULT '',
                    extracted_skills TEXT DEFAULT '[]',
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id INTEGER NOT NULL,
                    target_role TEXT NOT NULL,
                    match_score REAL DEFAULT 0,
                    gap_score REAL DEFAULT 0,
                    confidence REAL DEFAULT 0,
                    report_json TEXT DEFAULT '{}',
                    github_skills TEXT DEFAULT '[]',
                    missing_skills TEXT DEFAULT '[]',
                    analyzed_at REAL NOT NULL,
                    FOREIGN KEY (candidate_id) REFERENCES candidates(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS batch_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_role TEXT NOT NULL,
                    total_candidates INTEGER DEFAULT 0,
                    completed INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    created_at REAL NOT NULL,
                    completed_at REAL
                );

                CREATE TABLE IF NOT EXISTS batch_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id INTEGER NOT NULL,
                    candidate_id INTEGER NOT NULL,
                    analysis_id INTEGER NOT NULL,
                    rank INTEGER DEFAULT 0,
                    FOREIGN KEY (batch_id) REFERENCES batch_jobs(id) ON DELETE CASCADE,
                    FOREIGN KEY (candidate_id) REFERENCES candidates(id) ON DELETE CASCADE,
                    FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_analyses_candidate
                    ON analyses(candidate_id);
                CREATE INDEX IF NOT EXISTS idx_analyses_role
                    ON analyses(target_role);
                CREATE INDEX IF NOT EXISTS idx_analyses_score
                    ON analyses(match_score DESC);
                CREATE INDEX IF NOT EXISTS idx_batch_results_batch
                    ON batch_results(batch_id);
                CREATE INDEX IF NOT EXISTS idx_candidates_github
                    ON candidates(github_username);
                CREATE INDEX IF NOT EXISTS idx_candidates_email
                    ON candidates(email);
                CREATE INDEX IF NOT EXISTS idx_batch_results_analysis
                    ON batch_results(analysis_id);
            """)

            # Migration: add composite_score column if missing
            try:
                conn.execute("SELECT composite_score FROM analyses LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE analyses ADD COLUMN composite_score REAL DEFAULT 0")

            # Index on composite_score (must be after migration that adds the column)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_composite ON analyses(composite_score DESC)")

    # -----------------------------------------------------------------
    #  Candidate CRUD
    # -----------------------------------------------------------------
    def insert_candidate(self, candidate_data: Dict) -> int:
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO candidates
                    (name, email, phone, education, github_username, github_url,
                     linkedin_url, resume_text, resume_filename, extracted_skills, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                candidate_data.get("name", ""),
                candidate_data.get("email", ""),
                candidate_data.get("phone", ""),
                candidate_data.get("education", ""),
                candidate_data.get("github_username", ""),
                candidate_data.get("github_url", ""),
                candidate_data.get("linkedin_url", ""),
                candidate_data.get("resume_text", ""),
                candidate_data.get("resume_filename", ""),
                json.dumps(candidate_data.get("extracted_skills", [])),
                time.time(),
            ))
            return cursor.lastrowid

    def get_candidate(self, candidate_id: int) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM candidates WHERE id = ?", (candidate_id,)
            ).fetchone()
            if row:
                d = dict(row)
                d["extracted_skills"] = json.loads(d["extracted_skills"] or "[]")
                return d
            return None

    def get_all_candidates(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM candidates ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["extracted_skills"] = json.loads(d["extracted_skills"] or "[]")
                results.append(d)
            return results

    def delete_candidate(self, candidate_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM candidates WHERE id = ?", (candidate_id,)
            )
            return cursor.rowcount > 0

    def get_candidate_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM candidates").fetchone()
            return row["cnt"]

    # -----------------------------------------------------------------
    #  Analysis CRUD
    # -----------------------------------------------------------------
    def insert_analysis(self, analysis_data: Dict) -> int:
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO analyses
                    (candidate_id, target_role, match_score, gap_score,
                     confidence, composite_score, report_json, github_skills, missing_skills, analyzed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis_data["candidate_id"],
                analysis_data["target_role"],
                analysis_data.get("match_score", 0),
                analysis_data.get("gap_score", 0),
                analysis_data.get("confidence", 0),
                analysis_data.get("composite_score", 0),
                json.dumps(analysis_data.get("report", {})),
                json.dumps(analysis_data.get("github_skills", [])),
                json.dumps(analysis_data.get("missing_skills", [])),
                time.time(),
            ))
            return cursor.lastrowid

    def get_analyses_for_candidate(self, candidate_id: int) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM analyses WHERE candidate_id = ? ORDER BY analyzed_at DESC",
                (candidate_id,),
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["report_json"] = json.loads(d["report_json"] or "{}")
                d["github_skills"] = json.loads(d["github_skills"] or "[]")
                d["missing_skills"] = json.loads(d["missing_skills"] or "[]")
                results.append(d)
            return results

    def get_analysis(self, analysis_id: int) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
            ).fetchone()
            if row:
                d = dict(row)
                d["report_json"] = json.loads(d["report_json"] or "{}")
                d["github_skills"] = json.loads(d["github_skills"] or "[]")
                d["missing_skills"] = json.loads(d["missing_skills"] or "[]")
                return d
            return None

    # -----------------------------------------------------------------
    #  Ranking — Get top candidates for a role
    # -----------------------------------------------------------------
    def get_ranked_candidates(
        self, target_role: str, limit: int = 50
    ) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT
                    c.id as candidate_id,
                    c.name,
                    c.email,
                    c.github_username,
                    c.education,
                    c.extracted_skills,
                    a.id as analysis_id,
                    a.match_score,
                    a.gap_score,
                    a.confidence,
                    a.composite_score,
                    a.github_skills,
                    a.missing_skills,
                    a.analyzed_at
                FROM analyses a
                JOIN candidates c ON c.id = a.candidate_id
                WHERE a.target_role = ?
                ORDER BY a.composite_score DESC, a.match_score DESC, a.confidence DESC
                LIMIT ?
            """, (target_role, limit)).fetchall()

            results = []
            for i, row in enumerate(rows):
                d = dict(row)
                d["rank"] = i + 1
                extracted = json.loads(d.pop("extracted_skills", "[]") or "[]")
                github_sk = json.loads(d.pop("github_skills", "[]") or "[]")
                d["missing_skills"] = json.loads(d["missing_skills"] or "[]")
                d["resume_skills_count"] = len(extracted)
                d["github_skills_count"] = len(github_sk)
                d["missing_count"] = len(d["missing_skills"])
                results.append(d)
            return results

    # -----------------------------------------------------------------
    #  Batch Jobs
    # -----------------------------------------------------------------
    def create_batch_job(self, target_role: str, total: int) -> int:
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO batch_jobs (target_role, total_candidates, status, created_at)
                VALUES (?, ?, 'processing', ?)
            """, (target_role, total, time.time()))
            return cursor.lastrowid

    def add_batch_result(
        self, batch_id: int, candidate_id: int, analysis_id: int, rank: int
    ):
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO batch_results (batch_id, candidate_id, analysis_id, rank)
                VALUES (?, ?, ?, ?)
            """, (batch_id, candidate_id, analysis_id, rank))
            conn.execute("""
                UPDATE batch_jobs SET completed = completed + 1 WHERE id = ?
            """, (batch_id,))

    def complete_batch_job(self, batch_id: int):
        with self._connect() as conn:
            conn.execute("""
                UPDATE batch_jobs
                SET status = 'completed', completed_at = ?
                WHERE id = ?
            """, (time.time(), batch_id))

    def get_batch_job(self, batch_id: int) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM batch_jobs WHERE id = ?", (batch_id,)
            ).fetchone()
            if not row:
                return None
            job = dict(row)

            results = conn.execute("""
                SELECT
                    br.rank,
                    c.id as candidate_id, c.name, c.email, c.github_username,
                    a.match_score, a.confidence, a.missing_skills
                FROM batch_results br
                JOIN candidates c ON c.id = br.candidate_id
                JOIN analyses a ON a.id = br.analysis_id
                WHERE br.batch_id = ?
                ORDER BY br.rank ASC
            """, (batch_id,)).fetchall()

            job["results"] = []
            for r in results:
                d = dict(r)
                d["missing_skills"] = json.loads(d["missing_skills"] or "[]")
                job["results"].append(d)
            return job

    # -----------------------------------------------------------------
    #  Comparison
    # -----------------------------------------------------------------
    def get_candidates_comparison(
        self, candidate_ids: List[int], target_role: str
    ) -> List[Dict]:
        if not candidate_ids:
            return []
        placeholders = ",".join("?" * len(candidate_ids))
        with self._connect() as conn:
            rows = conn.execute(f"""
                SELECT
                    c.id as candidate_id, c.name, c.email, c.github_username,
                    c.education, c.extracted_skills,
                    a.id as analysis_id, a.match_score, a.gap_score,
                    a.confidence, a.github_skills, a.missing_skills,
                    a.report_json
                FROM candidates c
                LEFT JOIN analyses a ON a.candidate_id = c.id AND a.target_role = ?
                WHERE c.id IN ({placeholders})
                ORDER BY a.match_score DESC
            """, [target_role] + candidate_ids).fetchall()

            results = []
            for row in rows:
                d = dict(row)
                d["extracted_skills"] = json.loads(d["extracted_skills"] or "[]")
                d["github_skills"] = json.loads(d["github_skills"] or "[]")
                d["missing_skills"] = json.loads(d["missing_skills"] or "[]")
                d["report_json"] = json.loads(d["report_json"] or "{}")
                results.append(d)
            return results

    # -----------------------------------------------------------------
    #  Analysis History (for sidebar)
    # -----------------------------------------------------------------
    def get_recent_analyses(self, limit: int = 50) -> List[Dict]:
        """Get recent analyses with candidate info for the history sidebar."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT
                    a.id as analysis_id,
                    a.candidate_id,
                    c.name as candidate_name,
                    c.resume_filename,
                    a.target_role,
                    a.match_score,
                    a.confidence,
                    a.analyzed_at
                FROM analyses a
                JOIN candidates c ON c.id = a.candidate_id
                ORDER BY a.analyzed_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def get_analysis_by_id(self, analysis_id: int) -> Optional[Dict]:
        """Get a single analysis with full report_json by ID."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT
                    a.*,
                    c.name as candidate_name,
                    c.email as candidate_email,
                    c.github_username,
                    c.resume_filename
                FROM analyses a
                JOIN candidates c ON c.id = a.candidate_id
                WHERE a.id = ?
            """, (analysis_id,)).fetchone()
            if row:
                d = dict(row)
                d["report_json"] = json.loads(d["report_json"] or "{}")
                d["github_skills"] = json.loads(d["github_skills"] or "[]")
                d["missing_skills"] = json.loads(d["missing_skills"] or "[]")
                return d
            return None

    # -----------------------------------------------------------------
    #  Batch Inserts (for batch processing optimization)
    # -----------------------------------------------------------------
    def insert_candidate_batch(self, candidates: List[Dict]) -> List[int]:
        """Insert multiple candidates in a single transaction. Returns list of IDs."""
        ids = []
        with self._connect() as conn:
            for c in candidates:
                cursor = conn.execute("""
                    INSERT INTO candidates
                        (name, email, phone, education, github_username, github_url,
                         linkedin_url, resume_text, resume_filename, extracted_skills, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    c.get("name", ""), c.get("email", ""), c.get("phone", ""),
                    c.get("education", ""), c.get("github_username", ""),
                    c.get("github_url", ""), c.get("linkedin_url", ""),
                    c.get("resume_text", ""), c.get("resume_filename", ""),
                    json.dumps(c.get("extracted_skills", [])), time.time(),
                ))
                ids.append(cursor.lastrowid)
        return ids

    def insert_analysis_batch(self, analyses: List[Dict]) -> List[int]:
        """Insert multiple analyses in a single transaction. Returns list of IDs."""
        ids = []
        with self._connect() as conn:
            for a in analyses:
                cursor = conn.execute("""
                    INSERT INTO analyses
                        (candidate_id, target_role, match_score, gap_score,
                         confidence, composite_score, report_json, github_skills, missing_skills, analyzed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    a["candidate_id"], a["target_role"],
                    a.get("match_score", 0), a.get("gap_score", 0),
                    a.get("confidence", 0), a.get("composite_score", 0),
                    json.dumps(a.get("report", {})),
                    json.dumps(a.get("github_skills", [])),
                    json.dumps(a.get("missing_skills", [])), time.time(),
                ))
                ids.append(cursor.lastrowid)
        return ids

    # -----------------------------------------------------------------
    #  Stats
    # -----------------------------------------------------------------
    def get_dashboard_stats(self) -> Dict:
        with self._connect() as conn:
            candidates = conn.execute(
                "SELECT COUNT(*) as cnt FROM candidates"
            ).fetchone()["cnt"]
            analyses = conn.execute(
                "SELECT COUNT(*) as cnt FROM analyses"
            ).fetchone()["cnt"]
            batches = conn.execute(
                "SELECT COUNT(*) as cnt FROM batch_jobs"
            ).fetchone()["cnt"]

            avg_score_row = conn.execute(
                "SELECT AVG(match_score) as avg FROM analyses"
            ).fetchone()
            avg_score = round(avg_score_row["avg"] or 0, 1)

            top_roles = conn.execute("""
                SELECT target_role, COUNT(*) as cnt
                FROM analyses
                GROUP BY target_role
                ORDER BY cnt DESC
                LIMIT 5
            """).fetchall()

            # Recent activity (last 5 analyses)
            recent = conn.execute("""
                SELECT c.name, a.target_role, a.match_score, a.confidence, a.analyzed_at
                FROM analyses a
                JOIN candidates c ON c.id = a.candidate_id
                ORDER BY a.analyzed_at DESC
                LIMIT 5
            """).fetchall()

            # Top candidates (highest match scores)
            top_cands = conn.execute("""
                SELECT c.name, a.target_role, a.match_score, a.confidence
                FROM analyses a
                JOIN candidates c ON c.id = a.candidate_id
                ORDER BY a.match_score DESC, a.confidence DESC
                LIMIT 5
            """).fetchall()

            # Score distribution buckets
            score_dist = conn.execute("""
                SELECT
                    SUM(CASE WHEN match_score >= 75 THEN 1 ELSE 0 END) as excellent,
                    SUM(CASE WHEN match_score >= 50 AND match_score < 75 THEN 1 ELSE 0 END) as good,
                    SUM(CASE WHEN match_score >= 25 AND match_score < 50 THEN 1 ELSE 0 END) as fair,
                    SUM(CASE WHEN match_score < 25 THEN 1 ELSE 0 END) as poor
                FROM analyses
            """).fetchone()

            # Most common missing skills (aggregate from all analyses)
            all_missing_rows = conn.execute(
                "SELECT missing_skills FROM analyses WHERE missing_skills != '[]' ORDER BY analyzed_at DESC LIMIT 100"
            ).fetchall()
            skill_counts = {}
            for row in all_missing_rows:
                skills = json.loads(row["missing_skills"] or "[]")
                for s in skills:
                    skill_counts[s] = skill_counts.get(s, 0) + 1
            top_gaps = sorted(skill_counts.items(), key=lambda x: -x[1])[:10]

            return {
                "total_candidates": candidates,
                "total_analyses": analyses,
                "total_batches": batches,
                "average_match_score": avg_score,
                "top_roles": [{"role": r["target_role"], "count": r["cnt"]} for r in top_roles],
                "recent_activity": [
                    {"name": r["name"], "role": r["target_role"],
                     "match_score": r["match_score"], "confidence": r["confidence"],
                     "analyzed_at": r["analyzed_at"]}
                    for r in recent
                ],
                "top_candidates": [
                    {"name": r["name"], "role": r["target_role"],
                     "match_score": r["match_score"], "confidence": r["confidence"]}
                    for r in top_cands
                ],
                "score_distribution": {
                    "excellent": score_dist["excellent"] or 0,
                    "good": score_dist["good"] or 0,
                    "fair": score_dist["fair"] or 0,
                    "poor": score_dist["poor"] or 0,
                },
                "top_skill_gaps": [{"skill": s, "count": c} for s, c in top_gaps],
            }
