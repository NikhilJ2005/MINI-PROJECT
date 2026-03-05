"""
=============================================================================
 Resume Parser Module
=============================================================================
 Role in the pipeline:
   This is the FIRST stage of the analysis pipeline. It takes in a resume
   file (PDF or TXT), extracts the raw text, and then identifies technical
   skills by matching against the master skills list.

 Techniques used:
   - PyMuPDF (fitz) for PDF text extraction
   - spaCy NLP for noun-chunk and named-entity extraction
   - Regex word-boundary matching for precise skill detection
=============================================================================
"""

import re
from typing import Dict, List, Optional

import fitz  # PyMuPDF — high-performance PDF text extraction
import spacy


class ResumeParser:
    """Parses resume files and extracts technical skills and personal info from the text."""

    # Regex patterns for extracting personal info
    EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
    PHONE_PATTERN = re.compile(
        r"(?:\+?\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}"
    )
    GITHUB_URL_PATTERN = re.compile(
        r"(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9\-]+)", re.IGNORECASE
    )
    LINKEDIN_URL_PATTERN = re.compile(
        r"(?:https?://)?(?:www\.)?linkedin\.com/in/([a-zA-Z0-9\-]+)", re.IGNORECASE
    )
    EDUCATION_KEYWORDS = [
        "B.Tech", "B.E.", "B.Sc", "M.Tech", "M.E.", "M.Sc", "MBA", "Ph.D", "PhD",
        "Bachelor", "Master", "Diploma", "Associate", "B.S.", "M.S.", "B.A.", "M.A.",
        "Computer Science", "Information Technology", "Engineering",
        "University", "Institute", "College",
    ]

    def __init__(self) -> None:
        """Initialize the parser by loading the spaCy English NLP model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("   [ResumeParser] spaCy model loaded successfully.")
        except OSError:
            print("   [ResumeParser] WARNING: spaCy model 'en_core_web_sm' not found.")
            print("   Run: python -m spacy download en_core_web_sm")
            self.nlp = None

    # -----------------------------------------------------------------
    #  PDF Text Extraction
    # -----------------------------------------------------------------
    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        """
        Extract raw text from a PDF file using PyMuPDF.

        Args:
            file_bytes: The raw bytes of the uploaded PDF file.

        Returns:
            A single string containing all text from every page.
        """
        text = ""
        try:
            # Open the PDF from an in-memory byte stream
            doc = fitz.open(stream=file_bytes, filetype="pdf")

            # Iterate through every page and accumulate text
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                text += page_text + "\n"

            doc.close()
            print(f"   [ResumeParser] Extracted text from PDF ({len(doc)} pages).")

        except Exception as e:
            print(f"   [ResumeParser] ERROR extracting PDF text: {e}")
            text = ""

        return text.strip()

    # -----------------------------------------------------------------
    #  Plain Text Extraction
    # -----------------------------------------------------------------
    def extract_text_from_txt(self, file_bytes: bytes) -> str:
        """
        Decode plain-text resume content from raw bytes.

        Args:
            file_bytes: The raw bytes of the uploaded TXT file.

        Returns:
            The decoded text string.
        """
        try:
            # Try UTF-8 first, fall back to latin-1 for broader compatibility
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1")

        print(f"   [ResumeParser] Extracted text from TXT ({len(text)} characters).")
        return text.strip()

    # -----------------------------------------------------------------
    #  Skill Extraction (Core NLP + Regex Logic)
    # -----------------------------------------------------------------
    def extract_skills(self, text: str, skills_master: Dict[str, List[str]]) -> List[str]:
        """
        Identify technical skills mentioned in the resume text.

        Strategy:
          1. Flatten the skills_master dict into a single searchable list.
          2. Use regex word-boundary matching to find each skill in the text.
             This prevents partial matches (e.g., "R" inside "React").
          3. Use spaCy to extract noun chunks and named entities as
             supplementary signals for skill detection.
          4. Deduplicate and sort the final list alphabetically.

        Args:
            text:          The raw resume text (already extracted).
            skills_master: The master skills dictionary keyed by category.

        Returns:
            A sorted, deduplicated list of skill names found in the text.
        """
        found_skills = set()

        # Flatten the master skills dict into one list
        all_skills = []
        for category, skills in skills_master.items():
            all_skills.extend(skills)

        # Lowercase version of the text for case-insensitive matching
        text_lower = text.lower()

        # --- Regex-based skill matching ---
        for skill in all_skills:
            # Build a regex pattern with word boundaries
            # re.escape handles special characters like "C++", "C#", "Node.js"
            pattern = r"\b" + re.escape(skill.lower()) + r"\b"

            # Special handling for skills with special chars that break \b
            if skill in ("C++", "C#"):
                pattern = re.escape(skill.lower())

            if re.search(pattern, text_lower):
                found_skills.add(skill)

        # --- spaCy NLP-based extraction (supplementary) ---
        if self.nlp is not None:
            doc = self.nlp(text)

            # Extract noun chunks (e.g., "machine learning", "data science")
            noun_chunks = [chunk.text.lower().strip() for chunk in doc.noun_chunks]

            # Extract named entities (e.g., "Python", "AWS")
            entities = [ent.text.lower().strip() for ent in doc.ents]

            # Check if any master skill appears in noun chunks or entities
            for skill in all_skills:
                skill_lower = skill.lower()
                if skill_lower in noun_chunks or skill_lower in entities:
                    found_skills.add(skill)

        # Sort alphabetically for consistent output
        result = sorted(found_skills)
        print(f"   [ResumeParser] Found {len(result)} skills in resume text.")
        return result

    # -----------------------------------------------------------------
    #  Personal Info Extraction
    # -----------------------------------------------------------------
    def extract_personal_info(self, text: str) -> Dict:
        """
        Extract personal information from resume text.
        Returns dict with: name, email, phone, github_username, github_url,
                           linkedin_url, education
        """
        info: Dict = {
            "name": "",
            "email": "",
            "phone": "",
            "github_username": "",
            "github_url": "",
            "linkedin_url": "",
            "education": "",
        }

        # Email
        email_match = self.EMAIL_PATTERN.search(text)
        if email_match:
            info["email"] = email_match.group(0)

        # Phone
        phone_match = self.PHONE_PATTERN.search(text)
        if phone_match:
            info["phone"] = phone_match.group(0).strip()

        # GitHub URL and username
        github_match = self.GITHUB_URL_PATTERN.search(text)
        if github_match:
            info["github_username"] = github_match.group(1)
            info["github_url"] = github_match.group(0)
            if not info["github_url"].startswith("http"):
                info["github_url"] = "https://" + info["github_url"]

        # LinkedIn URL
        linkedin_match = self.LINKEDIN_URL_PATTERN.search(text)
        if linkedin_match:
            info["linkedin_url"] = linkedin_match.group(0)
            if not info["linkedin_url"].startswith("http"):
                info["linkedin_url"] = "https://" + info["linkedin_url"]

        # Name — use spaCy NER to find PERSON entities (first one is usually the candidate)
        if self.nlp is not None:
            # Only process the first ~500 chars for name detection (name is at the top)
            doc = self.nlp(text[:500])
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    name = ent.text.strip()
                    # Filter out single-char or obviously wrong names
                    if len(name) > 2 and not any(c.isdigit() for c in name):
                        info["name"] = name
                        break

        # If spaCy didn't find a name, use the first line heuristic
        if not info["name"]:
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            if lines:
                first_line = lines[0]
                # First line is often the name if it's short and has no special chars
                if (
                    len(first_line) < 50
                    and not self.EMAIL_PATTERN.search(first_line)
                    and not self.PHONE_PATTERN.search(first_line)
                    and not any(kw.lower() in first_line.lower() for kw in ["resume", "curriculum", "objective"])
                ):
                    info["name"] = first_line

        # Education — find lines containing education keywords
        education_lines = []
        for line in text.split("\n"):
            line_stripped = line.strip()
            if any(kw.lower() in line_stripped.lower() for kw in self.EDUCATION_KEYWORDS):
                if len(line_stripped) > 5 and len(line_stripped) < 200:
                    education_lines.append(line_stripped)
        if education_lines:
            info["education"] = " | ".join(education_lines[:3])

        print(f"   [ResumeParser] Extracted info — Name: {info['name']}, "
              f"Email: {info['email']}, GitHub: {info['github_username']}")
        return info

    # -----------------------------------------------------------------
    #  Main Parse Method (Entry Point)
    # -----------------------------------------------------------------
    def parse(
        self,
        file_bytes: bytes,
        filename: str,
        skills_master: Dict[str, List[str]],
    ) -> Dict:
        """
        Parse a resume file and extract skills — main entry point.

        Routes to the correct text extractor based on file extension,
        then runs skill extraction on the resulting text.

        Args:
            file_bytes:    Raw bytes of the uploaded file.
            filename:      Original filename (used to determine file type).
            skills_master: The master skills dictionary.

        Returns:
            A dict containing:
              - raw_text:         The full extracted text
              - extracted_skills: List of identified skill names
              - skill_count:      Number of skills found
        """
        print(f"\n{'='*60}")
        print(f"   [ResumeParser] Parsing file: {filename}")
        print(f"{'='*60}")

        # Route to the appropriate extractor based on file extension
        if filename.lower().endswith(".pdf"):
            raw_text = self.extract_text_from_pdf(file_bytes)
        elif filename.lower().endswith(".txt"):
            raw_text = self.extract_text_from_txt(file_bytes)
        else:
            # Unsupported format — return empty results
            print(f"   [ResumeParser] ERROR: Unsupported file type: {filename}")
            return {
                "raw_text": "",
                "extracted_skills": [],
                "skill_count": 0,
            }

        # Handle empty extraction (corrupted file, etc.)
        if not raw_text:
            print("   [ResumeParser] WARNING: No text extracted from file.")
            return {
                "raw_text": "",
                "extracted_skills": [],
                "skill_count": 0,
            }

        # Run skill extraction on the raw text
        extracted_skills = self.extract_skills(raw_text, skills_master)

        # Extract personal info (name, email, phone, github, education)
        personal_info = self.extract_personal_info(raw_text)

        return {
            "raw_text": raw_text,
            "extracted_skills": extracted_skills,
            "skill_count": len(extracted_skills),
            "personal_info": personal_info,
        }
