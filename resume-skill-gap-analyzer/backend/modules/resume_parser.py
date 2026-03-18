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
from loguru import logger


def compile_skill_patterns(
    skills_master: Dict[str, List[str]],
    skill_aliases: Dict[str, str] = None,
) -> Dict[str, "re.Pattern"]:
    """Pre-compile regex patterns for all skills and aliases. Call once at startup."""
    patterns = {}
    for category, skills in skills_master.items():
        for skill in skills:
            if skill in ("C++", "C#"):
                patterns[skill] = re.compile(re.escape(skill.lower()))
            else:
                patterns[skill] = re.compile(r"\b" + re.escape(skill.lower()) + r"\b")
    # Compile alias patterns — map each alias to its canonical skill name
    if skill_aliases:
        for alias, canonical in skill_aliases.items():
            if alias in ("C++", "C#"):
                patterns[f"__alias__{alias}"] = re.compile(re.escape(alias.lower()))
            else:
                patterns[f"__alias__{alias}"] = re.compile(r"\b" + re.escape(alias.lower()) + r"\b")
    return patterns


# Module-level alias mapping
_skill_aliases: Dict[str, str] = {}


def set_skill_aliases(aliases: Dict[str, str]) -> None:
    """Set the module-level alias mapping."""
    global _skill_aliases
    _skill_aliases = aliases


def get_skill_aliases() -> Dict[str, str]:
    """Get the module-level alias mapping."""
    return _skill_aliases


# Module-level cache for compiled patterns
_compiled_skill_patterns: Dict[str, "re.Pattern"] = {}


def set_compiled_patterns(patterns: Dict[str, "re.Pattern"]) -> None:
    """Set the module-level compiled patterns cache."""
    global _compiled_skill_patterns
    _compiled_skill_patterns = patterns


def get_compiled_patterns() -> Dict[str, "re.Pattern"]:
    """Get the module-level compiled patterns cache."""
    return _compiled_skill_patterns


# Module-level flattened skills list (computed once at startup)
_all_skills_flat: List[str] = []


def set_flat_skills(skills_master: Dict[str, List[str]]) -> None:
    """Flatten skills_master once at startup."""
    global _all_skills_flat
    _all_skills_flat = [skill for skills in skills_master.values() for skill in skills]


def get_flat_skills() -> List[str]:
    """Get the pre-flattened skills list."""
    return _all_skills_flat


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
            logger.info("[ResumeParser] spaCy model loaded successfully.")
        except OSError:
            logger.warning("[ResumeParser] spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
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

            page_count = len(doc)
            doc.close()
            logger.info(f"[ResumeParser] Extracted text from PDF ({page_count} pages).")

        except Exception as e:
            logger.error(f"[ResumeParser] Error extracting PDF text: {e}")
            text = ""

        return text.strip()

    # -----------------------------------------------------------------
    #  DOCX Text Extraction
    # -----------------------------------------------------------------
    def extract_text_from_docx(self, file_bytes: bytes) -> str:
        """Extract raw text from a DOCX file using python-docx."""
        import io
        try:
            from docx import Document
            doc = Document(io.BytesIO(file_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text = "\n".join(paragraphs)
            logger.info(f"[ResumeParser] Extracted text from DOCX ({len(paragraphs)} paragraphs).")
            return text.strip()
        except Exception as e:
            logger.error(f"[ResumeParser] Error extracting DOCX text: {e}")
            return ""

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

        logger.info(f"[ResumeParser] Extracted text from TXT ({len(text)} characters).")
        return text.strip()

    # -----------------------------------------------------------------
    #  Skill Extraction (Core NLP + Regex Logic)
    # -----------------------------------------------------------------
    def extract_skills(self, text: str, skills_master: Dict[str, List[str]], spacy_doc=None) -> List[str]:
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

        # Lowercase version of the text for case-insensitive matching
        text_lower = text.lower()

        # Use pre-compiled patterns if available, otherwise compile on the fly
        aliases = get_skill_aliases()
        patterns = get_compiled_patterns()
        if patterns:
            for key, pattern in patterns.items():
                if pattern.search(text_lower):
                    if key.startswith("__alias__"):
                        alias_name = key[len("__alias__"):]
                        canonical = aliases.get(alias_name)
                        if canonical:
                            found_skills.add(canonical)
                    else:
                        found_skills.add(key)
        else:
            # Fallback: flatten and compile per-request (slow path)
            all_skills = []
            for category, skills in skills_master.items():
                all_skills.extend(skills)
            for skill in all_skills:
                if skill in ("C++", "C#"):
                    pat = re.escape(skill.lower())
                else:
                    pat = r"\b" + re.escape(skill.lower()) + r"\b"
                if re.search(pat, text_lower):
                    found_skills.add(skill)

        # Flatten for spaCy NLP matching below
        all_skills = []
        for category, skills in skills_master.items():
            all_skills.extend(skills)

        # --- spaCy NLP-based extraction (supplementary) ---
        if self.nlp is not None:
            doc = spacy_doc if spacy_doc is not None else self.nlp(text)

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
        logger.info(f"[ResumeParser] Found {len(result)} skills in resume text.")
        return result

    # -----------------------------------------------------------------
    #  Personal Info Extraction
    # -----------------------------------------------------------------
    def extract_personal_info(self, text: str, spacy_doc=None) -> Dict:
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

        # --- Name Extraction ---
        # Strategy: try first-line heuristic first (most reliable for standard resumes),
        # then validate/supplement with spaCy NER.

        # Blacklist: words that spaCy commonly misidentifies as PERSON
        _name_blacklist = {
            "algorithm", "algorithms", "summary", "objective", "education",
            "experience", "skills", "projects", "references", "professional",
            "technical", "curriculum", "vitae", "resume", "data", "engineering",
            "computer", "science", "information", "technology", "university",
            "institute", "college", "bachelor", "master", "work", "career",
            "contact", "personal", "about", "overview", "profile", "introduction",
        }

        # First: try the first-line heuristic (most resumes put the name on line 1)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        first_line_candidate = ""
        if lines:
            first_line = lines[0]
            # First line is often the name if it's short, has no special chars,
            # and doesn't look like a section header or technical term
            fl_lower = first_line.lower()
            if (
                len(first_line) < 50
                and not self.EMAIL_PATTERN.search(first_line)
                and not self.PHONE_PATTERN.search(first_line)
                and not any(kw in fl_lower for kw in _name_blacklist)
                # Name should have at least 2 words (first + last)
                and len(first_line.split()) >= 2
                # All words should be alphabetic (names don't contain special chars)
                and all(word.isalpha() for word in first_line.replace("-", "").replace(".", "").split())
            ):
                first_line_candidate = first_line

        # Second: use spaCy NER to find PERSON entities as validation/fallback
        spacy_name_candidate = ""
        if self.nlp is not None:
            if spacy_doc is not None:
                doc = spacy_doc
            else:
                doc = self.nlp(text[:500])
            for ent in doc.ents:
                if ent.start_char > 500:
                    break
                if ent.label_ == "PERSON":
                    # Clean up: spaCy may span multiple lines — take only the name-like parts
                    raw_name = ent.text.strip()
                    # If it spans newlines, take the part that looks like a name
                    name_parts = [p.strip() for p in raw_name.split("\n") if p.strip()]
                    name = ""
                    for part in name_parts:
                        part_lower = part.lower()
                        if (
                            len(part) > 2
                            and not any(c.isdigit() for c in part)
                            and not any(word in _name_blacklist for word in part_lower.split())
                            and part.replace("-", "").replace(".", "").replace(" ", "").isalpha()
                        ):
                            name = part
                            break
                    if name and len(name.split()) >= 2:
                        spacy_name_candidate = name
                        break

        # Decide: prefer first-line if valid, else spaCy, else single-word first-line
        if first_line_candidate:
            info["name"] = first_line_candidate
        elif spacy_name_candidate:
            info["name"] = spacy_name_candidate
        elif lines:
            # Last resort: first line if it's short and looks like a name (even single word)
            first_line = lines[0]
            fl_lower = first_line.lower()
            if (
                len(first_line) < 40
                and first_line.replace("-", "").replace(".", "").replace(" ", "").isalpha()
                and not self.EMAIL_PATTERN.search(first_line)
                and fl_lower not in _name_blacklist
                and not any(kw in fl_lower for kw in _name_blacklist)
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

        logger.info(f"[ResumeParser] Extracted info — Name: {info['name']}, "
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
        logger.info(f"[ResumeParser] Parsing file: {filename}")

        # Route to the appropriate extractor based on file extension
        if filename.lower().endswith(".pdf"):
            raw_text = self.extract_text_from_pdf(file_bytes)
        elif filename.lower().endswith(".docx"):
            raw_text = self.extract_text_from_docx(file_bytes)
        elif filename.lower().endswith(".txt"):
            raw_text = self.extract_text_from_txt(file_bytes)
        else:
            # Unsupported format — return empty results
            logger.error(f"[ResumeParser] Unsupported file type: {filename}")
            return {
                "raw_text": "",
                "extracted_skills": [],
                "skill_count": 0,
            }

        # Handle empty extraction (corrupted file, etc.)
        if not raw_text or not raw_text.strip():
            logger.warning("[ResumeParser] No text extracted from file.")
            raise ValueError(f"Could not extract text from '{filename}'. The file may be empty, corrupted, or image-based.")

        # Run spaCy NLP once and share the doc across both extraction methods
        spacy_doc = self.nlp(raw_text) if self.nlp else None

        # Run skill extraction on the raw text (reuse spaCy doc)
        extracted_skills = self.extract_skills(raw_text, skills_master, spacy_doc=spacy_doc)

        # Extract personal info (reuse spaCy doc — filters entities by position)
        personal_info = self.extract_personal_info(raw_text, spacy_doc=spacy_doc)

        return {
            "raw_text": raw_text,
            "extracted_skills": extracted_skills,
            "skill_count": len(extracted_skills),
            "personal_info": personal_info,
        }
