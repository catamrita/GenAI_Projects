"""Minimal CV parsing and ranking pipeline."""
from __future__ import annotations

import io
import os
import re
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from llm_provider import configured_models

PROVIDER = os.getenv("LLM_PROVIDER_LABEL", "OpenAI-compatible API")
SHORTLIST_TOP_K = int(os.getenv("SHORTLIST_TOP_K", "25"))
LLM_MAX_WORKERS = max(1, int(os.getenv("LLM_MAX_WORKERS", "4")))
LIGHT_LLM, HEAVY_LLM = configured_models()

JD_PROMPT = """You are an expert technical recruiter. Extract a normalized job specification.
Return ONLY JSON with: title (string), summary (string), and requirements (array).
Each requirement must have text, category (skill, experience, responsibility,
qualification, or other), must_have (boolean), and weight (number 1-5).
Split compound requirements into 6-15 atomic requirements. Do not invent facts.

JOB DESCRIPTION:
{text}
"""

RESUME_PROMPT = """You are an expert resume parser. Return ONLY JSON with: name,
email, phone, skills (array), experience_years (number), education (array), and
work_history (array of objects with company, role, duration, highlights). Use empty
values when information is unknown and never invent facts.

RESUME:
{text}
"""

SCORE_PROMPT = """You are a rigorous, unbiased hiring evaluator. Using only resume
evidence, score every requirement from 0.0 (none) to 1.0 (strong direct match).
Return ONLY JSON: {{"scores": [{{"requirement_id": "R1", "score": 0.0,
"evidence": "", "reasoning": "one sentence"}}], "summary": "two sentences"}}.

JOB: {title}
REQUIREMENTS:
{requirements}

RESUME:
{resume}
"""


@dataclass
class Requirement:
    id: str
    text: str
    category: str = "other"
    must_have: bool = False
    weight: float = 1.0


@dataclass
class Job:
    title: str = ""
    summary: str = ""
    requirements: list[Requirement] = field(default_factory=list)


@dataclass
class RequirementScore:
    requirement_id: str
    score: float = 0.0
    evidence: str = ""
    reasoning: str = ""


@dataclass
class Result:
    candidate_id: str
    name: str = ""
    overall_score: float = 0.0
    passed_must_haves: bool = True
    breakdown: list[RequirementScore] = field(default_factory=list)
    summary: str = ""
    rank: int = 0


@dataclass
class RankingOutput:
    job: Job
    results: list[Result]


def extract_text(data: bytes, filename: str) -> str:
    """Extract text from an uploaded TXT, PDF, or DOCX file."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages).strip()
    if suffix == ".docx":
        import docx2txt

        path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as temp:
                temp.write(data)
                path = temp.name
            return (docx2txt.process(path) or "").strip()
        finally:
            if path:
                Path(path).unlink(missing_ok=True)
    return data.decode("utf-8", "ignore").strip()


def _redact(text: str, name: str) -> str:
    """Remove direct identity and contact signals before candidate scoring."""
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[EMAIL]", text)
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    text = re.sub(r"(?:(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3}[\s-]?\d{3,4})", "[PHONE]", text)
    for token in name.split():
        if len(token) > 1:
            text = re.sub(rf"\b{re.escape(token)}\b", "[NAME]", text, flags=re.IGNORECASE)
    return text


def _normalize_text(text: str) -> str:
    """Canonicalize equivalent file text before hashing or model calls."""
    normalized_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized_newlines.split("\n")).strip()


def _parse_job(text: str) -> Job:
    """Use the light LLM to normalize a job description into requirements."""
    data = LIGHT_LLM.json_call(JD_PROMPT.format(text=text[:12000]))
    requirements = []
    valid_categories = {"skill", "experience", "responsibility", "qualification", "other"}
    for index, item in enumerate(data.get("requirements", []), 1):
        if not isinstance(item, dict) or not item.get("text"):
            continue
        category = str(item.get("category", "other")).lower()
        requirements.append(Requirement(
            id=f"R{index}",
            text=str(item["text"]),
            category=category if category in valid_categories else "other",
            must_have=bool(item.get("must_have", False)),
            weight=float(item.get("weight", 1) or 1),
        ))
    return Job(str(data.get("title", "")), str(data.get("summary", "")), requirements)


def _header_name(text: str) -> str:
    """Return a likely candidate name from a conventional CV header."""
    role_words = {
        "analyst", "curriculum", "developer", "education", "engineer",
        "experience", "manager", "profile", "resume", "skills", "summary", "vitae",
    }
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        words = candidate.split()
        lowered = {re.sub(r"[^a-z]", "", word.lower()) for word in words}
        if (
            2 <= len(words) <= 5
            and len(candidate) <= 80
            and all(any(character.isalpha() for character in word) for word in words)
            and not lowered.intersection(role_words)
        ):
            return candidate
        break
    return ""


def _parse_resume(filename: str, text: str) -> dict:
    """Read a conventional name header or use the light LLM as fallback."""
    name = _header_name(text)
    if not name:
        data = LIGHT_LLM.json_call(RESUME_PROMPT.format(text=text[:12000]), use_cache=False)
        name = str(data.get("name", ""))
    return {
        "candidate_id": filename,
        "name": name,
        "text": _redact(text, name),
    }


def _tokens(text: str) -> Counter:
    """Count normalized terms for inexpensive first-stage shortlisting."""
    return Counter(re.findall(r"[a-z0-9+#.]{2,}", text.lower()))


def _similarity(left: str, right: str) -> float:
    """Calculate directional lexical overlap between job and resume text."""
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    overlap = sum((left_tokens & right_tokens).values())
    return overlap / max(sum(left_tokens.values()), 1)


def _score(job: Job, resume: dict) -> Result:
    """Use the heavy LLM to score one resume against every requirement."""
    requirements = "\n".join(
        f"{item.id} | {'MUST' if item.must_have else 'nice'} | {item.text}"
        for item in job.requirements
    )
    data = HEAVY_LLM.json_call(SCORE_PROMPT.format(
        title=job.title or "the role",
        requirements=requirements,
        resume=resume["text"][:12000],
    ))
    scores = {}
    for item in data.get("scores", []):
        if not isinstance(item, dict) or "requirement_id" not in item:
            continue
        requirement_id = str(item["requirement_id"])
        try:
            value = max(0.0, min(1.0, float(item.get("score", 0) or 0)))
        except (TypeError, ValueError):
            value = 0.0
        scores[requirement_id] = RequirementScore(
            requirement_id, value, str(item.get("evidence", "")), str(item.get("reasoning", ""))
        )
    breakdown = [scores.get(item.id, RequirementScore(item.id)) for item in job.requirements]
    weights = {item.id: max(item.weight, 0) for item in job.requirements}
    overall = 100 * sum(item.score * weights.get(item.requirement_id, 1) for item in breakdown)
    overall /= sum(weights.values()) or 1
    must_ids = {item.id for item in job.requirements if item.must_have}
    passed = all(item.score >= 0.5 for item in breakdown if item.requirement_id in must_ids)
    return Result(
        candidate_id=resume["candidate_id"],
        name=resume["name"],
        overall_score=round(overall, 1),
        passed_must_haves=passed,
        breakdown=breakdown,
        summary=str(data.get("summary", "")),
    )


def rank_resumes(
    jd_text: str,
    resume_docs: list[tuple[str, str]],
    top_k: int,
    progress: Callable[[str, float], None],
) -> RankingOutput:
    """Parse, shortlist, score, and rank all submitted resumes."""
    progress("Parsing job description", 0.05)
    job = _parse_job(_normalize_text(jd_text))
    valid_docs = [
        (filename, _normalize_text(text))
        for filename, text in resume_docs
        if text.strip()
    ]
    parsed_resumes = [None] * len(valid_docs)
    with ThreadPoolExecutor(max_workers=min(LLM_MAX_WORKERS, len(valid_docs) or 1)) as executor:
        futures = {
            executor.submit(_parse_resume, filename, text): index
            for index, (filename, text) in enumerate(valid_docs)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            parsed_resumes[futures[future]] = future.result()
            progress(
                f"Parsed resume {completed}/{len(valid_docs)}",
                0.05 + 0.35 * completed / max(len(valid_docs), 1),
            )
    resumes = [resume for resume in parsed_resumes if resume is not None]

    progress("Shortlisting candidates", 0.45)
    query = " ".join([job.title, job.summary, *(item.text for item in job.requirements)])
    resumes.sort(key=lambda item: _similarity(query, item["text"]), reverse=True)
    shortlisted = resumes[:top_k]

    scored_results = [None] * len(shortlisted)
    with ThreadPoolExecutor(max_workers=min(LLM_MAX_WORKERS, len(shortlisted) or 1)) as executor:
        futures = {
            executor.submit(_score, job, resume): index
            for index, resume in enumerate(shortlisted)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            scored_results[futures[future]] = future.result()
            progress(
                f"Scored candidate {completed}/{len(shortlisted)}",
                0.5 + 0.4 * completed / max(len(shortlisted), 1),
            )
    results = [result for result in scored_results if result is not None]
    results.sort(key=lambda item: (item.passed_must_haves, item.overall_score), reverse=True)
    for index, result in enumerate(results, 1):
        result.rank = index
    progress("Done", 1.0)
    return RankingOutput(job, results)