"""Command-line entry point for the CV Sorting capstone submission."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Collect data, provider, model, and credential parameters."""
    parser = argparse.ArgumentParser(description="Rank ten CVs against one job description.")
    parser.add_argument("--jd", default="job_description.txt")
    parser.add_argument("--output", default="ranked_list.csv")
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""))
    light_default = os.getenv("LIGHT_LLM_MODEL", "gpt-4o-mini")
    heavy_default = os.getenv("HEAVY_LLM_MODEL", "gpt-5.2")

    # LIGHT LLM: cheaper/faster model used only for JD and CV extraction.
    parser.add_argument("--light-model", default=light_default)

    # HEAVY LLM: stronger model used for evidence-based candidate scoring.
    parser.add_argument("--heavy-model", default=heavy_default)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("LLM_MAX_WORKERS", "4")),
        help="Concurrent LLM calls; reduce this value if the provider rate-limits requests.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Discard saved model responses and create a new scoring baseline.",
    )
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    """Expose command-line settings to the provider factory before import."""
    if not args.light_model or not args.heavy_model:
        raise ValueError("Both --light-model and --heavy-model are required.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    values = {
        "LLM_BASE_URL": args.base_url,
        "LLM_API_KEY": args.api_key,
        "LIGHT_LLM_PROVIDER": "openai_compatible",
        "LIGHT_LLM_MODEL": args.light_model,
        "HEAVY_LLM_PROVIDER": "openai_compatible",
        "HEAVY_LLM_MODEL": args.heavy_model,
        "LLM_MAX_WORKERS": str(args.workers),
    }
    for name, value in values.items():
        if value:
            os.environ[name] = value


def load_inputs(jd_path: Path, output_path: Path) -> tuple[str, list[tuple[str, str]]]:
    """Read the single JD and all flat TXT resume files in the current directory."""
    resume_paths = sorted(Path(".").glob("*_Resume.txt"))
    if len(resume_paths) != 10:
        raise ValueError(
            f"Expected exactly 10 files ending with _Resume.txt; found {len(resume_paths)}."
        )
    resumes = [(path.name, path.read_text(encoding="utf-8")) for path in resume_paths]
    return jd_path.read_text(encoding="utf-8"), resumes


def write_results(output_path: Path, results: list) -> None:
    """Write the ranked candidate list required by the assignment."""
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["rank", "candidate", "overall_score", "passed_must_haves", "summary"])
        for result in results:
            writer.writerow([
                result.rank,
                result.name or result.candidate_id,
                result.overall_score,
                result.passed_must_haves,
                result.summary,
            ])


def main() -> None:
    """Configure model roles, run ranking, and save the output file."""
    args = parse_args()
    configure_environment(args)

    # Import after configuration because core constructs both provider clients.
    from core import rank_resumes
    from llm_provider import clear_response_cache

    if args.refresh_cache:
        clear_response_cache()
        print("CACHE      : refreshed; this run creates a new baseline")
    else:
        print("CACHE      : enabled; unchanged inputs reuse the reviewed baseline")

    jd_text, resumes = load_inputs(Path(args.jd), Path(args.output))
    print(f"LIGHT LLM  : {args.light_model}")
    print(f"HEAVY LLM  : {args.heavy_model}")
    print(f"CONCURRENCY: {args.workers} workers")
    ranked = rank_resumes(
        jd_text,
        resumes,
        top_k=10,
        progress=lambda message, fraction: print(f"{fraction:>5.0%}  {message}"),
    )
    write_results(Path(args.output), ranked.results)
    print(f"Saved {len(ranked.results)} ranked candidates to {args.output}")


if __name__ == "__main__":
    main()