"""Professional Streamlit dashboard for explainable CV ranking."""
from dataclasses import asdict

import pandas as pd
import streamlit as st

from core import PROVIDER, SHORTLIST_TOP_K, extract_text, rank_resumes
from llm_provider import clear_response_cache

st.set_page_config(
    page_title="Candidate Match Console",
    page_icon="CM",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown("""
<style>
    :root {
        --ink: #17324d;
        --teal: #147d80;
        --line: #dbe4e8;
        --paper: #f5f8f8;
    }
    .stApp { background: var(--paper); }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] h2 { color: var(--ink); font-size: 1rem; }
    .block-container { max-width: 1240px; padding-top: 3.5rem; padding-bottom: 3rem; }
    .app-header { border-bottom: 1px solid var(--line); padding-bottom: 1.25rem; margin-bottom: 1.5rem; }
    .app-kicker { color: var(--teal); font-size: .76rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
    .app-title { color: var(--ink); font-family: Georgia, serif; font-size: 2.15rem; line-height: 1.15; margin: .2rem 0 .35rem; }
    .app-subtitle { color: #536773; font-size: .98rem; margin: 0; }
    .provider-label { display: inline-block; margin-top: .8rem; color: #2e525e; font-size: .8rem; font-weight: 600; }
    div[data-testid="stMetric"] { background: #ffffff; border-top: 3px solid var(--teal); padding: .85rem 1rem; }
    div[data-testid="stMetricLabel"] { color: #52636b; }
    div[data-testid="stDataFrame"] { border: 1px solid var(--line); }
    .stButton > button, .stDownloadButton > button { border-radius: 4px; font-weight: 650; }
    .stButton > button[kind="primary"] { background: var(--ink); border-color: var(--ink); }
    .stButton > button[kind="primary"]:hover { background: #234a68; border-color: #234a68; }
    details { background: #ffffff; border: 1px solid var(--line) !important; border-radius: 4px !important; }
    .empty-state { background: #ffffff; border: 1px dashed #b9c9ce; padding: 2.2rem; text-align: center; color: #58707a; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="app-header">
        <div class="app-kicker">Talent Decision Support</div>
        <div class="app-title">Candidate Match Console</div>
        <p class="app-subtitle">Evidence-led resume ranking against role-specific requirements.</p>
        <span class="provider-label">LLM provider: {PROVIDER.upper()}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("1. Role profile")
    st.caption("Paste a job description or upload a document.")
    jd_text = st.text_area("Job description", height=200, placeholder="Paste the role description here...")
    jd_file = st.file_uploader("Upload role document", type=["txt", "pdf", "docx"])
    if jd_file:
        jd_text = extract_text(jd_file.getvalue(), jd_file.name)
        st.success(f"Loaded {jd_file.name}")

    st.divider()
    st.header("2. Candidate pool")
    st.caption("Upload the resumes to compare with this role.")
    resume_files = st.file_uploader(
        "Candidate resumes", type=["txt", "pdf", "docx"], accept_multiple_files=True
    )
    if resume_files:
        st.caption(f"{len(resume_files)} resume{'s' if len(resume_files) != 1 else ''} ready")

    st.divider()
    st.header("3. Analysis")
    max_shortlist = max(1, len(resume_files) if resume_files else 100)
    top_k = st.slider(
        "Candidates to score",
        1,
        max_shortlist,
        min(SHORTLIST_TOP_K, max_shortlist),
        help="The lexical shortlist sent to the evidence-scoring model.",
    )
    refresh_cache = st.checkbox(
        "Create a fresh scoring baseline",
        help="Normally, unchanged inputs reuse reviewed model responses for stable results.",
    )
    run = st.button("Run candidate ranking", type="primary", width="stretch")

if run:
    if not jd_text.strip():
        st.error("Please provide a job description.")
    elif not resume_files:
        st.error("Please upload at least one resume.")
    else:
        try:
            if refresh_cache:
                clear_response_cache()
            documents = [(file.name, extract_text(file.getvalue(), file.name)) for file in resume_files]
            bar = st.progress(0.0, text="Preparing analysis")

            def update_progress(message: str, value: float) -> None:
                bar.progress(min(value, 1.0), text=message)

            st.session_state["output"] = rank_resumes(
                jd_text,
                documents,
                top_k,
                update_progress,
            )
            bar.empty()
        except Exception as error:
            st.error(f"Ranking failed: {error}")

output = st.session_state.get("output")
if output:
    job, results = output.job, output.results
    st.subheader(job.title or "Candidate ranking")
    st.caption(job.summary or "Ranked against the extracted role requirements.")
    st.caption("Stable baseline mode: unchanged files reuse the same structured model responses.")

    passing = sum(result.passed_must_haves for result in results)
    average_score = sum(result.overall_score for result in results) / len(results) if results else 0
    best_score = results[0].overall_score if results else 0
    metric_columns = st.columns(4)
    metric_columns[0].metric("Candidates scored", len(results))
    metric_columns[1].metric("Requirements", len(job.requirements))
    metric_columns[2].metric("Top score", f"{best_score:.1f}")
    metric_columns[3].metric("Must-have passes", f"{passing} / {len(results)}", f"Avg {average_score:.1f}", delta_color="off")

    table = pd.DataFrame([{
        "Rank": result.rank,
        "Candidate": result.name or result.candidate_id,
        "Fit score": result.overall_score,
        "Must-have review": "Passed" if result.passed_must_haves else "Attention required",
    } for result in results])
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn(width="small", format="%d"),
            "Fit score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
        },
    )
    left, right = st.columns([1, 4])
    with left:
        st.download_button(
            "Download results",
            pd.DataFrame([asdict(result) for result in results]).to_csv(index=False).encode(),
            "candidate_ranking.csv",
            "text/csv",
            width="stretch",
        )
    with right:
        st.caption("Review requirement evidence before making any recruitment decision.")

    labels = {item.id: item.text for item in job.requirements}
    st.subheader("Evidence review")
    for result in results:
        status = "Passed must-haves" if result.passed_must_haves else "Review must-haves"
        with st.expander(f"#{result.rank}  {result.name or result.candidate_id}  |  {result.overall_score:.1f}  |  {status}"):
            st.write(result.summary)
            st.dataframe(pd.DataFrame([{
                "Requirement": labels.get(score.requirement_id, score.requirement_id),
                "Score": round(score.score, 2),
                "Evidence": score.evidence,
                "Assessment": score.reasoning,
            } for score in result.breakdown]), width="stretch", hide_index=True)
else:
    st.markdown(
        """
        <div class="empty-state">
            <strong>No analysis yet</strong><br/>
            Add a role profile and candidate resumes in the sidebar, then run the ranking.
        </div>
        """,
        unsafe_allow_html=True,
    )