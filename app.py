import streamlit as st
from dotenv import load_dotenv
import os
import tempfile

# Markdown + PDF
import markdown
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Crew runners
from crew.data_analyzer_crew import (
    run_validator,
    run_problem_analyzer,
    run_data_cleaner,
    run_data_analyzer,
    run_insight_generator,
    run_executive_report
)

# ----------------------------
# SESSION STATE (CRITICAL)
# ----------------------------
if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

if "results" not in st.session_state:
    st.session_state.results = None

if "file_path" not in st.session_state:
    st.session_state.file_path = None

# ----------------------------
# ENV
# ----------------------------
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY missing")
    st.stop()

# ----------------------------
# UI CONFIG
# ----------------------------
st.set_page_config(
    page_title="Data Analyzer AI",
    layout="centered"
)

st.title("🧠 Data Analyzer AI")
st.write(
    "Do not close or refresh. You may switch tabs safely while agents are running."
)

# ----------------------------
# INPUTS
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload dataset",
    type=["csv", "xlsx", "xls"]
)

problem_statement = st.text_area(
    "Describe your problem statement",
    placeholder="e.g. Validate data quality and generate business insights"
)

# ----------------------------
# HELPERS
# ----------------------------
def save_temp_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def raw_to_markdown(raw_text: str) -> str:
    return raw_text.strip()


def save_markdown(md_text: str, base_name: str) -> str:
    os.makedirs("outputs/reports", exist_ok=True)
    path = f"outputs/reports/{base_name}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(md_text)
    return path


def markdown_to_pdf(md_text: str, pdf_path: str):
    html = markdown.markdown(md_text)
    styles = getSampleStyleSheet()
    story = [Paragraph(html, styles["Normal"])]
    pdf = SimpleDocTemplate(pdf_path)
    pdf.build(story)

# ----------------------------
# START BUTTON
# ----------------------------
if (
    uploaded_file
    and problem_statement.strip()
    and not st.session_state.analysis_started
):
    if st.button("🚀 Start Analysis (10–15 mins)"):
        st.session_state.analysis_started = True
        st.session_state.file_path = save_temp_file(uploaded_file)
        st.session_state.results = None
        st.rerun()

# ----------------------------
# RUN PIPELINE (ONCE)
# ----------------------------
if st.session_state.analysis_started and st.session_state.results is None:
    progress = st.progress(0)
    status = st.empty()

    results = {}
    file_path = st.session_state.file_path

    status.info("🟢 Agent 1: Validating input file...")
    results["validation"] = run_validator(file_path, problem_statement)
    progress.progress(15)

    status.info("🟡 Agent 2: Understanding problem statement...")
    results["problem_analysis"] = run_problem_analyzer(file_path, problem_statement)
    progress.progress(30)

    status.info("🟡 Agent 3: Cleaning data...")
    results["cleaning"] = run_data_cleaner(file_path, problem_statement)
    progress.progress(50)

    status.info("🟡 Agent 4: Analyzing data...")
    results["analysis"] = run_data_analyzer(file_path, problem_statement)
    progress.progress(70)

    status.info("🟡 Agent 6: Generating insights...")
    results["insights"] = run_insight_generator(file_path, problem_statement)
    progress.progress(85)

    status.info("🟢 Agent 9: Creating executive report...")
    results["report"] = run_executive_report(file_path, problem_statement)
    progress.progress(100)

    status.success("✅ All agents finished")

    st.session_state.results = results
    st.rerun()

# ----------------------------
# SHOW RESULTS (PERSISTENT)
# ----------------------------
if st.session_state.results:
    st.markdown("## 📊 Agent-wise Results")

    for agent, output in st.session_state.results.items():
        with st.expander(agent.upper()):
            if hasattr(output, "raw"):
                st.text(output.raw)
            else:
                st.text(str(output))

    # ----------------------------
    # FINAL REPORT (RAW → MD → PDF)
    # ----------------------------
    st.markdown("## 📄 Final Executive Report")

    crew_output = st.session_state.results.get("report")
    raw_report = crew_output.raw if crew_output else ""

    md_report = raw_to_markdown(raw_report)

    base_name = "Executive_Report"
    md_path = save_markdown(md_report, base_name)
    pdf_path = f"outputs/reports/{base_name}.pdf"

    markdown_to_pdf(md_report, pdf_path)

    st.download_button(
        "⬇️ Download Markdown Report",
        data=md_report,
        file_name=f"{base_name}.md",
        mime="text/markdown"
    )

    with open(pdf_path, "rb") as f:
        st.download_button(
            "⬇️ Download PDF Report",
            data=f,
            file_name=f"{base_name}.pdf",
            mime="application/pdf"
        )
