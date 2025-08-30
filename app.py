import os
import json
import tempfile
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

from utils import (
    analyze_single,
    compute_similarity,
    find_matches_with_context,
    SPACY_AVAILABLE,
)

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Smart Document Analyzer (Upgraded)", layout="wide")
st.title("🧠 Smart Document Analyzer — Upgraded (Free)")

st.markdown("""
Small, privacy-first document analyzer. Upload files (PDF/DOCX/TXT/CSV/LOG), run OCR on scanned PDFs,
extract keywords, summary, entities (spaCy), sentiment (VADER), word-frequency chart, search with context,
and compare multiple documents for similarity.
""")

with st.sidebar:
    st.header("Analysis settings")
    max_sum = st.slider("Max summary sentences", 1, 15, 5)
    max_kw = st.slider("Max keywords", 3, 30, 10)
    ocr = st.checkbox("Enable OCR for scanned PDFs", value=True)
    ocr_pages = st.slider("OCR first N pages", 1, 10, 3)
    context_chars = st.slider("Search context characters", 20, 300, 80)
    show_entities = st.checkbox("Attempt Named Entity Recognition (spaCy)", value=SPACY_AVAILABLE)
    detect_duplicates = st.checkbox("Detect duplicates / similarity", value=True)

uploaded_files = st.file_uploader(
    "Upload files", type=["pdf", "docx", "txt", "md", "csv", "log"], accept_multiple_files=True
)

tmpdir = Path(tempfile.mkdtemp(prefix="sda_"))
analyze_now = st.button("Analyze uploaded files") if uploaded_files else False

if show_entities and not SPACY_AVAILABLE:
    st.warning("spaCy not installed. Run: `pip install spacy && python -m spacy download en_core_web_sm`")

reports, paths = [], []

if analyze_now:
    with st.spinner("Saving files and analyzing..."):
        try:
            for uf in uploaded_files:
                dest = tmpdir / uf.name
                with open(dest, "wb") as f:
                    f.write(uf.getbuffer())
                paths.append(str(dest))

            for p in paths:
                reports.append(analyze_single(p, ocr=ocr, ocr_pages=ocr_pages, max_sum=max_sum, max_kw=max_kw))

            sims = compute_similarity([r["full_text"] for r in reports]) if detect_duplicates and len(reports) > 1 else []
            st.success("Analysis complete.")
        except Exception as e:
            st.error(f"Error: {e}")
            sims = []

    for idx, rep in enumerate(reports):
        st.subheader(f"{idx+1}. {rep['file_name']} — {rep['file_type']}")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("**Summary:**")
            for s in rep["summary"]: st.write("•", s)

            st.markdown("**Keywords:**")
            st.write(", ".join(rep["keywords"]) if rep["keywords"] else "—")

            st.markdown("**Top words:**")
            df_top = pd.DataFrame(rep["top_words"], columns=["word", "count"])
            st.dataframe(df_top.head(20))
            st.bar_chart(df_top.set_index("word")["count"])

            st.markdown("**Sentiment:**")
            st.write(rep["sentiment"])

            if show_entities and rep["entities"]:
                st.markdown("**Named Entities (spaCy):**")
                st.dataframe(pd.DataFrame(rep["entities"]))

            st.markdown("**Preview:**")
            st.code(rep["preview"][:1000])

            st.markdown("**Search:**")
            q = st.text_input(f"Search term in {rep['file_name']}", key=f"search_{idx}")
            if q:
                matches = find_matches_with_context(rep["full_text"], q, context_chars)
                for m in matches[:20]: st.write("...", m, "...")

        with col2:
            st.markdown("**File stats**")
            st.write(f"Size: {rep['size_bytes']} bytes")
            st.write(f"Language: {rep['language']}")
            st.write(f"Words: {rep['word_count']}")
            st.write(f"Sentences: {rep['sentence_count']}")
            st.write(f"Reading time: {rep['reading_time_min']} min")

            st.download_button(
                "Download JSON",
                json.dumps({k:v for k,v in rep.items() if k!='full_text'}, indent=2),
                f"{Path(rep['file_name']).stem}_report.json",
                "application/json"
            )
            st.download_button(
                "Download Full Text",
                rep["full_text"],
                f"{Path(rep['file_name']).stem}_fulltext.txt"
            )

    if sims:
        st.subheader("📎 Document similarity")
        sim_df = pd.DataFrame(sims)
        sim_df["doc_i"] = sim_df["i"].apply(lambda x: reports[x]["file_name"])
        sim_df["doc_j"] = sim_df["j"].apply(lambda x: reports[x]["file_name"])
        st.dataframe(sim_df[["doc_i", "doc_j", "score"]])

    agg = pd.DataFrame([{
        "file_name": r["file_name"],
        "file_type": r["file_type"],
        "size_bytes": r["size_bytes"],
        "language": r["language"],
        "word_count": r["word_count"],
        "sentence_count": r["sentence_count"],
        "reading_time_min": r["reading_time_min"],
        "top_keywords": "|".join(r["keywords"]),
        "sentiment_compound": r["sentiment"]["compound"],
    } for r in reports])
    st.subheader("Export aggregated report")
    st.dataframe(agg)
    st.download_button("Download CSV", agg.to_csv(index=False).encode("utf-8"), "aggregated_report.csv", "text/csv")

    try: shutil.rmtree(tmpdir)
    except: pass
else:
    st.info("Upload files and click 'Analyze uploaded files'.")

