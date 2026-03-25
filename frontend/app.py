import os
import streamlit as st
import requests
import pandas as pd
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
BACKEND_BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
API_URL = f"{BACKEND_BASE_URL}/api/v1"

st.set_page_config(
    page_title="Autonomous ICD-10 Coding Tool",
    page_icon="🏥",
    layout="wide"
)

# ── Session State Initialisation ────────────────────────────────────────────
for key, default in {
    "document_id": None,
    "extract_data": None,
    "structure_data": None,
    "unified_data": None,
    "filename": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 System Status")
    try:
        r = requests.get(f"{BACKEND_BASE_URL}/health", timeout=2)
        if r.status_code == 200:
            hd = r.json()
            st.success("✅ Backend Connected")
            st.caption(f"Status: {hd.get('status')} | Version: {hd.get('version')}")
        else:
            st.error("❌ Backend Error")
    except Exception:
        st.error("❌ Backend Offline")

    st.markdown("---")
    st.markdown("**3-Step Pipeline**")
    st.caption("Step 1 → Upload & Extract")
    st.caption("Step 2 → Structure & Analyze")
    st.caption("Step 3 → MEAT + ICD-10 Codes")
    st.caption("Supported: PDF | Max: 25MB")

    st.markdown("---")
    st.markdown("**🛠 Debugging logs:**")
    if Path("backend_logs.txt").exists():
        with st.expander("View Backend Logs", expanded=False):
            st.code(Path("backend_logs.txt").read_text()[-1500:], language="bash")
    else:
        st.caption("No backend logs found.")

# ── Title ───────────────────────────────────────────────────────────────────
st.title("🏥 Autonomous ICD-10 Coding Tool")
st.markdown("Upload a clinical PDF → Extract text → Detect sections & diseases → Validate MEAT & map ICD-10 codes")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Upload & Extract
# ═══════════════════════════════════════════════════════════════════════════
st.header("📄 Step 1: Upload & Extract Text")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help="Upload a medical document in PDF format"
)

if uploaded_file:
    col1, col2 = st.columns(2)
    col1.info(f"**Filename:** {uploaded_file.name}")
    col2.info(f"**Size:** {uploaded_file.size / 1_048_576:.2f} MB")

    if st.button("🚀 Upload & Extract Text", type="primary"):
        # Reset downstream state
        st.session_state.document_id = None
        st.session_state.extract_data = None
        st.session_state.structure_data = None
        st.session_state.unified_data = None
        st.session_state.filename = uploaded_file.name

        with st.spinner("Uploading document…"):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                up_resp = requests.post(f"{API_URL}/documents/upload", files=files, timeout=60)

                if up_resp.status_code == 200:
                    up_data = up_resp.json()
                    doc_id = up_data["document_id"]
                    st.session_state.document_id = doc_id
                    st.success("✅ Uploaded successfully!")
                    st.caption(f"Document ID: {doc_id}")

                    with st.spinner("Extracting text (OCR if needed)…"):
                        ex_resp = requests.post(f"{API_URL}/documents/{doc_id}/extract", timeout=180)
                        if ex_resp.status_code == 200:
                            st.session_state.extract_data = ex_resp.json()
                            st.success("✅ Text extracted!")
                        else:
                            st.error(f"❌ Extraction failed: {ex_resp.text}")
                else:
                    st.error(f"❌ Upload failed: {up_resp.text}")
            except Exception as e:
                st.error(f"❌ Connection Error: {e}")
                st.info(f"Ensure FastAPI backend is running on {BACKEND_BASE_URL}")

# ── Extraction Results ───────────────────────────────────────────────────
if st.session_state.extract_data:
    ed = st.session_state.extract_data

    st.divider()
    st.subheader("🏁 Extraction Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pages", ed["page_count"])
    m2.metric("Method", ed["extraction_method"].upper())
    m3.metric("Confidence", f"{ed['confidence_score']:.1%}")
    m4.metric("Quality", f"{ed['quality_score']:.1%}")

    if "processing_time" in ed:
        st.caption(f"Processing time: {ed['processing_time']:.2f}s")

    with st.expander("📝 View Extracted Text"):
        st.text_area("Raw Text", ed["raw_text"], height=200, disabled=False)

    st.download_button(
        "⬇️ Download Extracted Text",
        data=ed["raw_text"],
        file_name=f"{Path(st.session_state.filename or 'doc').stem}_extracted.txt",
        mime="text/plain"
    )

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Structure & Analyze (Sections + Diseases)
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.extract_data and st.session_state.document_id:
    st.markdown("---")
    st.header("📑 Step 2: Structure & Analyze")
    st.info("Detect clinical sections and their content, then extract all diseases from the document.")

    if st.button("🔍 Detect Sections & Diseases", type="primary"):
        # Reset step 3 data
        st.session_state.structure_data = None
        st.session_state.unified_data = None

        doc_id = st.session_state.document_id
        with st.spinner("Analyzing document structure and detecting diseases… This may take 1-2 minutes."):
            try:
                resp = requests.post(f"{API_URL}/documents/{doc_id}/structure", timeout=300)
                if resp.status_code == 200:
                    st.session_state.structure_data = resp.json()
                    st.success("✅ Sections & diseases detected!")
                elif resp.status_code == 503:
                    st.warning(
                        "⚠️ Gemini API is temporarily unavailable (high demand). "
                        "All retry attempts were exhausted. Please wait a minute and try again."
                    )
                else:
                    st.error(f"❌ Structuring failed: {resp.text}")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ── Structure Results ────────────────────────────────────────────────────
if st.session_state.structure_data:
    sd = st.session_state.structure_data

    st.divider()
    st.subheader("📊 Structure & Analysis Results")

    # Summary metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Sections Found", sd.get("sections_found", 0))
    m2.metric("Total Diseases", sd.get("total_diseases", 0))
    m3.metric("Total Sentences", sd.get("total_sentences", 0))

    if sd.get("processing_stats"):
        ps = sd["processing_stats"]
        st.caption(
            f"Processing: {ps.get('total_processing_time', 0):.2f}s | "
            f"NER: {ps.get('ner_diseases', 0)} | "
            f"LLM: {ps.get('llm_diseases', 0)}"
        )

    # ── Sections with Content ────────────────────────────────────────
    sections = sd.get("sections", {})
    if sections:
        st.markdown("#### 📂 Detected Sections & Content")
        for sec_name, sec_data in sections.items():
            display_name = sec_name.replace("_", " ").title()
            sec_text = sec_data.get("text", "") if isinstance(sec_data, dict) else str(sec_data)

            with st.expander(f"📌 {display_name}", expanded=False):
                st.text_area(
                    f"Content — {display_name}",
                    sec_text,
                    height=150,
                    disabled=True,
                    key=f"sec_{sec_name}"
                )

    # Disease detection happens internally — results shown after MEAT validation in Step 3
    diseases_count = len(sd.get("detected_diseases", []))
    if diseases_count:
        st.success(f"✅ {diseases_count} potential diseases identified. Proceed to Step 3 for MEAT validation & ICD mapping.")
    else:
        st.warning("No diseases detected in the document.")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: MEAT Validation + ICD-10 Codes
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.structure_data and st.session_state.document_id:
    st.markdown("---")
    st.header("🧬 Step 3: MEAT Validation + ICD-10 Codes")
    st.info("Validate MEAT criteria (Monitoring, Evaluation, Assessment, Treatment) and map ICD-10 codes for all detected diseases.")

    dietary_toggle = st.checkbox(
        "🥗 Include Dietary Analysis (Phase 8)",
        value=False,
        help="When enabled, Phase 8 dietary/nutritional analysis annotations are included. Disabled by default to avoid filtering valid metabolic diseases."
    )

    if st.button("✅ Run MEAT + ICD Mapping", type="primary"):
        st.session_state.unified_data = None

        doc_id = st.session_state.document_id
        with st.spinner("Running MEAT validation & ICD mapping… This may take 1-3 minutes."):
            try:
                resp = requests.post(
                    f"{API_URL}/documents/{doc_id}/process-all",
                    params={"dietary_analysis": str(dietary_toggle).lower()},
                    timeout=600
                )
                if resp.status_code == 200:
                    st.session_state.unified_data = resp.json()
                    st.success("✅ MEAT validation & ICD mapping complete!")
                elif resp.status_code == 503:
                    st.warning(
                        "⚠️ Gemini API is temporarily unavailable (high demand). "
                        "All retry attempts were exhausted. Please wait a minute and try again."
                    )
                else:
                    st.error(f"❌ Processing failed: {resp.text}")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# Show Active MEAT-Based Diseases
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.unified_data:
    ud = st.session_state.unified_data
    unified_results = ud.get("unified_results", [])
    summary = ud.get("summary", {})

    st.divider()

    # ── Summary Metrics ──────────────────────────────────────────────
    excluded = ud.get("excluded_summary", {})
    total_detected = ud.get("total_detected", ud["total_diseases"])
    total_excluded = excluded.get("total_excluded", 0)

    m1, m2, m3 = st.columns(3)
    m1.metric("Active MEAT Diseases", ud["total_diseases"])
    m2.metric("🔢 ICD Mapped", summary.get("icd_mapped", 0))
    m3.metric("🚫 Excluded", total_excluded)

    st.caption(f"Processing time: {ud.get('processing_time', 0):.2f}s | Detected: {total_detected} → Active (Full MEAT): {ud['total_diseases']}")

    # ── Active MEAT-Based Diseases Table (Screenshot Format) ─────────
    st.markdown("---")
    st.subheader("Active MEAT-Based Diseases")

    if unified_results:
        table_data = []
        for r in unified_results:
            conf = r.get("confidence", 0)
            table_data.append({
                "Disease": r["disease"],
                "ICD-10 Code": r["icd_code"],
                "Segment": r["segment"],
                "Monitor": r.get("monitoring_evidence", "")[:80] or "—",
                "Evaluate": r.get("evaluation_evidence", "")[:80] or "—",
                "Assess": r.get("assessment_evidence", "")[:80] or "—",
                "Treatment": r.get("treatment_evidence", "")[:80] or "—",
                "MEAT Level": r.get("meat_level", ""),
                "Confidence": f"{conf:.0%}" if isinstance(conf, (int, float)) else str(conf),
            })

        df = pd.DataFrame(table_data)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Disease": st.column_config.TextColumn("Disease", width="medium"),
                "ICD-10 Code": st.column_config.TextColumn("ICD-10 Code", width="small"),
                "Segment": st.column_config.TextColumn("Segment", width="small"),
                "Monitor": st.column_config.TextColumn("Monitor", width="medium"),
                "Evaluate": st.column_config.TextColumn("Evaluate", width="medium"),
                "Assess": st.column_config.TextColumn("Assess", width="medium"),
                "Treatment": st.column_config.TextColumn("Treatment", width="medium"),
                "MEAT Level": st.column_config.TextColumn("MEAT Level", width="small"),
                "Confidence": st.column_config.TextColumn("Confidence", width="small"),
            }
        )

        # ── Detailed Expandable Cards ────────────────────────────────
        st.markdown("---")
        st.subheader("🔍 Detailed Disease Analysis")

        for r in unified_results:
            with st.expander(
                f"#{r['number']} {r['disease']} → {r['icd_code']} | {r['disease_status']}",
                expanded=False
            ):
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"**ICD-10:** {r['icd_code']}")
                c2.markdown(f"**Status:** {r['disease_status']}")
                c3.markdown(f"**Segment:** {r['segment']}")

                st.markdown("**MEAT Evidence:**")
                g1, g2, g3, g4 = st.columns(4)
                with g1:
                    st.markdown("✅ **Monitoring**")
                    if r.get("monitoring_evidence"):
                        st.caption(r["monitoring_evidence"][:200])
                with g2:
                    st.markdown("✅ **Evaluation**")
                    if r.get("evaluation_evidence"):
                        st.caption(r["evaluation_evidence"][:200])
                with g3:
                    st.markdown("✅ **Assessment**")
                    if r.get("assessment_evidence"):
                        st.caption(r["assessment_evidence"][:200])
                with g4:
                    st.markdown("✅ **Treatment**")
                    if r.get("treatment_evidence"):
                        st.caption(r["treatment_evidence"][:200])

                if r.get("llm_reasoning"):
                    st.caption(f"💡 Reasoning: {r['llm_reasoning']}")

                if r.get("icd_description"):
                    st.caption(f"📖 ICD Description: {r['icd_description']}")

        # ── Download Options ─────────────────────────────────────────
        st.markdown("---")
        st.subheader("⬇️ Download Results")

        # 8-column billing-ready format
        export_rows = []
        for r in unified_results:
            conf = r.get("confidence", 0)
            export_rows.append({
                "Disease": r["disease"],
                "ICD-10 Code": r["icd_code"],
                "Segment": r["segment"],
                "Monitor": r.get("monitoring_evidence", ""),
                "Evaluate": r.get("evaluation_evidence", ""),
                "Assess": r.get("assessment_evidence", ""),
                "Treatment": r.get("treatment_evidence", ""),
                "MEAT Level": r.get("meat_level", ""),
                "Confidence": f"{conf:.0%}" if isinstance(conf, (int, float)) else str(conf),
            })

        export_df = pd.DataFrame(export_rows)
        filename_stem = Path(st.session_state.filename or "document").stem

        dcol1, dcol2, dcol3 = st.columns(3)

        with dcol1:
            st.download_button(
                "⬇️ Billing Ready (CSV)",
                export_df.to_csv(index=False),
                file_name=f"{filename_stem}_billing_ready.csv",
                mime="text/csv",
                help="8-column CSV: Disease, ICD-10 Code, Segment, Monitor, Evaluate, Assess, Treatment, MEAT Level"
            )

        with dcol2:
            # Full report includes ICD description
            full_rows = []
            for r in unified_results:
                conf = r.get("confidence", 0)
                full_rows.append({
                    "Disease": r["disease"],
                    "ICD-10 Code": r["icd_code"],
                    "ICD Description": r.get("icd_description", ""),
                    "Segment": r["segment"],
                    "Monitor": r.get("monitoring_evidence", ""),
                    "Evaluate": r.get("evaluation_evidence", ""),
                    "Assess": r.get("assessment_evidence", ""),
                    "Treatment": r.get("treatment_evidence", ""),
                    "MEAT Level": r.get("meat_level", ""),
                    "Confidence": f"{conf:.0%}" if isinstance(conf, (int, float)) else str(conf),
                    "LLM Reasoning": r.get("llm_reasoning", ""),
                })
            full_df = pd.DataFrame(full_rows)
            st.download_button(
                "⬇️ Full Report (CSV)",
                full_df.to_csv(index=False),
                file_name=f"{filename_stem}_full_report.csv",
                mime="text/csv",
                help="Full report including ICD descriptions and LLM reasoning"
            )

        with dcol3:
            # Server-side CSV export via backend endpoint
            doc_id = st.session_state.document_id
            try:
                api_csv = requests.get(
                    f"{API_URL}/documents/{doc_id}/export-csv",
                    timeout=30
                )
                if api_csv.status_code == 200:
                    st.download_button(
                        "⬇️ Server Export (CSV)",
                        api_csv.content,
                        file_name=f"{filename_stem}_server_export.csv",
                        mime="text/csv",
                        help="CSV generated directly from the server database"
                    )
                else:
                    st.caption("Server export unavailable")
            except Exception:
                st.caption("Server export unavailable")

    else:
        st.warning("No active MEAT-based diseases found.")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Autonomous ICD-10 Coding Tool | Developed for clinical document intelligence")
