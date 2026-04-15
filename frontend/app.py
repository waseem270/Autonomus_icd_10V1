import os
import streamlit as st
import requests
import pandas as pd
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
BACKEND_BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
API_URL = f"{BACKEND_BASE_URL}/api/v1"

st.set_page_config(
    page_title="ICD-10 Medical Coder",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark Theme CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark background */
    .stApp { background-color: #0e1117; }
    header[data-testid="stHeader"] { background-color: #0e1117; }

    /* Card-style containers */
    div[data-testid="stMetric"] {
        background: #1a1d24;
        border: 1px solid #2d333b;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label { color: #8b949e !important; font-size: 0.8rem; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #e6edf3 !important; }

    /* Expander */
    details { background: #161b22 !important; border: 1px solid #21262d !important; border-radius: 8px !important; }
    details summary { color: #e6edf3 !important; }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background: #238636 !important; color: #fff !important;
        border: none; border-radius: 6px; font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover { background: #2ea043 !important; }

    /* Download buttons */
    .stDownloadButton > button {
        background: #21262d !important; color: #c9d1d9 !important;
        border: 1px solid #30363d !important; border-radius: 6px;
    }
    .stDownloadButton > button:hover { border-color: #58a6ff !important; }

    /* Progress bar */
    .stProgress > div > div > div { background-color: #238636 !important; }

    /* Dataframe */
    .stDataFrame { border: 1px solid #21262d; border-radius: 8px; }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #161b22; border: 2px dashed #30363d; border-radius: 8px; padding: 20px;
    }

    /* Divider */
    hr { border-color: #21262d !important; }

    /* Status indicator */
    .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
    .status-online { background: #238636; }
    .status-offline { background: #da3633; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }

    /* Pipeline step progress */
    .pipeline-step {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 12px; border-radius: 16px; font-size: 0.8rem;
        color: #8b949e; background: #161b22; border: 1px solid #21262d;
    }
    .pipeline-step.active { color: #58a6ff; border-color: #58a6ff; }
    .pipeline-step.done { color: #3fb950; border-color: #3fb950; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────────────
for key in ("document_id", "extract_data", "structure_data", "unified_data", "filename", "pipeline_error"):
    if key not in st.session_state:
        st.session_state[key] = None

# ── Backend Status ─────────────────────────────────────────────────────────
backend_ok = False
health_data = {}
try:
    _h = requests.get(f"{BACKEND_BASE_URL}/health", timeout=2)
    backend_ok = _h.status_code == 200
    if backend_ok:
        health_data = _h.json()
except Exception:
    pass

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚕️ ICD-10 Coder")
    st.markdown("---")

    # System status
    st.markdown("**System Status**")
    if backend_ok:
        st.success(f"Backend Online  v{health_data.get('version', '1.0')}")
    else:
        st.error("Backend Offline")
        st.caption(f"Expected at: {BACKEND_BASE_URL}")

    st.markdown("---")

    # Pipeline info
    st.markdown("**Pipeline**")
    st.markdown("""
- 📤 Upload PDF  
- 🔍 Extract text (OCR)  
- 📑 Detect sections  
- 🧬 MEAT validation  
- 🏷️ ICD-10 mapping  
""")
    st.caption("Max file size: 25 MB")

    st.markdown("---")

    # Session info
    if st.session_state.unified_data:
        ud = st.session_state.unified_data
        sm = ud.get("summary", {})
        st.markdown("**Current Document**")
        if st.session_state.filename:
            st.caption(f"📎 {st.session_state.filename}")
        st.metric("Active Codes", ud.get("total_diseases", 0))
        st.metric("ICD-10 Mapped", sm.get("icd_mapped", 0))
        st.metric("Processing Time", f"{ud.get('processing_time', 0):.1f}s")

        st.markdown("---")
        st.markdown("**MEAT Tiers**")
        tier_data = {
            "🟢 Strong":   sm.get("strong_evidence", 0),
            "🟡 Moderate": sm.get("moderate_evidence", 0),
            "🟠 Weak":     sm.get("weak_evidence", 0),
            "🔴 None":     sm.get("no_meat", 0),
        }
        for label, count in tier_data.items():
            st.caption(f"{label}: **{count}**")

        st.markdown("---")
        if st.button("🔄 Analyze Another Document", use_container_width=True):
            for k in ("document_id", "extract_data", "structure_data", "unified_data", "filename", "pipeline_error"):
                st.session_state[k] = None
            st.rerun()
    else:
        st.markdown("**How it works**")
        st.markdown("""
1. Drop a clinical PDF  
2. Press **Analyze Document**  
3. Pipeline runs automatically  
4. Download billing/full CSV  
""")
        st.markdown("---")
        st.caption("LLM: GPT-5.4")
        st.caption("Evidence: MEAT criteria")
        st.caption("Codes: ICD-10-CM")

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("## ⚕️ ICD-10 Medical Coder")
st.caption("Upload a clinical PDF — automated extraction, MEAT validation & ICD-10 mapping")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# Upload & Automated Pipeline
# ═══════════════════════════════════════════════════════════════════════════
uploaded_file = st.file_uploader(
    "Drop a clinical PDF here",
    type=["pdf"],
    label_visibility="collapsed",
    help="Supported: PDF up to 25 MB",
)

if uploaded_file and not st.session_state.unified_data:
    st.caption(f"📎 **{uploaded_file.name}** — {uploaded_file.size / 1_048_576:.2f} MB")

if uploaded_file:
    if st.button("⚡ Analyze Document", type="primary", use_container_width=True):
        # Reset all state
        for key in ("document_id", "extract_data", "structure_data", "unified_data", "pipeline_error"):
            st.session_state[key] = None
        st.session_state.filename = uploaded_file.name

        progress = st.progress(0, text="Uploading document…")
        error_occurred = False

        try:
            # Step 1: Upload
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            up_resp = requests.post(f"{API_URL}/documents/upload", files=files, timeout=60)
            if up_resp.status_code != 200:
                st.error(f"Upload failed: {up_resp.text}")
                error_occurred = True

            if not error_occurred:
                doc_id = up_resp.json()["document_id"]
                st.session_state.document_id = doc_id
                progress.progress(15, text="Extracting text…")

                # Step 2: Extract
                ex_resp = requests.post(f"{API_URL}/documents/{doc_id}/extract", timeout=180)
                if ex_resp.status_code != 200:
                    st.error(f"Text extraction failed: {ex_resp.text}")
                    error_occurred = True

            if not error_occurred:
                st.session_state.extract_data = ex_resp.json()
                progress.progress(30, text="Analyzing document structure…")

                # Step 3: Structure
                st_resp = requests.post(f"{API_URL}/documents/{doc_id}/structure", timeout=300)
                if st_resp.status_code == 503:
                    st.warning("LLM API temporarily unavailable. Please retry in a moment.")
                    error_occurred = True
                elif st_resp.status_code != 200:
                    st.error(f"Structure analysis failed: {st_resp.text}")
                    error_occurred = True

            if not error_occurred:
                st.session_state.structure_data = st_resp.json()
                progress.progress(50, text="Running MEAT validation & ICD-10 mapping…")

                # Step 4: Process all (MEAT + ICD)
                proc_resp = requests.post(
                    f"{API_URL}/documents/{doc_id}/process-all",
                    params={"dietary_analysis": "false"},
                    timeout=600,
                )
                if proc_resp.status_code == 503:
                    st.warning("LLM API temporarily unavailable. Please retry in a moment.")
                    error_occurred = True
                elif proc_resp.status_code != 200:
                    st.error(f"Processing failed: {proc_resp.text}")
                    error_occurred = True

            if not error_occurred:
                st.session_state.unified_data = proc_resp.json()
                progress.progress(100, text="Complete")

        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to backend at {BACKEND_BASE_URL}")
            error_occurred = True
        except requests.exceptions.Timeout:
            st.error("Request timed out. The document may be too large or the server is busy.")
            error_occurred = True
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            error_occurred = True

        if not error_occurred:
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.unified_data:
    ud = st.session_state.unified_data
    unified_results = ud.get("unified_results", [])
    summary = ud.get("summary", {})
    processing_time = ud.get("processing_time", 0)
    total_detected = ud.get("total_detected", ud.get("total_diseases", 0))
    total_excluded = ud.get("excluded_summary", {}).get("total_excluded", 0)

    # ── Filename banner ──────────────────────────────────────────────
    if st.session_state.filename:
        st.markdown(f"**Results for** `{st.session_state.filename}`")

    # ── Key Metrics ──────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Active Diseases", ud.get("total_diseases", 0))
    m2.metric("ICD-10 Mapped", summary.get("icd_mapped", 0))
    m3.metric("Excluded", total_excluded)
    m4.metric("Processing Time", f"{processing_time:.1f}s")
    m5.metric("Source", ud.get("analysis_source", "dual_agent").replace("_", " ").title())

    # MEAT tier breakdown
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Strong Evidence", summary.get("strong_evidence", 0))
    t2.metric("Moderate Evidence", summary.get("moderate_evidence", 0))
    t3.metric("Weak Evidence", summary.get("weak_evidence", 0))
    t4.metric("No MEAT", summary.get("no_meat", 0))

    # ── Disease Table ────────────────────────────────────────────────
    _TIER_BADGE = {
        "strong_evidence":   "🟢 Strong",
        "moderate_evidence": "🟡 Moderate",
        "weak_evidence":     "🟠 Weak",
        "no_meat":           "🔴 None",
    }

    if unified_results:
        st.markdown("---")
        st.markdown("#### Disease Summary")

        table_data = []
        for r in unified_results:
            conf = r.get("confidence", 0)
            tier = r.get("meat_tier", "")
            table_data.append({
                "#":          r.get("number", ""),
                "Disease":    r.get("disease", ""),
                "ICD-10":     r.get("icd_code", "—"),
                "MEAT":       _TIER_BADGE.get(tier, tier),
                "Score":      r.get("meat_score", 0),
                "M": "✓" if r.get("monitoring")  else "",
                "E": "✓" if r.get("evaluation")  else "",
                "A": "✓" if r.get("assessment")  else "",
                "T": "✓" if r.get("treatment")   else "",
                "Confidence": f"{conf:.0%}" if isinstance(conf, (int, float)) else "—",
            })

        df = pd.DataFrame(table_data)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "#":          st.column_config.NumberColumn("#", width="small"),
                "Disease":    st.column_config.TextColumn("Disease", width="large"),
                "ICD-10":     st.column_config.TextColumn("ICD-10", width="small"),
                "MEAT":       st.column_config.TextColumn("MEAT", width="medium"),
                "Score":      st.column_config.NumberColumn("Score /4", width="small"),
                "M":          st.column_config.TextColumn("M", width="small"),
                "E":          st.column_config.TextColumn("E", width="small"),
                "A":          st.column_config.TextColumn("A", width="small"),
                "T":          st.column_config.TextColumn("T", width="small"),
                "Confidence": st.column_config.TextColumn("Conf", width="small"),
            },
        )

        # ── Disease Detail Cards ─────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Disease Details")

        for r in unified_results:
            tier       = r.get("meat_tier", "")
            tier_badge = _TIER_BADGE.get(tier, tier)
            conf       = r.get("confidence", 0)
            conf_str   = f"{conf:.0%}" if isinstance(conf, (int, float)) else "—"

            with st.expander(
                f"#{r.get('number')} {r.get('disease', '')}  —  {r.get('icd_code', '—')}  |  {tier_badge}  |  {conf_str}",
                expanded=False,
            ):
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"**ICD-10:** `{r.get('icd_code', '—')}`")
                c2.markdown(f"**MEAT Tier:** {tier_badge}")
                c3.markdown(f"**Score:** {r.get('meat_score', 0)} / 4")

                if r.get("icd_description"):
                    st.caption(r["icd_description"])

                # MEAT evidence grid
                cols = st.columns(4)
                for i, (label, key) in enumerate([
                    ("Monitoring", "monitoring_evidence"),
                    ("Evaluation", "evaluation_evidence"),
                    ("Assessment", "assessment_evidence"),
                    ("Treatment", "treatment_evidence"),
                ]):
                    with cols[i]:
                        ev = r.get(key, "")
                        if ev:
                            st.markdown(f"✅ **{label}**")
                            st.caption(ev[:250])
                        else:
                            st.markdown(f"— **{label}**")

                # Reasoning
                reasoning_parts = []
                if r.get("llm_reasoning"):
                    reasoning_parts.append(r["llm_reasoning"])
                if r.get("icd_selection_reasoning"):
                    reasoning_parts.append(r["icd_selection_reasoning"])
                if reasoning_parts:
                    st.caption(" | ".join(reasoning_parts))

        # ── Downloads ────────────────────────────────────────────────
        st.markdown("---")
        filename_stem = Path(st.session_state.filename or "document").stem

        billing_rows = []
        for r in unified_results:
            conf = r.get("confidence", 0)
            billing_rows.append({
                "Disease":       r.get("disease", ""),
                "ICD-10 Code":   r.get("icd_code", ""),
                "ICD Description": r.get("icd_description", ""),
                "MEAT Tier":     r.get("meat_tier", ""),
                "MEAT Score":    r.get("meat_score", 0),
                "M": "✓" if r.get("monitoring") else "",
                "E": "✓" if r.get("evaluation") else "",
                "A": "✓" if r.get("assessment") else "",
                "T": "✓" if r.get("treatment") else "",
                "Confidence":    f"{conf:.0%}" if isinstance(conf, (int, float)) else "—",
            })
        billing_df = pd.DataFrame(billing_rows)

        full_rows = []
        for r in unified_results:
            conf = r.get("confidence", 0)
            full_rows.append({
                "Disease":                r.get("disease", ""),
                "ICD-10 Code":            r.get("icd_code", ""),
                "ICD Description":        r.get("icd_description", ""),
                "Section":                r.get("segment", ""),
                "MEAT Tier":              r.get("meat_tier", ""),
                "MEAT Score":             r.get("meat_score", 0),
                "MEAT Status":            r.get("meat_status", ""),
                "M":                      "✓" if r.get("monitoring") else "",
                "E":                      "✓" if r.get("evaluation") else "",
                "A":                      "✓" if r.get("assessment") else "",
                "T":                      "✓" if r.get("treatment") else "",
                "Monitoring Evidence":    r.get("monitoring_evidence", ""),
                "Evaluation Evidence":    r.get("evaluation_evidence", ""),
                "Assessment Evidence":    r.get("assessment_evidence", ""),
                "Treatment Evidence":     r.get("treatment_evidence", ""),
                "Confidence":             f"{conf:.0%}" if isinstance(conf, (int, float)) else "—",
                "LLM Reasoning":          r.get("llm_reasoning", ""),
                "ICD Selection Reasoning": r.get("icd_selection_reasoning", ""),
            })
        full_df = pd.DataFrame(full_rows)

        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.download_button(
                "📋 Billing CSV",
                billing_df.to_csv(index=False),
                file_name=f"{filename_stem}_billing.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dcol2:
            st.download_button(
                "📄 Full Report CSV",
                full_df.to_csv(index=False),
                file_name=f"{filename_stem}_full_report.csv",
                mime="text/csv",
                use_container_width=True,
            )

    else:
        st.info("No active diseases found after MEAT validation.")
