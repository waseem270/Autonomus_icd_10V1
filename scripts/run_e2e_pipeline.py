"""
End-to-End Pipeline Runner — connects PostgreSQL, creates tables,
processes test PDFs, detects/fixes bugs, and produces an accuracy report.

Usage:
    python -m scripts.run_e2e_pipeline
"""

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

# ── Add project root to path ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ── Logging (plain-text for this script) ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("e2e_runner")

# ════════════════════════════════════════════════════════════════════
# STEP 1 + 2 — Database Connection
# ════════════════════════════════════════════════════════════════════

def connect_database() -> Dict[str, Any]:
    """Connect to PostgreSQL and return engine + session factory."""
    from backend.app.core.database import engine, SessionLocal, create_tables
    from sqlalchemy import text, inspect

    result = {
        "connected": False,
        "tables_created": [],
        "db_type": "unknown",
        "host": "",
    }

    url_str = str(engine.url)
    parsed = urlparse(url_str)
    result["db_type"] = "postgresql" if "postgres" in url_str else "sqlite"
    result["host"] = parsed.hostname or "local"

    # Test connection
    try:
        with engine.connect() as conn:
            row = conn.execute(text("SELECT 1")).fetchone()
            assert row[0] == 1
        result["connected"] = True
        logger.info(f"✅ Connected to {result['db_type']} at {result['host']}")
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        raise

    # STEP 3 — Create tables
    create_tables()
    inspector = inspect(engine)
    result["tables_created"] = sorted(inspector.get_table_names())
    logger.info(f"✅ Tables: {result['tables_created']}")

    return result


# ════════════════════════════════════════════════════════════════════
# STEP 4 — SQLite Migration (optional)
# ════════════════════════════════════════════════════════════════════

def migrate_sqlite_if_exists() -> Dict[str, Any]:
    """If a SQLite DB exists, copy documents into PostgreSQL."""
    from backend.app.core.database import engine as pg_engine, SessionLocal
    from backend.app.models.document import Document
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    stats = {"migrated": False, "documents_copied": 0}

    sqlite_path = PROJECT_ROOT / "database" / "medical_icd.db"
    if not sqlite_path.exists():
        logger.info("No SQLite DB found — skipping migration.")
        return stats

    # Only migrate if PostgreSQL, not if still on SQLite
    url_str = str(pg_engine.url)
    if "sqlite" in url_str:
        logger.info("Target is SQLite — skipping migration.")
        return stats

    sqlite_url = f"sqlite:///{sqlite_path}"
    sqlite_engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})
    SQLiteSession = sessionmaker(bind=sqlite_engine)

    try:
        sqlite_db = SQLiteSession()
        pg_db = SessionLocal()

        existing_count = pg_db.query(Document).count()
        if existing_count > 0:
            logger.info(f"PostgreSQL already has {existing_count} documents — skip migration.")
            stats["migrated"] = True
            stats["documents_copied"] = existing_count
            return stats

        # Copy documents (simple — just documents table for now)
        from sqlalchemy import text as sql_text
        rows = sqlite_db.execute(sql_text("SELECT * FROM documents")).fetchall()
        cols = sqlite_db.execute(sql_text("SELECT * FROM documents LIMIT 0")).keys()
        col_list = list(cols)

        for row in rows:
            data = dict(zip(col_list, row))
            pg_db.execute(
                Document.__table__.insert().values(**data)
            )

        pg_db.commit()
        stats["migrated"] = True
        stats["documents_copied"] = len(rows)
        logger.info(f"✅ Migrated {len(rows)} documents from SQLite → PostgreSQL")
    except Exception as e:
        logger.warning(f"SQLite migration skipped/failed: {e}")
    finally:
        sqlite_db.close()
        pg_db.close()

    return stats


# ════════════════════════════════════════════════════════════════════
# STEP 5 + 6 — End-to-End Pipeline Run
# ════════════════════════════════════════════════════════════════════

async def process_single_pdf(
    pdf_path: str,
    db_session,
) -> Dict[str, Any]:
    """Run the full pipeline on a single PDF and return results + metrics."""
    from backend.app.models.document import Document
    from backend.app.models.enums import DocumentStatus
    from backend.app.services.pipeline_orchestrator import pipeline_orchestrator

    filename = os.path.basename(pdf_path)
    result = {
        "filename": filename,
        "success": False,
        "error": None,
        "diseases_detected": 0,
        "diseases_after_filter": 0,
        "icd_mapped": 0,
        "hallucinated_icd": 0,
        "meat_coverage": 0.0,
        "duplicates_found": 0,
        "symptom_overcoding": 0,
        "confidence_avg": 0.0,
        "processing_time": 0.0,
        "rules_applied": 0,
        "rules_actions": [],
    }

    start = time.time()

    try:
        # 1. Create document record
        doc = Document(
            filename=filename,
            file_path=pdf_path,
            status=DocumentStatus.UPLOADED,
        )
        db_session.add(doc)
        db_session.commit()
        db_session.refresh(doc)

        # 2. Extract text
        extract_result = await pipeline_orchestrator.extract_text(doc, db_session)
        db_session.refresh(doc)

        if not doc.raw_text or len(doc.raw_text.strip()) < 50:
            result["error"] = "Insufficient text extracted"
            result["processing_time"] = round(time.time() - start, 2)
            return result

        # 3. Full pipeline
        pipeline_result = await pipeline_orchestrator.process_all(doc, db_session)

        unified = pipeline_result.get("unified_results", [])
        all_detected = pipeline_result.get("total_detected", 0)
        rules_data = pipeline_result.get("medical_coding_rules", {})

        result["diseases_detected"] = all_detected
        result["diseases_after_filter"] = len(unified)
        result["icd_mapped"] = sum(
            1 for r in unified
            if r.get("icd_code") and r["icd_code"] not in ("—", "-", "")
        )

        # Hallucination check: ICD codes that are placeholder/invalid
        from backend.app.services.deterministic_validator import deterministic_validator
        hallucinated = 0
        for r in unified:
            icd = r.get("icd_code", "")
            if icd and icd not in ("—", "-", ""):
                v = deterministic_validator.validate_icd_code(icd)
                if not v["valid"]:
                    hallucinated += 1
        result["hallucinated_icd"] = hallucinated

        # MEAT coverage (evidence-based tiers)
        strong = sum(1 for r in unified if r.get("meat_tier") == "strong_evidence")
        moderate = sum(1 for r in unified if r.get("meat_tier") == "moderate_evidence")
        weak = sum(1 for r in unified if r.get("meat_tier") == "weak_evidence")
        no_meat = sum(1 for r in unified if r.get("meat_tier") == "no_meat")
        total = len(unified) or 1
        result["meat_coverage"] = round((strong + moderate) / total * 100, 1)
        result["meat_strong"] = strong
        result["meat_moderate"] = moderate
        result["meat_weak"] = weak
        result["meat_no_meat"] = no_meat

        # Evidence-based validation metrics
        evidence_based_count = sum(1 for r in unified if r.get("evidence_based", False))
        result["evidence_based_count"] = evidence_based_count
        result["evidence_based_rate"] = round(evidence_based_count / total * 100, 1)

        # MEAT validation pass rate (score >= 2)
        meat_valid = sum(1 for r in unified if r.get("meat_score", 0) >= 2)
        result["meat_validation_pass_rate"] = round(meat_valid / total * 100, 1)

        # Per-disease detail for reporting
        disease_details = []
        for r in unified:
            disease_details.append({
                "disease": r.get("disease", ""),
                "icd_code": r.get("icd_code", ""),
                "icd_description": r.get("icd_description", ""),
                "meat_tier": r.get("meat_tier", ""),
                "meat_score": r.get("meat_score", 0),
                "meat_status": r.get("meat_status", ""),
                "evidence_based": r.get("evidence_based", False),
                "confidence": r.get("confidence", 0),
                "segment": r.get("segment", ""),
                "monitoring": bool(r.get("monitoring")),
                "evaluation": bool(r.get("evaluation")),
                "assessment": bool(r.get("assessment")),
                "treatment": bool(r.get("treatment")),
            })
        result["disease_details"] = disease_details

        # Duplicates (same ICD code appearing multiple times)
        icd_codes = [
            r.get("icd_code") for r in unified
            if r.get("icd_code") and r["icd_code"] not in ("—", "-", "")
        ]
        result["duplicates_found"] = len(icd_codes) - len(set(icd_codes))

        # Symptom overcoding (R-codes when diagnoses present)
        r_codes = [r for r in unified if (r.get("icd_code") or "").upper().startswith("R")]
        non_r = [r for r in unified if r.get("icd_code") and not (r.get("icd_code") or "").upper().startswith("R") and r.get("icd_code") not in ("—", "-")]
        result["symptom_overcoding"] = len(r_codes) if non_r and r_codes else 0

        # Confidence average
        confs = [r.get("confidence", 0) for r in unified if r.get("confidence")]
        result["confidence_avg"] = round(sum(confs) / max(len(confs), 1), 3)

        # Rules stats
        result["rules_applied"] = rules_data.get("stats", {}).get("rules_applied", 0)
        result["rules_actions"] = rules_data.get("audit_log", [])

        result["success"] = True

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        logger.error(f"  ❌ {filename}: {result['error']}")
        logger.debug(traceback.format_exc())

    result["processing_time"] = round(time.time() - start, 2)
    return result


async def run_all_pdfs(pdf_dir: str) -> List[Dict[str, Any]]:
    """Process every PDF in a directory sequentially."""
    from backend.app.core.database import SessionLocal

    pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDFs found in {pdf_dir}")
        return []

    logger.info(f"Found {len(pdf_files)} PDFs in {pdf_dir}")
    results = []

    for pdf in pdf_files:
        db = SessionLocal()
        try:
            logger.info(f"  Processing: {pdf.name}")
            r = await process_single_pdf(str(pdf), db)
            results.append(r)
            status = "✅" if r["success"] else "❌"
            logger.info(
                f"  {status} {pdf.name} — {r['diseases_after_filter']} diseases, "
                f"MEAT {r['meat_coverage']}%, "
                f"halluc {r['hallucinated_icd']}, "
                f"time {r['processing_time']}s"
            )
        except Exception as e:
            logger.error(f"  Fatal error on {pdf.name}: {e}")
            results.append({"filename": pdf.name, "success": False, "error": str(e)})
        finally:
            try:
                db.close()
            except Exception:
                pass

    return results


# ════════════════════════════════════════════════════════════════════
# STEP 9 — Generate Report
# ════════════════════════════════════════════════════════════════════

def generate_report(
    db_info: Dict,
    migration_info: Dict,
    pdf_results: List[Dict],
    bugs_fixed: List[str],
    output_path: str,
) -> str:
    """Write a Markdown accuracy report with evidence-based MEAT metrics."""
    total = len(pdf_results)
    successes = sum(1 for r in pdf_results if r.get("success"))
    failures = total - successes

    total_diseases = sum(r.get("diseases_after_filter", 0) for r in pdf_results)
    total_icd = sum(r.get("icd_mapped", 0) for r in pdf_results)
    total_halluc = sum(r.get("hallucinated_icd", 0) for r in pdf_results)
    total_dupes = sum(r.get("duplicates_found", 0) for r in pdf_results)
    total_symptom_oc = sum(r.get("symptom_overcoding", 0) for r in pdf_results)

    halluc_rate = round(total_halluc / max(total_icd, 1) * 100, 1)
    icd_precision = round((total_icd - total_halluc) / max(total_icd, 1) * 100, 1)

    # Evidence-based MEAT metrics
    total_strong = sum(r.get("meat_strong", 0) for r in pdf_results if r.get("success"))
    total_moderate = sum(r.get("meat_moderate", 0) for r in pdf_results if r.get("success"))
    total_weak = sum(r.get("meat_weak", 0) for r in pdf_results if r.get("success"))
    total_no_meat = sum(r.get("meat_no_meat", 0) for r in pdf_results if r.get("success"))
    total_evidence_based = sum(r.get("evidence_based_count", 0) for r in pdf_results if r.get("success"))

    meat_coverages = [r.get("meat_coverage", 0) for r in pdf_results if r.get("success")]
    avg_meat = round(sum(meat_coverages) / max(len(meat_coverages), 1), 1)
    evidence_rates = [r.get("evidence_based_rate", 0) for r in pdf_results if r.get("success")]
    avg_evidence_rate = round(sum(evidence_rates) / max(len(evidence_rates), 1), 1)
    validation_rates = [r.get("meat_validation_pass_rate", 0) for r in pdf_results if r.get("success")]
    avg_validation_rate = round(sum(validation_rates) / max(len(validation_rates), 1), 1)

    conf_avgs = [r.get("confidence_avg", 0) for r in pdf_results if r.get("success")]
    avg_conf = round(sum(conf_avgs) / max(len(conf_avgs), 1), 3)
    total_time = sum(r.get("processing_time", 0) for r in pdf_results)

    # ICD specificity: % of valid (non-hallucinated) ICD codes
    icd_specificity = round((total_icd - total_halluc) / max(total_diseases, 1) * 100, 1)
    # Mismatch rate: diseases detected but not ICD-mapped
    icd_mismatch = total_diseases - total_icd
    mismatch_rate = round(icd_mismatch / max(total_diseases, 1) * 100, 1)

    # Final criteria
    criteria = {
        "db_connected": db_info.get("connected", False),
        "tables_created": len(db_info.get("tables_created", [])) >= 5,
        "all_pdfs_processed": failures == 0,
        "no_crashes": failures == 0,
        "hallucination_lt_10pct": halluc_rate < 10,
        "meat_coverage_gt_50pct": avg_meat > 50,
        "duplicates_removed": total_dupes == 0,
        "evidence_based_gt_60pct": avg_evidence_rate > 60,
    }
    all_pass = all(criteria.values())

    report = f"""# End-to-End Accuracy Report (Evidence-Based MEAT)

**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Pipeline Mode**: Evidence-Only (No Fallback Enrichment)  
**Pipeline Status**: {'✅ ALL CRITERIA MET' if all_pass else '⚠️ SOME CRITERIA NOT MET'}

---

## 1. Database Connection

| Metric | Value |
|--------|-------|
| DB Type | {db_info.get('db_type', 'unknown')} |
| Host | {db_info.get('host', 'unknown')} |
| Connected | {'✅' if db_info.get('connected') else '❌'} |
| Tables Created | {', '.join(db_info.get('tables_created', []))} |
| Total Tables | {len(db_info.get('tables_created', []))} |

## 2. PDF Processing Summary

| Metric | Value |
|--------|-------|
| Total PDFs | {total} |
| Successful | {successes} |
| Failed | {failures} |
| Total Diseases Found | {total_diseases} |
| Total ICD Mapped | {total_icd} |
| Total Processing Time | {round(total_time, 1)}s |
| Avg Time per PDF | {round(total_time / max(total, 1), 1)}s |

## 3. ICD Accuracy Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| ICD Precision | {icd_precision}% | > 90% | {'✅' if icd_precision > 90 else '⚠️'} |
| Hallucination Rate | {halluc_rate}% | < 10% | {'✅' if halluc_rate < 10 else '❌'} |
| ICD Specificity | {icd_specificity}% | > 80% | {'✅' if icd_specificity > 80 else '⚠️'} |
| ICD Mismatch Rate | {mismatch_rate}% | < 20% | {'✅' if mismatch_rate < 20 else '⚠️'} |
| Duplicates | {total_dupes} | 0 | {'✅' if total_dupes == 0 else '❌'} |
| Symptom Overcoding | {total_symptom_oc} | 0 | {'✅' if total_symptom_oc == 0 else '⚠️'} |
| Confidence Average | {avg_conf} | > 0.5 | {'✅' if avg_conf > 0.5 else '⚠️'} |

## 4. Evidence-Based MEAT Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| MEAT Coverage (Strong+Moderate) | {avg_meat}% | > 50% | {'✅' if avg_meat > 50 else '⚠️'} |
| Evidence-Based Rate | {avg_evidence_rate}% | > 60% | {'✅' if avg_evidence_rate > 60 else '⚠️'} |
| MEAT Validation Pass Rate | {avg_validation_rate}% | > 50% | {'✅' if avg_validation_rate > 50 else '⚠️'} |

### MEAT Tier Distribution (All PDFs)

| Tier | Count | % |
|------|-------|---|
| Strong Evidence (3-4 components) | {total_strong} | {round(total_strong / max(total_diseases, 1) * 100, 1)}% |
| Moderate Evidence (2 components) | {total_moderate} | {round(total_moderate / max(total_diseases, 1) * 100, 1)}% |
| Weak Evidence (1 component) | {total_weak} | {round(total_weak / max(total_diseases, 1) * 100, 1)}% |
| No MEAT (0 components) | {total_no_meat} | {round(total_no_meat / max(total_diseases, 1) * 100, 1)}% |
| **Evidence-Based Total** | **{total_evidence_based}** | **{round(total_evidence_based / max(total_diseases, 1) * 100, 1)}%** |

## 5. Per-PDF Summary

| PDF | Diseases | ICD | MEAT% | Strong | Mod | Weak | None | Halluc | Conf | Time |
|-----|----------|-----|-------|--------|-----|------|------|--------|------|------|
"""
    for r in pdf_results:
        status = "✅" if r.get("success") else "❌"
        report += (
            f"| {status} {r.get('filename', '?')} "
            f"| {r.get('diseases_after_filter', 0)} "
            f"| {r.get('icd_mapped', 0)} "
            f"| {r.get('meat_coverage', 0)}% "
            f"| {r.get('meat_strong', 0)} "
            f"| {r.get('meat_moderate', 0)} "
            f"| {r.get('meat_weak', 0)} "
            f"| {r.get('meat_no_meat', 0)} "
            f"| {r.get('hallucinated_icd', 0)} "
            f"| {r.get('confidence_avg', 0)} "
            f"| {r.get('processing_time', 0)}s |\n"
        )

    # Per-PDF disease detail tables
    report += "\n## 6. Per-PDF Disease Details\n\n"
    for r in pdf_results:
        if not r.get("success") or not r.get("disease_details"):
            continue
        report += f"### {r['filename']}\n\n"
        report += "| # | Disease | ICD Code | MEAT Tier | Score | M | E | A | T | Conf | Section |\n"
        report += "|---|---------|----------|-----------|-------|---|---|---|---|------|---------|\n"
        for i, dd in enumerate(r["disease_details"], 1):
            m = "✓" if dd["monitoring"] else "✗"
            e = "✓" if dd["evaluation"] else "✗"
            a = "✓" if dd["assessment"] else "✗"
            t = "✓" if dd["treatment"] else "✗"
            report += (
                f"| {i} | {dd['disease']} | {dd['icd_code']} | {dd['meat_tier']} "
                f"| {dd['meat_score']} | {m} | {e} | {a} | {t} "
                f"| {dd['confidence']} | {dd['segment']} |\n"
            )
        report += "\n"

    report += f"""## 7. Final Criteria

| Criterion | Result |
|-----------|--------|
| DB Connected | {'✅' if criteria['db_connected'] else '❌'} |
| Tables Created (≥5) | {'✅' if criteria['tables_created'] else '❌'} |
| All PDFs Processed | {'✅' if criteria['all_pdfs_processed'] else '❌'} |
| No Crashes | {'✅' if criteria['no_crashes'] else '❌'} |
| Hallucination < 10% | {'✅' if criteria['hallucination_lt_10pct'] else '❌'} |
| MEAT Coverage > 50% | {'✅' if criteria['meat_coverage_gt_50pct'] else '⚠️'} |
| Duplicates Removed | {'✅' if criteria['duplicates_removed'] else '❌'} |
| Evidence-Based > 60% | {'✅' if criteria['evidence_based_gt_60pct'] else '⚠️'} |

**Overall**: {'✅ PASS' if all_pass else '⚠️ PARTIAL'}
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"✅ Report written to {output_path}")
    return report


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

async def main():
    bugs_fixed: List[str] = []

    # ── STEP 1+2: Connect DB ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1-2: Connecting database...")
    db_info = connect_database()

    # ── STEP 4: Migrate SQLite ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Checking SQLite migration...")
    migration_info = migrate_sqlite_if_exists()

    # ── STEP 5+6: Run PDFs ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5-6: Running E2E pipeline on test PDFs...")

    test_pdf_dir = PROJECT_ROOT / "test pdf"
    if not test_pdf_dir.exists():
        # Fallback to Data1/
        test_pdf_dir = PROJECT_ROOT.parent / "Data1"
    if not test_pdf_dir.exists():
        logger.error("No test PDF directory found!")
        pdf_results = []
    else:
        pdf_results = await run_all_pdfs(str(test_pdf_dir))

    # ── STEP 7: Accuracy analysis ───────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 7: Accuracy analysis...")

    total_icd = sum(r.get("icd_mapped", 0) for r in pdf_results)
    total_halluc = sum(r.get("hallucinated_icd", 0) for r in pdf_results)
    halluc_rate = round(total_halluc / max(total_icd, 1) * 100, 1)
    meat_coverages = [r.get("meat_coverage", 0) for r in pdf_results if r.get("success")]
    avg_meat = round(sum(meat_coverages) / max(len(meat_coverages), 1), 1)

    logger.info(f"  Hallucination rate: {halluc_rate}%")
    logger.info(f"  MEAT coverage avg: {avg_meat}%")
    logger.info(f"  Total ICD mapped: {total_icd}")

    # ── STEP 9: Generate report ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 9: Generating report...")

    report_path = str(PROJECT_ROOT / "reports" / "e2e_accuracy_report.md")
    generate_report(db_info, migration_info, pdf_results, bugs_fixed, report_path)

    # ── Summary ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  PDFs: {len(pdf_results)} processed")
    logger.info(f"  Successes: {sum(1 for r in pdf_results if r.get('success'))}")
    logger.info(f"  Hallucination: {halluc_rate}%")
    logger.info(f"  MEAT coverage: {avg_meat}%")
    logger.info(f"  Report: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
