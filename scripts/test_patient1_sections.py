"""Test section detection on actual Patient 1 PDF text from DB."""
from backend.app.core.database import SessionLocal
from backend.app.models.document import Document
from backend.app.utils.section_candidate_extractor import SectionCandidateExtractor
from backend.app.utils.fuzzy_section_matcher import match_canonical
from backend.app.utils.content_section_inferrer import ContentSectionInferrer
from backend.app.services.clinical_document_analyzer import ClinicalDocumentAnalyzer

import logging
logging.basicConfig(level=logging.WARNING)

db = SessionLocal()
doc = db.query(Document).filter(
    Document.id == "63ceb29a-99dd-4b8a-ad0c-47914aae53f4"
).first()
db.close()

if not doc or not doc.raw_text:
    print("Document not found or no text!")
    exit(1)

text = doc.raw_text
print(f"Document text: {len(text)} chars")
print()

# Layer 1: Regex
ext = SectionCandidateExtractor()
cands = ext.extract_candidates(text)
high, low = ext.filter_by_confidence(cands, threshold=0.65)
print(f"=== REGEX DETECTION: {len(high)} high-confidence headers ===")
regex_matched = 0
for c in high:
    canon, score = match_canonical(c["header"])
    tag = f"-> {canon} ({score:.2f})" if canon else "-> NO MATCH"
    print(f"  [{c['confidence']:.2f}] {c['header'][:50]:50s} {tag}")
    if canon:
        regex_matched += 1
print(f"  Matched to canonical: {regex_matched}/{len(high)}")

# Layer 2: Content inference
print()
inf = ContentSectionInferrer()
inferred = inf.infer_sections(text)
print(f"=== CONTENT INFERENCE: {len(inferred)} sections ===")
for s in inferred:
    print(f"  {s['canonical']:25s} lines={s['line_count']:2d}  conf={s['confidence']:.3f}")

# Combined
print()
analyzer = ClinicalDocumentAnalyzer()
marked = analyzer._inject_section_markers(text)
marker_lines = [l.strip() for l in marked.split("\n") if "═══" in l]
print(f"=== COMBINED: {len(marker_lines)} markers injected ===")
for ml in marker_lines:
    print(f"  {ml}")
