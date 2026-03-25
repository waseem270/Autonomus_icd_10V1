"""Quick test of section detection on headerless clinical text."""
from backend.app.utils.section_candidate_extractor import SectionCandidateExtractor
from backend.app.utils.fuzzy_section_matcher import match_canonical
from backend.app.utils.content_section_inferrer import ContentSectionInferrer

sample = """
acetaminophen (TYLENOL) 500 MG tablet
lisinopril 10 mg daily
metformin 500 mg twice daily
atorvastatin 20 mg at bedtime

BP 130/85 HR 72 Temp 98.6 SpO2 98%

1. Essential Hypertension
2. Type 2 Diabetes Mellitus
3. Mixed Hyperlipidemia

HEENT: normal
Lungs: clear
Heart: RRR no murmurs
Abdomen: soft, non-tender

Denies chest pain, denies shortness of breath
Denies headache, denies dizziness
"""

# Test regex detection
ext = SectionCandidateExtractor()
cands = ext.extract_candidates(sample)
high, low = ext.filter_by_confidence(cands, threshold=0.65)
print("=== REGEX DETECTION ===")
print(f"Total candidates: {len(cands)}, High: {len(high)}, Low: {len(low)}")
for c in high:
    canon, score = match_canonical(c["header"])
    print(f"  {c['header']:30s} -> {canon} (score={score:.2f}, conf={c['confidence']:.2f})")

# Test content inference
inf = ContentSectionInferrer()
inferred = inf.infer_sections(sample)
print()
print("=== CONTENT INFERENCE ===")
for s in inferred:
    print(f"  {s['canonical']:25s} lines={s['line_count']} conf={s['confidence']} pos={s['position']}")

# Test full injection
print()
print("=== COMBINED INJECTION ===")
from backend.app.services.clinical_document_analyzer import ClinicalDocumentAnalyzer
analyzer = ClinicalDocumentAnalyzer()
marked = analyzer._inject_section_markers(sample)
# Print just the marker lines
for line in marked.split("\n"):
    if "═══" in line:
        print(f"  {line.strip()}")
