from typing import Dict, List, Any, Set
import logging
import asyncio

from ..utils.text_preprocessor import (
    segment_sentences,
    remove_noise,
    normalize_whitespace,
    fix_line_breaks
)
from ..utils.abbreviation_expander import expand_abbreviations
from .smart_section_detector import smart_section_detector
from .regex_disease_extractor import extract_disease_candidates
from .llm_disease_extractor import llm_disease_extractor


class TextStructuringService:
    """
    Orchestrate the full clinical text structuring and analysis pipeline.
    
    Pipeline order:
        1. Clean & normalize
        2. Section detection (Gemini)
        3. Regex candidate extraction + LLM primary disease detection
        4. Medication-implied disease inference
        5. Post-LLM Assessment deterministic patching
        6. Cross-section deduplication
        7. Abbreviation expansion on FULL TEXT (for downstream MEAT / ICD)
        8. Global sentence segmentation (on expanded text)
        9. Remap disease sentence numbers to the global sentence list
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _normalize_for_dedup(self, disease_name: str) -> str:
        """
        Aggressive normalization for cross-section deduplication.
        
        Removes:
        - List item prefixes (1., 2., a), b), •, -, *)
        - Leading/trailing whitespace and punctuation
        - Extra internal whitespace
        
        Returns:
            Normalized lowercase string for comparison
        
        Examples:
            "1. Type 2 Diabetes" → "type 2 diabetes"
            "• Anxiety disorder" → "anxiety disorder"
            "3. Hypertension, uncontrolled  " → "hypertension, uncontrolled"
        """
        import re
        
        # Remove list item prefixes: "1.", "2)", "a.", "b)", "•", "-", "*"
        cleaned = re.sub(r'^\s*(?:\d+[.\)\s]|[a-zA-Z][.\)]|[•\-\*])\s*', '', disease_name)
        
        # Remove trailing punctuation and whitespace
        cleaned = cleaned.strip('.,;:\t\n ')
        
        # Normalize internal whitespace (multiple spaces → single space)
        cleaned = ' '.join(cleaned.split())
        
        # Return lowercase for case-insensitive comparison
        cleaned_lower = cleaned.lower()
        
        # Normalize slashes to spaces for dedup: "major depression/moderate/recurrent"
        # becomes "major depression moderate recurrent" so it matches the expanded form
        cleaned_lower = cleaned_lower.replace('/', ' ')
        cleaned_lower = ' '.join(cleaned_lower.split())
        
        return cleaned_lower

    def _clean_disease_name(self, disease_name: str) -> str:
        """
        Clean disease name for storage (preserve case, remove artifacts).
        
        Similar to _normalize_for_dedup but preserves original case.
        """
        import re
        
        # Remove list prefixes
        cleaned = re.sub(r'^\s*(?:\d+[.\)\s]|[a-zA-Z][.\)]|[•\-\*])\s*', '', disease_name)
        
        # Remove trailing punctuation
        cleaned = cleaned.strip('.,;:\t\n ')
        
        # Normalize whitespace
        cleaned = ' '.join(cleaned.split())
        
        return cleaned
    
    async def structure_text(self, raw_text: str) -> Dict[str, Any]:
        """
        Complete text structuring and clinical analysis pipeline.
        """
        if not raw_text:
            return {
                "cleaned_text": "",
                "sentences": [],
                "sections": {},
                "detected_diseases": [],
                "total_sentences": 0,
                "total_diseases": 0
            }

        self.logger.info("Starting clinical text structuring pipeline...")
        stats = {}
        
        # ── Phase 1: Cleaning & Normalization ──
        self.logger.info("Phase 1: Normalizing text and removing document noise...")
        
        text_no_noise = remove_noise(raw_text)
        # Preserve line breaks for section detection
        text_for_sections = normalize_whitespace(text_no_noise)
        stats["normalization"] = {"status": "completed"}
        
        # ── Phase 2: Structural Analysis ──
        self.logger.info("Phase 2: Detecting clinical note structure (Gemini)...")
        section_result = await smart_section_detector.detect_sections(text_for_sections)
        sections = section_result.get("sections", {})
        llm_validated_diseases = section_result.get("validated_diseases", [])
        
        stats["structural_analysis"] = {
            "sections_found": len(sections),
            "has_structure": section_result.get("has_structure", False),
            "confidence": section_result.get("confidence", 0.0),
            "llm_validated_count": len(llm_validated_diseases)
        }
        
        # ── Phase 3: Disease Detection (on UNEXPANDED text) ──
        # Primary detector: LLM (Gemini) — reads prefixes/suffixes properly.
        # Backup: Regex-based candidate extraction from structured patterns.
        self.logger.info("Phase 3: Running disease detection (regex + LLM)...")
        
        all_diseases = []
        
        # Integrate LLM Validated Diseases from Structural Phase
        for vd in llm_validated_diseases:
            if vd.get("is_valid_disease") and vd.get("icd_billable"):
                raw_name = vd.get("disease_name", "").strip()
                if not raw_name:
                    continue
                all_diseases.append({
                    "disease_name": raw_name,
                    "normalized_name": raw_name.lower(),
                    "confidence_score": vd.get("confidence", 0.9),
                    "negated": False,
                    "section": vd.get("found_in_section", "medical_history"),
                    "section_sources": [vd.get("found_in_section", "medical_history")],
                    "is_llm_validated": True,
                    "entity_type": "CONDITION",
                    "sentence_number": None,
                    "start_char": 0,
                    "end_char": 0,
                    "clinical_context": vd.get("clinical_context", ""),
                    "reasoning": vd.get("reasoning", "")
                })

        if sections:
            self.logger.info(f"Processing diseases across {len(sections)} clinical sections...")

            # 3a. PMH structured-entry parsing (deterministic)
            for section_name, section_data in sections.items():
                section_content = section_data.get("text", "")
                if not section_content:
                    continue
                if section_name.lower() == "past_medical_history":
                    pmh_diseases = self._parse_pmh_entries(section_content)
                    all_diseases.extend(pmh_diseases)

            # 3b. Regex-based candidate extraction per section (backup / seed)
            for section_name, section_data in sections.items():
                section_content = section_data.get("text", "")
                if not section_content:
                    continue
                regex_candidates = extract_disease_candidates(
                    section_content, section=section_name
                )
                all_diseases.extend(regex_candidates)
        else:
            # Fallback: Regex extraction on the whole document
            self.logger.info("No clear sections detected, performing document-wide regex extraction...")
            text_for_extraction = fix_line_breaks(text_for_sections)
            text_for_extraction = normalize_whitespace(text_for_extraction)
            all_diseases = extract_disease_candidates(text_for_extraction, section="unknown")
        
        # Initialize section_sources for all NER-detected diseases
        for d in all_diseases:
            if "section_sources" not in d:
                d["section_sources"] = [d.get("section", "unknown")]

        # ── Medication-implied disease inference ──
        # If a medication is listed but no corresponding disease was detected,
        # infer the disease from the medication name.
        if sections:
            med_section = None
            for key in ("medications", "current_medications"):
                if key in sections and sections[key].get("text", "").strip():
                    med_section = sections[key]["text"]
                    break
            if med_section:
                med_diseases = self._infer_diseases_from_medications(
                    med_section, all_diseases
                )
                all_diseases.extend(med_diseases)

        # LLM-based extraction — PRIMARY disease detector (catches full specificity)
        # After retries by call_gemini, if it STILL fails the error propagates
        # up so the caller (route handler) can return a proper error response
        # instead of silently returning noisy regex-only results.
        if sections:
            llm_diseases = await llm_disease_extractor.extract_from_sections(sections)
            # Merge: add LLM diseases not already found by regex
            existing_names = {d["normalized_name"].lower().strip() for d in all_diseases}
            for ld in llm_diseases:
                norm = ld["normalized_name"].lower().strip()
                if norm not in existing_names:
                    all_diseases.append(ld)
                    existing_names.add(norm)
                else:
                    # Merge section_sources into existing disease
                    # Also upgrade canonical 'section' if LLM found Assessment source
                    _ASSESSMENT_MERGE_KEYS = {
                        "assessment_and_plan", "assessment",
                        "active_problems", "active_problem_list", "problem_list"
                    }
                    for d in all_diseases:
                        if d["normalized_name"].lower().strip() == norm:
                            existing_sources = d.get("section_sources", [d.get("section", "unknown")])
                            new_sources = ld.get("section_sources", [])
                            merged_sources = list(dict.fromkeys(existing_sources + new_sources))
                            d["section_sources"] = merged_sources
                            # If LLM determined this is an Assessment disease, upgrade section
                            merged_lower = {s.lower() for s in merged_sources}
                            if merged_lower & _ASSESSMENT_MERGE_KEYS:
                                # Use the most specific assessment key found
                                for key in ("assessment_and_plan", "assessment",
                                            "active_problems", "active_problem_list", "problem_list"):
                                    if key in merged_lower:
                                        d["section"] = key
                                        break
                            break

        # ── Post-LLM Assessment patch ──────────────────────────────────
        # Ensure any disease whose name appears verbatim in the Assessment section
        # text (from numbered plan items) gets "assessment" added to its sources
        # and its canonical section upgraded. This catches diseases that NER found
        # in HPI/PMH that are ALSO listed in the Assessment Plan.
        if sections:
            try:
                all_diseases = llm_disease_extractor._patch_assessment_sources(all_diseases, sections)
            except Exception as e:
                self.logger.warning(f"Post-LLM assessment patch failed (non-fatal): {e}")

        # ── Phase 4: Cross-section deduplication ──
        all_diseases = self._cross_section_dedup(all_diseases)
        
        stats["ner_results"] = {
            "total_detected": len(all_diseases),
            "negated_count": sum(1 for d in all_diseases if d.get("negated", False))
        }
        
        # ── Phase 5: Abbreviation expansion on FULL text ──
        # This produces the cleaned text used by MEAT validation and ICD mapper
        self.logger.info("Phase 4: Expanding clinical abbreviations (post-NER)...")
        text = fix_line_breaks(text_for_sections)
        text = normalize_whitespace(text)
        text = expand_abbreviations(text, context_aware=True)
        stats["abbreviation_expansion"] = {"status": "completed"}
        
        # ── Phase 6: Global sentence segmentation (on expanded text) ──
        self.logger.info("Phase 5: Segmenting global sentences...")
        global_sentences = segment_sentences(text)
        stats["segmentation"] = {"total_sentences": len(global_sentences)}
        
        # ── Phase 7: Remap disease → global sentence numbers ──
        # Disease sentence_number values are local to their section.
        # We remap them to the nearest matching global sentence for context_builder.
        self._remap_to_global_sentences(all_diseases, global_sentences)
        
        # Sort diseases by their global sentence number (None sorts as 0)
        all_diseases.sort(key=lambda x: x.get("sentence_number") or 0)
        
        self.logger.info(
            f"✅ Pipeline complete: {len(global_sentences)} sentences, "
            f"{len(all_diseases)} disease entities identified."
        )
        
        return {
            "cleaned_text": text,
            "sentences": global_sentences,
            "sections": sections,
            "detected_diseases": all_diseases,
            "abbreviations_expanded": True,
            "total_sentences": len(global_sentences),
            "total_diseases": len(all_diseases),
            "processing_stats": stats
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # Non-disease PMH entries that look like conditions but aren't
    _PMH_SKIP = {
        "follow-up consultation", "follow up consultation",
        "followup consultation", "orthopaedic complexity",
        "orthopedic complexity", "body mass index",
        "bmi", "history type", "none recorded", "none recorded.",
        "none", "n/a", "not applicable",
        # Admin / non-disease entries
        "sleep rest pattern finding", "screening",
        "general health check", "routine examination",
    }

    def _parse_pmh_entries(self, text: str) -> List[Dict]:
        """
        Extract disease entries from structured PMH text.
        Pattern: '[Disease Name] Active MM/DD/YYYY [metadata]'
        Returns disease dicts compatible with disease_detector output.
        """
        import re
        # Split on "Active DD/DD/DDDD" to extract preceding disease names
        parts = re.split(r'\s+Active\s+\d{1,2}/\d{1,2}/\d{2,4}', text)
        diseases = []
        seen = set()

        for part in parts:
            # The disease name is the tail end of each part (after prior metadata)
            # Clean up metadata suffixes from previous entry
            name = re.sub(
                r'(?i)History\s+Type:.*$|Orthopaedic\s+Complexity:.*$|'
                r'Acute,?\s*$|Uncomplicated\s*$|History\s*$',
                '', part
            ).strip()
            # Take only the last meaningful line (prior metadata can precede)
            lines = [ln.strip() for ln in name.split('\n') if ln.strip()]
            if lines:
                name = lines[-1]
            # Clean residual punctuation
            name = re.sub(r'^[,:\s]+|[,:\s]+$', '', name).strip()
            if not name or len(name) < 4:
                continue
            # Filter non-disease entries
            norm = name.lower()
            if norm in self._PMH_SKIP:
                continue
            if any(skip in norm for skip in ["body mass index", "follow-up", "followup",
                                             "none recorded", "pattern finding",
                                             "screening"]):
                continue
            # Filter procedure-like entries
            if re.search(r'ectomy|plasty|otomy|ostomy|post[\s-]?op', norm, re.I):
                continue
            # Dedup
            key = " ".join(sorted(norm.split()))
            if key in seen:
                continue
            seen.add(key)

            diseases.append({
                "disease_name": name,
                "normalized_name": norm,
                "confidence_score": 0.85,  # High confidence for structured PMH
                "negated": False,
                "start_char": 0,
                "end_char": 0,
                "section": "past_medical_history",
                "sentence_number": None,
                "entity_type": "CONDITION",
            })

        self.logger.info(f"PMH parser extracted {len(diseases)} disease entries")
        return diseases

    # Medication → Disease mapping (content-based, not hardcoded to specific patients)
    _MED_TO_DISEASE = {
        # Diabetes
        "metformin": "Type 2 Diabetes Mellitus",
        "glipizide": "Type 2 Diabetes Mellitus",
        "glimepiride": "Type 2 Diabetes Mellitus",
        "sitagliptin": "Type 2 Diabetes Mellitus",
        "empagliflozin": "Type 2 Diabetes Mellitus",
        "dapagliflozin": "Type 2 Diabetes Mellitus",
        "trulicity": "Type 2 Diabetes Mellitus",
        "dulaglutide": "Type 2 Diabetes Mellitus",
        "semaglutide": "Type 2 Diabetes Mellitus",
        "ozempic": "Type 2 Diabetes Mellitus",
        "jardiance": "Type 2 Diabetes Mellitus",
        "farxiga": "Type 2 Diabetes Mellitus",
        "januvia": "Type 2 Diabetes Mellitus",
        "insulin": "Type 2 Diabetes Mellitus",
        # Hypertension
        "lisinopril": "Essential Hypertension",
        "losartan": "Essential Hypertension",
        "valsartan": "Essential Hypertension",
        "olmesartan": "Essential Hypertension",
        "amlodipine": "Essential Hypertension",
        "nifedipine": "Essential Hypertension",
        "hydralazine": "Essential Hypertension",
        "hydrochlorothiazide": "Essential Hypertension",
        # Hyperlipidemia
        "atorvastatin": "Hyperlipidemia",
        "rosuvastatin": "Hyperlipidemia",
        "simvastatin": "Hyperlipidemia",
        "pravastatin": "Hyperlipidemia",
        "lovastatin": "Hyperlipidemia",
        # Thyroid
        "levothyroxine": "Hypothyroidism",
        "liothyronine": "Hypothyroidism",
        "synthroid": "Hypothyroidism",
        # GERD
        "omeprazole": "Gastroesophageal Reflux Disease",
        "pantoprazole": "Gastroesophageal Reflux Disease",
        "lansoprazole": "Gastroesophageal Reflux Disease",
        "esomeprazole": "Gastroesophageal Reflux Disease",
        "famotidine": "Gastroesophageal Reflux Disease",
        # Mental health
        "sertraline": "Major Depressive Disorder",
        "fluoxetine": "Major Depressive Disorder",
        "escitalopram": "Major Depressive Disorder",
        "citalopram": "Major Depressive Disorder",
        "paroxetine": "Major Depressive Disorder",
        # Pain / neuro
        "gabapentin": "Neuropathic Pain",
        "pregabalin": "Neuropathic Pain",
        "duloxetine": "Neuropathic Pain",
        # Gout
        "allopurinol": "Gout",
        "febuxostat": "Gout",
        "colchicine": "Gout",
        # BPH
        "finasteride": "Benign Prostatic Hyperplasia",
        "tamsulosin": "Benign Prostatic Hyperplasia",
        "dutasteride": "Benign Prostatic Hyperplasia",
        # Respiratory
        "albuterol": "Asthma",
        "fluticasone": "Asthma",
        "budesonide": "Asthma",
        "tiotropium": "Chronic Obstructive Pulmonary Disease",
        "montelukast": "Asthma",
        # Anticoagulation
        "warfarin": "Atrial Fibrillation",
        "apixaban": "Atrial Fibrillation",
        "rivaroxaban": "Atrial Fibrillation",
        # Heart failure
        "furosemide": "Heart Failure",
        "spironolactone": "Heart Failure",
        "carvedilol": "Heart Failure",
        # Osteoporosis
        "alendronate": "Osteoporosis",
        "risedronate": "Osteoporosis",
        "denosumab": "Osteoporosis",
        # Overactive bladder
        "myrbetriq": "Overactive Bladder Syndrome",
        "mirabegron": "Overactive Bladder Syndrome",
        "oxybutynin": "Overactive Bladder Syndrome",
        # Glaucoma
        "brimonidine": "Glaucoma",
        "timolol": "Glaucoma",
        "latanoprost": "Glaucoma",
        "lumigan": "Glaucoma",
        "bimatoprost": "Glaucoma",
        # Dementia
        "donepezil": "Dementia",
        "rivastigmine": "Dementia",
        "memantine": "Dementia",
        # Rheumatoid
        "methotrexate": "Rheumatoid Arthritis",
        "hydroxychloroquine": "Rheumatoid Arthritis",
        # Anemia
        "erythropoietin": "Anemia of Chronic Kidney Disease",
        "darbepoetin": "Anemia of Chronic Kidney Disease",
    }

    def _infer_diseases_from_medications(
        self, med_text: str, existing_diseases: List[Dict]
    ) -> List[Dict]:
        """
        Scan the medications section text for known drugs and infer diseases
        that are not already in the detected disease list.
        """
        import re
        med_text_lower = med_text.lower()
        existing_norms = {d["normalized_name"].lower().strip() for d in existing_diseases}
        # Also build a set of existing disease words for fuzzy dedup
        existing_words = set()
        for n in existing_norms:
            existing_words.update(n.split())

        inferred = []
        seen_diseases = set()

        for drug, disease in self._MED_TO_DISEASE.items():
            if drug.lower() not in med_text_lower:
                continue
            disease_norm = disease.lower().strip()
            if disease_norm in seen_diseases:
                continue

            # Check if disease already exists (exact or fuzzy)
            already_found = disease_norm in existing_norms
            if not already_found:
                # Fuzzy: check if key disease words overlap
                disease_words = set(disease_norm.split())
                # If >60% of disease name words appear in any existing disease
                for en in existing_norms:
                    en_words = set(en.split())
                    if disease_words and len(disease_words & en_words) / len(disease_words) >= 0.6:
                        already_found = True
                        break

            if already_found:
                continue

            seen_diseases.add(disease_norm)
            inferred.append({
                "disease_name": disease,
                "normalized_name": disease_norm,
                "confidence_score": 0.80,
                "negated": False,
                "section": "medications",
                "section_sources": ["medications"],
                "entity_type": "CONDITION",
                "sentence_number": None,
                "start_char": 0,
                "end_char": 0,
            })
            self.logger.info(f"Medication-implied disease: {drug} -> {disease}")

        self.logger.info(f"Medication inference added {len(inferred)} diseases")
        return inferred

    def _cross_section_dedup(self, diseases: List[Dict]) -> List[Dict]:
        """
        Remove duplicate diseases across sections using enhanced normalization.
        
        Handles:
        - List artifacts: "1. Diabetes" vs "Diabetes"
        - Bullet points: "• Hypertension" vs "Hypertension"
        - Whitespace variations
        - Case differences
        
        When duplicates found, keeps first occurrence and merges section_sources.
        """
        seen = {}
        deduplicated = []
        
        for disease in diseases:
            # Normalize for comparison
            norm_key = self._normalize_for_dedup(disease["disease_name"])
            
            if norm_key not in seen:
                # First occurrence - clean the name and store
                disease["disease_name"] = self._clean_disease_name(disease["disease_name"])
                disease["normalized_name"] = norm_key
                
                seen[norm_key] = disease
                deduplicated.append(disease)
                
            else:
                # Duplicate found - merge section sources into existing entry
                existing = seen[norm_key]
                
                # Get section sources from both
                existing_sources = existing.get("section_sources", [existing.get("section", "unknown")])
                new_sources = disease.get("section_sources", [disease.get("section", "unknown")])
                
                # Merge and deduplicate sources (preserve order)
                merged_sources = list(dict.fromkeys(existing_sources + new_sources))
                existing["section_sources"] = merged_sources
                
                # Use higher confidence score if available
                if disease.get("confidence_score", 0) > existing.get("confidence_score", 0):
                    existing["confidence_score"] = disease["confidence_score"]
                
                self.logger.debug(
                    f"Merged duplicate: '{disease['disease_name']}' "
                    f"(normalized: '{norm_key}') from sections {merged_sources}"
                )
        
        self.logger.info(
            f"Deduplication: {len(diseases)} → {len(deduplicated)} diseases "
            f"({len(diseases) - len(deduplicated)} duplicates removed)"
        )
        
        return deduplicated

    def _remap_to_global_sentences(
        self, diseases: List[Dict], global_sentences: List[Dict]
    ) -> None:
        """
        For each disease entity, find the global sentence that contains its text
        and update sentence_number to match the global list (used by context_builder).
        """
        if not global_sentences:
            return

        for disease in diseases:
            dname = disease.get("disease_name", "").lower()
            matched = False
            for gs in global_sentences:
                if dname in gs.get("text", "").lower():
                    disease["sentence_number"] = gs["sentence_number"]
                    matched = True
                    break
            if not matched:
                # Fallback: keep existing sentence_number (best-effort)
                pass


# Create singleton instance
text_structuring_service = TextStructuringService()
