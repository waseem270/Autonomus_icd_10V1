from typing import List, Dict, Optional
import re
import logging

logger = logging.getLogger(__name__)

# ── Disease-to-medication mapping for smarter context extraction ─────────────
DISEASE_MEDICATIONS = {
    "diabetes": ["metformin", "insulin", "glipizide", "sitagliptin", "jardiance", "ozempic"],
    "hypertension": ["lisinopril", "amlodipine", "losartan", "metoprolol", "atenolol", "hydrochlorothiazide"],
    "hyperlipidemia": ["atorvastatin", "rosuvastatin", "simvastatin", "pravastatin", "lipitor", "crestor"],
    "asthma": ["albuterol", "fluticasone", "montelukast", "salmeterol", "budesonide"],
    "depression": ["sertraline", "escitalopram", "fluoxetine", "bupropion", "venlafaxine"],
    "anxiety": ["buspirone", "sertraline", "lorazepam", "alprazolam", "citalopram"],
    "hypothyroidism": ["levothyroxine", "synthroid", "thyroid"],
    "gerd": ["omeprazole", "pantoprazole", "esomeprazole", "ranitidine", "famotidine"],
    "copd": ["tiotropium", "albuterol", "salmeterol", "ipratropium", "fluticasone"],
    "heart failure": ["furosemide", "carvedilol", "lisinopril", "digoxin", "spironolactone"],
}

# ── Disease-to-lab mapping for smarter context extraction ────────────────────
DISEASE_LABS = {
    "diabetes": ["hba1c", "glucose", "fasting glucose", "blood sugar", "a1c"],
    "hypertension": ["blood pressure", "bp", "systolic", "diastolic"],
    "hyperlipidemia": ["cholesterol", "ldl", "hdl", "triglycerides", "lipid panel"],
    "kidney disease": ["creatinine", "gfr", "egfr", "bun", "proteinuria"],
    "anemia": ["hemoglobin", "hematocrit", "rbc", "ferritin", "iron"],
    "thyroid": ["tsh", "t3", "t4", "thyroid function"],
    "liver disease": ["alt", "ast", "bilirubin", "albumin", "liver function"],
    "heart failure": ["bnp", "pro-bnp", "ejection fraction", "echocardiogram"],
}

# General lab keywords as fallback
GENERAL_LAB_KEYWORDS = [
    "lab", "test", "result", "level", "value", "panel",
    "hba1c", "glucose", "cholesterol", "creatinine", "hemoglobin",
    "blood pressure", "bp", "wbc", "rbc", "plt", "esr", "crp",
    "urine", "culture", "biopsy", "imaging", "x-ray", "mri", "ct scan"
]

# General medication keywords as fallback
GENERAL_MED_KEYWORDS = [
    "medication", "drug", "prescription", "prescribed", "rx",
    "taking", "continue", "start", "began", "initiated", "ordered",
    "mg", "mcg", "ml", "tablet", "capsule", "injection", "dose", "daily"
]


class ContextWindowBuilder:
    """
    Build rich context windows around detected diseases for MEAT validation.
    Context includes surrounding sentences, section text, medication references,
    and related lab values.
    """

    def __init__(self, context_window: int = 5):
        self.logger = logging.getLogger(__name__)
        self.context_window = context_window  # sentences before and after

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _get_disease_medications(self, disease_name: str) -> List[str]:
        """Return known medication keywords for a given disease."""
        dl = disease_name.lower()
        for key, meds in DISEASE_MEDICATIONS.items():
            if key in dl:
                return meds
        return []

    def _get_disease_labs(self, disease_name: str) -> List[str]:
        """Return known lab keywords for a given disease."""
        dl = disease_name.lower()
        for key, labs in DISEASE_LABS.items():
            if key in dl:
                return labs
        return []

    def _sentence_matches_keywords(self, sentence: str, keywords: List[str]) -> bool:
        """Check if a sentence contains any of the given keywords."""
        sl = sentence.lower()
        return any(kw in sl for kw in keywords)

    def _find_disease_sentences(
        self, disease_name: str, sentences: List[Dict]
    ) -> List[str]:
        """Find all sentences that mention the disease name (case-insensitive)."""
        dl = disease_name.lower()
        return [s["text"] for s in sentences if dl in s["text"].lower()]

    # ── Core Builder ─────────────────────────────────────────────────────────

    def build_disease_context(
        self,
        disease_name: str,
        full_text: str,
        sentences: List[Dict],
        sections: Dict[str, Dict],
        detected_disease: Dict,
    ) -> Dict:
        """
        Build a comprehensive context window for a single disease entity.

        Strategy:
          1. Find the primary sentence where the disease was first detected.
          2. Collect ±N surrounding sentences (context window).
          3. Attach the full section text where the disease appears.
          4. Search the rest of the document for other disease mentions.
          5. Pull medication sentences (disease-specific + general).
          6. Pull lab/test sentences (disease-specific + general).
        """
        context: Dict = {
            "disease": disease_name,
            "primary_mention": "",
            "context_sentences": [],
            "section_context": "",
            "related_sentences": [],
            "medication_mentions": [],
            "lab_mentions": [],
            "window_size": 0,
            "total_context_chars": 0,
            # Section provenance — used by MEAT validator to distinguish PMH vs active
            "section": detected_disease.get("section", "unknown"),
            "section_sources": detected_disease.get("section_sources", [detected_disease.get("section", "unknown")]),
        }

        sentence_num = detected_disease.get("sentence_number")
        sentences_sorted = sorted(sentences, key=lambda s: s["sentence_number"])

        # 1. Primary sentence
        primary_sent = None
        if sentence_num is not None:
            primary_sent = next(
                (s for s in sentences_sorted if s["sentence_number"] == sentence_num),
                None
            )
            if not primary_sent and sentences_sorted:
                # Fallback: first sentence that mentions the disease
                primary_sent = next(
                    (s for s in sentences_sorted
                     if disease_name.lower() in s["text"].lower()),
                    None
                )

        if primary_sent:
            context["primary_mention"] = primary_sent["text"]

            # 2. Context window (±N sentences)
            idx = sentences_sorted.index(primary_sent)
            start = max(0, idx - self.context_window)
            end = min(len(sentences_sorted), idx + self.context_window + 1)
            context["context_sentences"] = [s["text"] for s in sentences_sorted[start:end]]
            context["window_size"] = len(context["context_sentences"])

        # 3. Section context — gather text from ALL associated sections,
        #    but FILTER to only lines/sentences that are relevant to THIS specific disease.
        #    This prevents cross-disease evidence contamination when multiple diseases
        #    appear in the same Assessment/Plan section.
        section_texts = []
        disease_lower = disease_name.lower()
        # Build first-word keywords for broader matching (e.g. "anxiety" from "anxiety disorder")
        disease_keywords = [w for w in disease_lower.split() if len(w) >= 4]

        for src in detected_disease.get("section_sources", []):
            if sections and src in sections:
                sec_text = sections[src].get("text", "")
                if sec_text and sec_text not in section_texts:
                    # Filter: get lines that mention this specific disease or are the
                    # primary mention's line.  Keep a small window (+/- 1 line) around
                    # each matching line so treatment/monitoring clues aren't cut off.
                    filtered = self._filter_section_for_disease(
                        sec_text, disease_name, disease_keywords
                    )
                    section_texts.append(filtered if filtered else sec_text[:800])

        # Fallback to primary section if section_sources yielded nothing
        if not section_texts:
            section_name = detected_disease.get("section")
            if section_name and sections:
                section_data = sections.get(section_name, {})
                fallback_text = section_data.get("text", "")
                if fallback_text:
                    filtered = self._filter_section_for_disease(
                        fallback_text, disease_name, disease_keywords
                    )
                    section_texts.append(filtered if filtered else fallback_text[:800])
        context["section_context"] = "\n\n".join(section_texts)

        # 4. Related mentions (elsewhere in the document)
        primary_text = context["primary_mention"]
        all_mentions = self._find_disease_sentences(disease_name, sentences_sorted)
        context["related_sentences"] = [
            s for s in all_mentions if s != primary_text
        ]

        # 5. Medication mentions (disease-specific first, then general)
        disease_meds = self._get_disease_medications(disease_name)
        seen_med: set = set()
        for sent in sentences_sorted:
            text = sent["text"]
            if text in seen_med:
                continue
            if disease_meds and self._sentence_matches_keywords(text, disease_meds):
                context["medication_mentions"].append(text)
                seen_med.add(text)
            elif self._sentence_matches_keywords(text, GENERAL_MED_KEYWORDS):
                context["medication_mentions"].append(text)
                seen_med.add(text)

        # 6. Lab/test mentions
        disease_labs = self._get_disease_labs(disease_name)
        seen_lab: set = set()
        for sent in sentences_sorted:
            text = sent["text"]
            if text in seen_lab:
                continue
            if disease_labs and self._sentence_matches_keywords(text, disease_labs):
                context["lab_mentions"].append(text)
                seen_lab.add(text)
            elif self._sentence_matches_keywords(text, GENERAL_LAB_KEYWORDS):
                context["lab_mentions"].append(text)
                seen_lab.add(text)

        # 7. Total context size
        all_text = (
            context["primary_mention"]
            + " ".join(context["context_sentences"])
            + context["section_context"]
            + " ".join(context["related_sentences"])
        )
        context["total_context_chars"] = len(all_text)

        self.logger.info(
            f"Context for '{disease_name}': window={context['window_size']}, "
            f"related={len(context['related_sentences'])}, "
            f"meds={len(context['medication_mentions'])}, "
            f"labs={len(context['lab_mentions'])}"
        )

        return context

    # ── Disease-specific section filter ──────────────────────────────────────

    def _filter_section_for_disease(
        self, section_text: str, disease_name: str, disease_keywords: List[str]
    ) -> str:
        """
        Return only the lines/entries from section_text that are relevant to
        the specific disease. Uses a ±1 line window around each matching line.

        This prevents MEAT evidence cross-contamination when multiple diseases
        share the same Assessment/Plan section.
        """
        lines = [ln for ln in section_text.split("\n") if ln.strip()]
        if not lines:
            return section_text

        disease_lower = disease_name.lower()
        matched_indices: set = set()

        for i, line in enumerate(lines):
            line_lower = line.lower()
            # Direct name match
            if disease_lower in line_lower:
                for j in range(max(0, i - 1), min(len(lines), i + 3)):
                    matched_indices.add(j)
            # Keyword match (any 2 key words of the disease name)
            elif disease_keywords:
                hits = sum(1 for kw in disease_keywords if kw in line_lower)
                if hits >= min(2, len(disease_keywords)):
                    for j in range(max(0, i - 1), min(len(lines), i + 3)):
                        matched_indices.add(j)

        if not matched_indices:
            # Fallback: if no specific match found, return beginning of section
            return "\n".join(lines[:8])

        result_lines = [lines[i] for i in sorted(matched_indices)]
        return "\n".join(result_lines)

    # ── Batch Builder ─────────────────────────────────────────────────────────

    def build_contexts_for_all_diseases(
        self,
        detected_diseases: List[Dict],
        full_text: str,
        sentences: List[Dict],
        sections: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        """
        Build context windows for every non-negated detected disease.

        Returns:
            Dict mapping disease_name → context_data
        """
        contexts: Dict[str, Dict] = {}

        for disease in detected_diseases:
            if disease.get("negated", False):
                self.logger.debug(
                    f"Skipping negated disease: {disease.get('disease_name')}"
                )
                continue

            disease_name = disease["disease_name"]

            # De-duplicate by normalized name if needed
            if disease_name in contexts:
                self.logger.debug(f"Duplicate disease skipped: {disease_name}")
                continue

            contexts[disease_name] = self.build_disease_context(
                disease_name=disease_name,
                full_text=full_text,
                sentences=sentences,
                sections=sections,
                detected_disease=disease,
            )

        self.logger.info(f"Built context windows for {len(contexts)} diseases.")
        return contexts

    def get_context_summary(self, context: Dict) -> str:
        """
        Return a short human-readable summary of a context window for prompt injection.
        """
        parts: List[str] = []

        if context.get("primary_mention"):
            parts.append(f"Primary mention: {context['primary_mention']}")

        if context.get("context_sentences"):
            parts.append("Surrounding context:\n" + "\n".join(context["context_sentences"]))

        if context.get("section_context"):
            # Truncate long sections
            sc = context["section_context"][:1500]
            parts.append(f"Section context:\n{sc}")

        if context.get("medication_mentions"):
            meds = context["medication_mentions"][:5]  # top 5
            parts.append("Medications:\n" + "\n".join(meds))

        if context.get("lab_mentions"):
            labs = context["lab_mentions"][:5]  # top 5
            parts.append("Lab results:\n" + "\n".join(labs))

        return "\n\n".join(parts)


# ── Singleton ────────────────────────────────────────────────────────────────
context_builder = ContextWindowBuilder()
