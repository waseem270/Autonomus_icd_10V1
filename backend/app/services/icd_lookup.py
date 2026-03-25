import sqlite3
import re
import unicodedata
from typing import List, Dict, Optional, Any, Set
import logging
from fuzzywuzzy import fuzz
from pathlib import Path
import os

from ..core.config import settings

logger = logging.getLogger(__name__)


def format_icd_code(code: str) -> str:
    """
    Format a raw ICD-10 code to standard dot notation.

    ICD-10 codes have a dot after the third character when the code is
    longer than three characters:
        I10        → I10         (no change — 3-char code has no decimal)
        F419       → F41.9
        G4700      → G47.00
        M810       → M81.0
        F0150      → F01.50
        S3282XA    → S32.82XA
        M80001A    → M80.001A
    Already-formatted codes (containing a dot) are returned unchanged.
    """
    if not code or "." in code or len(code) <= 3:
        return code
    return code[:3] + "." + code[3:]


# ── Clinical synonym expansion ───────────────────────────────────────────────
# Maps common clinical terms to their ICD-10 description counterparts.
# Applied BEFORE search to increase recall.
DISEASE_SYNONYMS: Dict[str, List[str]] = {
    # ── Z-code clinical terms ──────────────────────────────────────────────
    "postoperative care": ["encounter for other specified surgical aftercare"],
    "post-operative care": ["encounter for other specified surgical aftercare"],
    "postoperative care following surgery": ["encounter for other specified surgical aftercare"],
    "surgical aftercare": ["encounter for other specified surgical aftercare"],
    "post surgical care": ["encounter for other specified surgical aftercare"],
    "postop care": ["encounter for other specified surgical aftercare"],
    "long term anticoagulant use": ["long term (current) use of anticoagulants"],
    "long term insulin use": ["long term (current) use of insulin"],
    # ── Abbreviations ──────────────────────────────────────────────────────
    "djd": ["osteoarthritis"],
    "htn": ["essential (primary) hypertension"],
    "dm": ["diabetes mellitus"],
    "dm2": ["type 2 diabetes mellitus"],
    "dm1": ["type 1 diabetes mellitus"],
    "t2dm": ["type 2 diabetes mellitus"],
    "t1dm": ["type 1 diabetes mellitus"],
    "ckd": ["chronic kidney disease"],
    "copd": ["chronic obstructive pulmonary disease"],
    "gerd": ["gastroesophageal reflux disease"],
    "afib": ["atrial fibrillation"],
    "af": ["atrial fibrillation"],
    "mi": ["myocardial infarction"],
    "cva": ["cerebrovascular accident", "stroke"],
    "chf": ["heart failure"],
    "cad": ["coronary artery disease"],
    "pad": ["peripheral artery disease"],
    "pvd": ["peripheral vascular disease"],
    "uti": ["urinary tract infection"],
    "dvt": ["deep vein thrombosis"],
    "pe": ["pulmonary embolism"],
    "ra": ["rheumatoid arthritis"],
    "sle": ["systemic lupus erythematosus"],
    "ms": ["multiple sclerosis"],
    "tia": ["transient ischemic attack"],
    "bph": ["benign prostatic hyperplasia"],
    "pcos": ["polycystic ovarian syndrome"],
    "ptsd": ["post-traumatic stress disorder"],
    "ocd": ["obsessive-compulsive disorder"],
    "adhd": ["attention-deficit hyperactivity disorder"],
    "asd": ["autism spectrum disorder"],
    "ibs": ["irritable bowel syndrome"],
    "ibd": ["inflammatory bowel disease"],
    "ckd stage 3": ["chronic kidney disease, stage 3"],
    "ckd stage 4": ["chronic kidney disease, stage 4"],
    "ckd stage 5": ["chronic kidney disease, stage 5"],
    # ── Common generic names → specific ICD-searchable descriptions ────────
    "high blood pressure": ["essential (primary) hypertension"],
    "degenerative joint disease": ["osteoarthritis", "degenerative arthritis"],
    "hypertension": ["essential (primary) hypertension"],
    "essential hypertension": ["essential (primary) hypertension"],
    "anxiety": ["anxiety disorder, unspecified"],
    "anxiety disorder": ["anxiety disorder, unspecified"],
    "generalized anxiety": ["generalized anxiety disorder"],
    "panic disorder": ["panic disorder without agoraphobia"],
    "depression": ["major depressive disorder, recurrent, unspecified"],
    "major depression": ["major depressive disorder, recurrent, unspecified"],
    "major depressive disorder": ["major depressive disorder, recurrent, unspecified"],
    "major depression recurrent moderate": ["major depressive disorder, recurrent, moderate"],
    "major depression moderate recurrent": ["major depressive disorder, recurrent, moderate"],
    "major depression, recurrent, moderate": ["major depressive disorder, recurrent, moderate"],
    "major depression, moderate, recurrent": ["major depressive disorder, recurrent, moderate"],
    "depression recurrent moderate": ["major depressive disorder, recurrent, moderate"],
    "depression moderate recurrent": ["major depressive disorder, recurrent, moderate"],
    "depression, recurrent, moderate": ["major depressive disorder, recurrent, moderate"],
    "depression, moderate, recurrent": ["major depressive disorder, recurrent, moderate"],
    "major depression single episode": ["major depressive disorder, single episode, unspecified"],
    "moderate major depression": ["major depressive disorder, single episode, moderate"],
    "severe major depression": ["major depressive disorder, single episode, severe"],
    "insomnia": ["insomnia, unspecified"],
    "chronic insomnia": ["insomnia, unspecified"],
    "sleep disorder": ["sleep disorder, unspecified"],
    "osteoporosis": ["age-related osteoporosis without current pathological fracture"],
    "osteoporosis without fracture": ["age-related osteoporosis without current pathological fracture"],
    "angina": ["angina pectoris"],
    "angina pectoris": ["unspecified angina pectoris"],
    "renal disorder": ["chronic kidney disease"],
    "morbid obesity": ["morbid (severe) obesity due to excess calories"],
    "obesity": ["obesity, unspecified"],
    "overweight": ["overweight"],
    "diabetes mellitus type 2": ["type 2 diabetes mellitus"],
    "type 2 diabetes": ["type 2 diabetes mellitus"],
    "type 2 diabetes mellitus": ["type 2 diabetes mellitus without complications"],
    "diabetes mellitus type 1": ["type 1 diabetes mellitus"],
    "type 1 diabetes": ["type 1 diabetes mellitus"],
    "atherosclerosis": ["atherosclerosis of aorta"],
    "secondary diabetes mellitus": ["diabetes mellitus due to underlying condition"],
    "secondary diabetes": ["diabetes mellitus due to underlying condition"],
    "glaucoma": ["unspecified glaucoma"],
    "primary malignant neoplasm of skin": ["malignant neoplasm of skin, unspecified", "unspecified malignant neoplasm of skin, unspecified"],
    "malignant neoplasm of skin": ["other malignant neoplasm of skin", "malignant neoplasm of skin, unspecified"],
    "heart murmur": ["cardiac murmur, unspecified"],
    "mononeuropathy": ["mononeuropathy, unspecified"],
    "sleep apnea": ["sleep apnea, unspecified"],
    "obstructive sleep apnea": ["obstructive sleep apnea (adult) (pediatric)"],
    "polyneuropathy": ["polyneuropathy, unspecified"],
    "neuropathy": ["polyneuropathy, unspecified", "neuropathy, unspecified"],
    "peripheral neuropathy": ["polyneuropathy, unspecified"],
    "hypothyroidism": ["hypothyroidism, unspecified"],
    "hyperthyroidism": ["thyrotoxicosis, unspecified"],
    "hyperlipidemia": ["hyperlipidemia, unspecified"],
    "dyslipidemia": ["hyperlipidemia, unspecified"],
    "high cholesterol": ["hyperlipidemia, unspecified"],
    "atrial fibrillation": ["atrial fibrillation, unspecified"],
    "chronic atrial fibrillation": ["chronic atrial fibrillation, unspecified"],
    "heart failure": ["heart failure, unspecified"],
    "congestive heart failure": ["congestive heart failure, unspecified"],
    "chf systolic": ["systolic (congestive) heart failure, unspecified"],
    "coronary artery disease": ["atherosclerotic heart disease of native coronary artery without angina pectoris"],
    "stroke": ["cerebral infarction, unspecified"],
    "cerebrovascular accident": ["cerebral infarction, unspecified"],
    "migraine": ["migraine, unspecified, not intractable"],
    "headache": ["headache, unspecified"],
    "chronic headache": ["headache, unspecified"],
    "back pain": ["low back pain"],
    "lower back pain": ["low back pain"],
    "low back pain": ["low back pain"],
    "neck pain": ["neck pain, unspecified"],
    "knee pain": ["pain in knee"],
    "hip pain": ["pain in hip, unspecified"],
    "shoulder pain": ["pain in shoulder"],
    "leg pain": ["pain in leg, unspecified"],
    "left leg pain": ["pain in left leg"],
    "right leg pain": ["pain in right leg"],
    "pelvic fracture": ["unspecified fracture of unspecified pubis, initial encounter for closed fracture"],
    "pelvic pain": ["pelvic and perineal pain"],
    "arthritis": ["arthritis, unspecified"],
    "osteoarthritis": ["osteoarthritis, unspecified"],
    "closed pelvic fracture": ["unspecified fracture of unspecified pubis, initial encounter for closed fracture"],
    "superior pubic ramus fracture": ["fracture of superior rim of right pubis, initial encounter for closed fracture"],
    "rheumatoid arthritis": ["rheumatoid arthritis, unspecified"],
    "fibromyalgia": ["fibromyalgia"],
    "chronic pain": ["chronic pain, not elsewhere classified"],
    "asthma": ["asthma, unspecified, uncomplicated"],
    "copd exacerbation": ["other chronic obstructive pulmonary disease with acute exacerbation"],
    "pneumonia": ["pneumonia, unspecified organism"],
    "urinary tract infection": ["urinary tract infection, site not specified"],
    "uti unspecified": ["urinary tract infection, site not specified"],
    "anemia": ["anemia, unspecified"],
    "iron deficiency anemia": ["iron deficiency anemia, unspecified"],
    "vitamin d deficiency": ["vitamin d deficiency, unspecified"],
    "hypokalemia": ["hypokalemia"],
    "hyponatremia": ["hyponatremia"],
    "alcohol use disorder": ["alcohol dependence, uncomplicated"],
    "alcohol abuse": ["alcohol abuse, uncomplicated"],
    "substance abuse": ["other psychoactive substance abuse, uncomplicated"],
    "bipolar disorder": ["bipolar disorder, unspecified"],
    "schizophrenia": ["schizophrenia, unspecified"],
    "dementia": ["unspecified dementia without behavioral disturbance"],
    "alzheimer disease": ["alzheimer's disease, unspecified"],
    "parkinsons disease": ["parkinson's disease"],
    "parkinson disease": ["parkinson's disease"],
    "epilepsy": ["epilepsy, unspecified, not intractable"],
    "seizure disorder": ["epilepsy, unspecified, not intractable"],
    "peripheral artery disease": ["peripheral vascular disease, unspecified"],
    "venous insufficiency": ["chronic venous insufficiency, unspecified"],
    "varicose veins": ["varicose veins of lower extremity without ulcer or inflammation"],
    "gout": ["gout, unspecified"],
    "lupus": ["systemic lupus erythematosus, unspecified"],
    "psoriasis": ["psoriasis, unspecified"],
    "eczema": ["atopic dermatitis, unspecified"],
    "dermatitis": ["contact dermatitis, unspecified cause"],
    "cellulitis": ["cellulitis, unspecified"],
    "peptic ulcer": ["peptic ulcer, site unspecified, without hemorrhage or perforation"],
    "gastric ulcer": ["gastric ulcer, unspecified"],
    "crohn disease": ["crohn's disease of small intestine without complications"],
    "crohns disease": ["crohn's disease of small intestine without complications"],
    "ulcerative colitis": ["ulcerative colitis, unspecified, without complications"],
    "liver disease": ["liver disease, unspecified"],
    "cirrhosis": ["hepatic cirrhosis, unspecified"],
    "fatty liver": ["fatty (change of) liver, not elsewhere classified"],
    "hepatitis c": ["chronic viral hepatitis c"],
    "hepatitis b": ["chronic viral hepatitis b without delta-agent"],
    "prostate cancer": ["malignant neoplasm of prostate"],
    "breast cancer": ["malignant neoplasm of breast, unspecified"],
    "lung cancer": ["malignant neoplasm of bronchus and lung, unspecified"],
    "colon cancer": ["malignant neoplasm of colon, unspecified"],
    "cervical cancer": ["malignant neoplasm of cervix uteri, unspecified"],
    "ovarian cancer": ["malignant neoplasm of ovary"],
    "bladder cancer": ["malignant neoplasm of bladder, unspecified"],
    "kidney cancer": ["malignant neoplasm of kidney, except renal pelvis"],
    "thyroid cancer": ["malignant neoplasm of thyroid gland"],
    "deep vein thrombosis": ["deep vein thrombosis of unspecified deep vessels of proximal lower extremity"],
    "pulmonary embolism": ["other pulmonary embolism without acute cor pulmonale"],
    "aortic stenosis": ["aortic stenosis, unspecified"],
    "mitral valve prolapse": ["mitral valve prolapse"],
    "benign prostatic hyperplasia": ["benign prostatic hyperplasia without lower urinary tract symptoms"],
    "bph without symptoms": ["benign prostatic hyperplasia without lower urinary tract symptoms"],
    "polycystic ovary syndrome": ["polycystic ovarian syndrome"],
    "endometriosis": ["endometriosis, unspecified"],
    "uterine fibroids": ["leiomyoma of uterus, unspecified"],
    "carpal tunnel syndrome": ["carpal tunnel syndrome, unspecified upper limb"],
    "carpal tunnel": ["carpal tunnel syndrome, unspecified upper limb"],
    "rotator cuff tear": ["complete rotator cuff tear or rupture of unspecified shoulder"],
    "tennis elbow": ["lateral epicondylitis"],
    "plantar fasciitis": ["plantar fasciitis"],
    "cataracts": ["age-related cataract, unspecified eye"],
    "cataract": ["age-related cataract, unspecified eye"],
    "macular degeneration": ["age-related macular degeneration, unspecified, unspecified eye"],
    "glaucoma unspecified": ["unspecified glaucoma"],
    "hearing loss": ["hearing loss, unspecified"],
    "benign paroxysmal positional vertigo": ["benign paroxysmal vertigo, unspecified ear"],
    "bppv": ["benign paroxysmal vertigo, unspecified ear"],
    "tinnitus": ["tinnitus, unspecified"],
    "urinary incontinence": ["urinary incontinence, unspecified"],
    "stress urinary incontinence": ["stress incontinence (female) (male)"],
    "overactive bladder": ["overactive bladder"],
    "erectile dysfunction": ["erectile dysfunction, unspecified"],
    "hypothyroidism unspecified": ["hypothyroidism, unspecified"],
}


def _strip_diacritics(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _clean_for_search(name: str) -> str:
    """
    Strip abbreviation expansion artifacts and noise for ICD search.
    e.g. "Type 2 Diabetes Mellitus (DM)" → "type 2 diabetes mellitus"
    Also normalizes unicode diacritics: Sjögren → sjogren
    Also strips possessive 's: Sjogren's → sjogren
    """
    cleaned = _strip_diacritics(name).lower().strip()
    # Remove possessive 's (e.g., "Sjogren's" → "Sjogren")
    cleaned = re.sub(r"[''']s\b", "", cleaned)
    # Remove trailing parenthetical abbreviations: "(DM)", "(HTN)", etc.
    cleaned = re.sub(r'\s*\([a-z]{1,8}\)\s*$', '', cleaned)
    # Remove leading/trailing punctuation
    cleaned = re.sub(r'^[,.\s]+|[,.\s]+$', '', cleaned)
    # Collapse whitespace
    cleaned = " ".join(cleaned.split())
    return cleaned


def _extract_search_words(name: str) -> List[str]:
    """
    Extract clinically meaningful words (>=3 chars) for multi-word LIKE queries.
    Drops common stop words that add noise.
    """
    STOP = {
        "the", "and", "for", "with", "without", "not", "from", "due",
        "type", "stage", "grade", "unspecified", "other", "bilateral",
        "left", "right", "acute", "chronic", "primary", "secondary",
    }
    words = _clean_for_search(name).split()
    return [w for w in words if len(w) >= 3 and w not in STOP]


class ICDLookupService:
    """
    Service to lookup ICD-10 codes from a structured SQLite database.
    
    Uses a multi-strategy search approach:
    1. Exact description match
    2. Multi-word LIKE query (all significant words)
    3. Per-word LIKE queries (broadening)
    4. Fuzzy scoring on the combined candidate pool
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Use config-based path (resolves from project root, not cwd)
        from ..core.config import DATABASE_DIR
        self.db_path = str(DATABASE_DIR / "medical_icd.db")
        
        if not Path(self.db_path).exists():
            self.logger.warning(f"ICD database not found at {self.db_path}. Search will fail until seeded.")
        else:
            self.logger.info(f"ICD Lookup Service initialized with DB: {self.db_path}")
    
    def _get_connection(self) -> Optional[sqlite3.Connection]:
        """Get a database connection, or None if DB doesn't exist."""
        if not Path(self.db_path).exists():
            return None
        return sqlite3.connect(self.db_path)

    def exact_match(self, disease_name: str) -> Optional[Dict[str, Any]]:
        """
        Search for an exact ICD code match based on the description.
        Also tries synonym-expanded variants and diacritic-stripped forms.
        """
        conn = self._get_connection()
        if not conn:
            return None

        query = """
        SELECT code, description 
        FROM icd10_codes 
        WHERE LOWER(description) = LOWER(?)
        LIMIT 1
        """
        
        # Build search variants: original + diacritic-stripped + synonyms
        cleaned = _clean_for_search(disease_name)
        search_terms = [cleaned]
        
        # Add synonym-expanded variants
        # Sort by descending key length so more-specific keys (e.g. 'left leg pain')
        # are tried before shorter substrings (e.g. 'leg pain'),
        # preventing the shorter synonym from winning in the DB query loop.
        for key in sorted(DISEASE_SYNONYMS, key=len, reverse=True):
            synonyms = DISEASE_SYNONYMS[key]
            if re.search(r'\b' + re.escape(key) + r'\b', cleaned):
                for syn in synonyms:
                    variant = re.sub(r'\b' + re.escape(key) + r'\b', syn, cleaned)
                    if variant not in search_terms:
                        search_terms.append(variant)
                for syn in synonyms:
                    if syn not in search_terms:
                        search_terms.append(syn)
        
        try:
            cursor = conn.cursor()
            for term in search_terms:
                cursor.execute(query, (term,))
                result = cursor.fetchone()
                if result:
                    self.logger.info(f"Exact match found: '{disease_name}' -> {result[0]} (via '{term}')")
                    conn.close()
                    return {
                        "icd_code": format_icd_code(result[0]),
                        "description": result[1],
                        "match_type": "exact",
                        "confidence": 1.0
                    }
            
            conn.close()
            return None
            
        except Exception as e:
            self.logger.error(f"Exact match query failed: {e}")
            conn.close()
            return None
    
    def fuzzy_match(
        self, 
        disease_name: str, 
        threshold: Optional[float] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Multi-strategy fuzzy search for ICD codes.
        
        Strategy:
        1. Expand disease name with clinical synonyms.
        2. Build a broad candidate pool using multiple LIKE queries.
        3. Score with Token Sort Ratio + contradiction penalties.
        4. Return top-N above threshold.
        """
        conn = self._get_connection()
        if not conn:
            return []

        if threshold is None:
            threshold = getattr(settings, "ICD_FUZZY_THRESHOLD", 0.70) * 100
        else:
            threshold = threshold * 100
        
        cleaned = _clean_for_search(disease_name)
        search_words = _extract_search_words(disease_name)
        
        # Build list of search variants (original + synonyms)
        search_variants = [cleaned]
        for key in sorted(DISEASE_SYNONYMS, key=len, reverse=True):
            synonyms = DISEASE_SYNONYMS[key]
            if re.search(r'\b' + re.escape(key) + r'\b', cleaned):
                for syn in synonyms:
                    variant = re.sub(r'\b' + re.escape(key) + r'\b', syn, cleaned)
                    if variant != cleaned:
                        search_variants.append(variant)
                search_variants.extend(synonyms)  # Also search synonyms directly
        
        try:
            cursor = conn.cursor()
            seen_codes: Set[str] = set()
            candidates: List[tuple] = []
            
            for variant in search_variants:
                variant_words = _extract_search_words(variant)
                
                # ── Strategy 1: Full phrase LIKE ──
                phrase_query = """
                SELECT code, description FROM icd10_codes
                WHERE LOWER(description) LIKE LOWER(?)
                LIMIT 200
                """
                cursor.execute(phrase_query, (f"%{variant}%",))
                for row in cursor.fetchall():
                    if row[0] not in seen_codes:
                        candidates.append(row)
                        seen_codes.add(row[0])
                
                # ── Strategy 2: Multi-word AND query ──
                if len(variant_words) >= 2:
                    where_clauses = " AND ".join(
                        [f"LOWER(description) LIKE ?" for _ in variant_words]
                    )
                    and_query = f"""
                    SELECT code, description FROM icd10_codes
                    WHERE {where_clauses}
                    LIMIT 500
                    """
                    params = [f"%{w}%" for w in variant_words]
                    cursor.execute(and_query, params)
                    for row in cursor.fetchall():
                        if row[0] not in seen_codes:
                            candidates.append(row)
                            seen_codes.add(row[0])
            
            # ── Strategy 3: Per-word broadening (original terms only) ──
            for word in search_words[:3]:
                if len(word) < 4:
                    continue
                word_query = """
                SELECT code, description FROM icd10_codes
                WHERE LOWER(description) LIKE LOWER(?)
                LIMIT 100
                """
                cursor.execute(word_query, (f"%{word}%",))
                for row in cursor.fetchall():
                    if row[0] not in seen_codes:
                        candidates.append(row)
                        seen_codes.add(row[0])
            
            conn.close()
            
            # ── Score all candidates with improved scoring ──
            scored_results = []
            for code, description in candidates:
                desc_lower = _strip_diacritics(description).lower()
                
                # Primary: token_set_ratio (handles token subsets well)
                set_score = fuzz.token_set_ratio(cleaned, desc_lower)
                
                # ── Contradiction penalty ──
                # Critical: "type 1" vs "type 2", "stage 2" vs "stage 5", etc.
                score = self._apply_contradiction_penalty(
                    cleaned, desc_lower, float(set_score)
                )
                
                # ── Exact substring bonus ──
                if cleaned in desc_lower:
                    score = min(100, score + 5)
                
                # ── Negation contradiction ──
                # "myelopathy" matching "without myelopathy" is wrong
                # Check if the candidate negates the disease itself
                for neg_prefix in ("without ", "not ", "no ", "absent "):
                    if neg_prefix + cleaned in desc_lower:
                        score -= 30  # Heavy penalty
                        break
                    # Also check individual significant words
                    for w in _extract_search_words(cleaned):
                        if len(w) >= 5 and (neg_prefix + w) in desc_lower:
                            score -= 20
                            break

                # ── Primary-term position bonus / secondary-only penalty ──
                # If the description STARTS with (or primarily IS) the search term,
                # it is the main condition → big bonus.
                # If the search term only appears as a secondary qualifier near the
                # end (e.g. "Vascular dementia … and anxiety"), penalise heavily.
                first_word = cleaned.split()[0] if cleaned.split() else ""
                if first_word and len(first_word) >= 4:
                    # Bonus: description's first meaningful word matches
                    if desc_lower.startswith(first_word):
                        score = min(100, score + 15)
                    # Penalty: search term appears only after a comma/slash/and separator
                    elif re.search(
                        r'(?:,\s*and\s+|,\s+|;\s+|\band\s+)' + re.escape(first_word),
                        desc_lower,
                    ):
                        score -= 20

                # ── Also score using synonym variants ──
                # If the original disease name doesn't match well, the synonym might
                for variant in search_variants[1:]:  # Skip original (already scored)
                    vscore = fuzz.token_set_ratio(variant, desc_lower)
                    vscore = self._apply_contradiction_penalty(variant, desc_lower, float(vscore))
                    if variant in desc_lower:
                        vscore = min(100, vscore + 5)
                    if vscore > score:
                        score = vscore
                
                # ── "Without" qualifier enforcement ──
                # If search says "without X" but candidate says "with X" (not "without"),
                # apply a heavy penalty. Variants should not override this.
                if "without " in cleaned:
                    import re as _re3
                    if _re3.search(r'\bwith\s+(?!out\b)', desc_lower):
                        score -= 20  # Candidate has "with ..." when search says "without"
                    if "without " in desc_lower:
                        score += 5  # Bonus for matching "without" qualifier
                
                # ── Specificity preference ──
                # When scoring is tied at high levels, prefer simpler/unspecified codes
                # if the disease name doesn't mention specific complications
                if score >= 85:
                    complication_words = [
                        "complicated", "complicating",
                        "hyperosmolarity", "ketoacidosis",
                        "nephropathy", "retinopathy", "neuropathy", "coma",
                        "gangrene", "bypass graft", "native", "prosthetic",
                        "macular", "proliferative", "cerebral", "coronary",
                        "peripheral", "unstable", "refractory",
                        "presymptomatic",
                        # Obstetric / gestational context (should not match generic conditions)
                        "pregnancy", "trimester", "gestational", "postpartum",
                        "antepartum", "childbirth", "fetus", "newborn",
                        "neonatal", "perinatal", "puerperal",
                        # Infectious / parasitic disease context
                        "onchocerciasis", "toxoplasm", "zoster", "herpes",
                        "gonococcal", "syphilitic", "chlamydial", "tubercul",
                        # External cause / injury context
                        "traumatic", "sequela",
                        # Other disease context (condition is mentioned as complication)
                        "in diseases classified elsewhere",
                        "due to underlying condition",
                        # Body-site specificity (when search is site-unspecified)
                        "anal ", "scalp", "eyelid", "trunk", "nipple",
                        "areola", "vulva", "penis", "scrotal",
                    ]
                    # Check "with" carefully: avoid matching "without"
                    def _has_complication_context(text):
                        for w in complication_words:
                            if w in text:
                                return True
                        # Check "with " but exclude "without "
                        import re as _re2
                        if _re2.search(r'\bwith\s+(?!out\b)', text):
                            return True
                        if "due to " in text or "in " in text:
                            return True
                        return False

                    has_specific_in_search = _has_complication_context(cleaned)
                    has_specific_in_candidate = _has_complication_context(desc_lower)
                    
                    if has_specific_in_candidate and not has_specific_in_search:
                        # Count how many extra complication words appear
                        extra = sum(1 for w in complication_words if w in desc_lower and w not in cleaned)
                        penalty = max(5, min(15, extra * 5))  # At least -5 for any complication context mismatch
                        score -= penalty
                    
                    # Prefer "unspecified" or "without complications" codes when search is generic
                    if not has_specific_in_search:
                        if "without complications" in desc_lower:
                            score += 4  # Strong bonus: this is the "NOS" code
                        elif "unspecified" in desc_lower:
                            score += 3  # Unspecified is also a good default
                
                if score >= threshold:
                    # Cap score at 100 for confidence (but keep raw for sorting)
                    capped_conf = min(1.0, round(score / 100.0, 3))
                    scored_results.append({
                        "icd_code": format_icd_code(code),
                        "description": description,
                        "match_type": "fuzzy",
                        "confidence": capped_conf,
                        "similarity_score": score  # raw score for sorting
                    })
            
            scored_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            top_results = scored_results[:max_results]
            
            if top_results:
                self.logger.info(
                    f"Fuzzy match for '{disease_name}': {len(top_results)} candidates "
                    f"(pool={len(candidates)}, variants={len(search_variants)}). "
                    f"Best: {top_results[0]['icd_code']} ({top_results[0]['similarity_score']:.0f}%)"
                )
            else:
                self.logger.info(
                    f"No fuzzy matches above threshold for '{disease_name}' "
                    f"(pool={len(candidates)}, threshold={threshold:.0f}%)"
                )
            
            return top_results
            
        except Exception as e:
            self.logger.error(f"Fuzzy match query failed: {e}")
            if conn:
                conn.close()
            return []

    def _apply_contradiction_penalty(
        self, search: str, candidate: str, score: float
    ) -> float:
        """
        Penalize candidates that contain contradicting clinical qualifiers.
        e.g. searching "type 2" but candidate says "type 1" → big penalty.
        """
        import re as _re
        
        # ── Numbered qualifier contradictions ──
        # Handles type 1/2, stage 1-5, class I-IV, grade 1-4 dynamically
        for prefix in ("type ", "stage ", "grade ", "class "):
            search_match = _re.search(prefix + r'(\w+)', search)
            if search_match:
                search_val = search_match.group(1)
                cand_match = _re.search(prefix + r'(\w+)', candidate)
                if cand_match and cand_match.group(1) != search_val:
                    score -= 30
                    break
        
        # ── Static contradictions ──
        STATIC_CONTRADICTIONS = [
            ("left", "right"), ("right", "left"),
            ("acute", "chronic"), ("chronic", "acute"),
            ("unilateral", "bilateral"), ("bilateral", "unilateral"),
            ("single episode", "recurrent"), ("recurrent", "single episode"),
            ("unstable", "stable"),
            ("mild", "severe"), ("severe", "mild"),
            ("moderate", "severe"), ("severe", "moderate"),
            ("aorta", "coronary"), ("coronary", "aorta"),
        ]
        
        for search_term, contradiction in STATIC_CONTRADICTIONS:
            if search_term in search and contradiction in candidate:
                score -= 25
                break
        
        return max(0, score)
    
    def search_by_code(self, icd_code: str) -> Optional[Dict[str, str]]:
        """Direct lookup by ICD-10 code string."""
        conn = self._get_connection()
        if not conn:
            return None

        query = "SELECT code, description FROM icd10_codes WHERE code = ? LIMIT 1"
        try:
            cursor = conn.cursor()
            cursor.execute(query, (icd_code,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {"icd_code": format_icd_code(result[0]), "description": result[1]}
            return None
        except Exception as e:
            self.logger.error(f"Code lookup failed: {e}")
            conn.close()
            return None
    
    def search_combined(
        self,
        disease_name: str,
        normalized_name: str
    ) -> Dict[str, Any]:
        """
        Multi-tier search strategy:
        1. Exact match on Normalized name (highest weight)
        2. Exact match on Original name (fallback)
        3. Fuzzy match on Normalized name (widening the net)
        """
        result = {
            "exact_match": None,
            "fuzzy_matches": [],
            "best_match": None,
            "search_method": "none"
        }
        
        # Tier 1: Exact Normalized
        exact = self.exact_match(normalized_name)
        if exact:
            result.update({
                "exact_match": exact,
                "best_match": exact,
                "search_method": "exact_normalized"
            })
            return result
        
        # Tier 2: Exact Original
        if disease_name.lower() != normalized_name.lower():
            exact_orig = self.exact_match(disease_name)
            if exact_orig:
                result.update({
                    "exact_match": exact_orig,
                    "best_match": exact_orig,
                    "search_method": "exact_original"
                })
                return result
        
        # Tier 3: Fuzzy broadening
        fuzzy_list = self.fuzzy_match(normalized_name, max_results=5)
        result["fuzzy_matches"] = fuzzy_list
        
        if fuzzy_list:
            result["best_match"] = fuzzy_list[0]
            result["search_method"] = "fuzzy"
            
        return result

# Singleton instance for high-performance reuse
icd_lookup = ICDLookupService()
