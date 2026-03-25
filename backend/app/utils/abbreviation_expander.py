import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Comprehensive Medical Abbreviation Dictionary (100+ items)
MEDICAL_ABBREVIATIONS = {
    # Diagnoses & Conditions
    "DM": "Diabetes Mellitus",
    "DM1": "Type 1 Diabetes Mellitus",
    "DM2": "Type 2 Diabetes Mellitus",
    "T1DM": "Type 1 Diabetes Mellitus",
    "T2DM": "Type 2 Diabetes Mellitus",
    "HTN": "Hypertension",
    "CHF": "Congestive Heart Failure",
    "CAD": "Coronary Artery Disease",
    "COPD": "Chronic Obstructive Pulmonary Disease",
    "MI": "Myocardial Infarction",
    "CVA": "Cerebrovascular Accident",
    "CKD": "Chronic Kidney Disease",
    "ESRD": "End Stage Renal Disease",
    "GERD": "Gastroesophageal Reflux Disease",
    "OSA": "Obstructive Sleep Apnea",
    "RA": "Rheumatoid Arthritis",
    "OA": "Osteoarthritis",
    "PE": "Pulmonary Embolism",
    "DVT": "Deep Vein Thrombosis",
    "UTI": "Urinary Tract Infection",
    "URI": "Upper Respiratory Infection",
    "PNA": "Pneumonia",
    "SLE": "Systemic Lupus Erythematosus",
    "IBS": "Irritable Bowel Syndrome",
    "IBD": "Inflammatory Bowel Disease",
    "AFib": "Atrial Fibrillation",
    "SVT": "Supraventricular Tachycardia",
    "MS": "Multiple Sclerosis",
    "CP": "Chest Pain",
    "ADHD": "Attention Deficit Hyperactivity Disorder",
    "ASD": "Autism Spectrum Disorder",
    "MDD": "Major Depressive Disorder",
    "GAD": "Generalized Anxiety Disorder",
    "HLD": "Hyperlipidemia",
    "BPH": "Benign Prostatic Hyperplasia",
    "AAA": "Abdominal Aortic Aneurysm",
    "STD": "Sexually Transmitted Disease",
    "STI": "Sexually Transmitted Infection",
    
    # Symptoms & Physical Findings
    "SOB": "Shortness of Breath",
    "N/V": "Nausea and Vomiting",
    "HA": "Headache",
    "DOE": "Dyspnea on Exertion",
    "RUQ": "Right Upper Quadrant",
    "LUQ": "Left Upper Quadrant",
    "RLQ": "Right Lower Quadrant",
    "LLQ": "Left Lower Quadrant",
    "BRBPR": "Bright Red Blood Per Rectum",
    "BM": "Bowel Movement",
    "ADL": "Activities of Daily Living",
    "NAD": "No Acute Distress",
    "WNL": "Within Normal Limits",
    "RRR": "Regular Rate and Rhythm",
    "JVD": "Jugular Venous Distension",
    "HSM": "Hepatosplenomegaly",
    "PERRLA": "Pupils Equal, Round, Reactive to Light and Accommodation",
    
    # Medications & Classes
    "ASA": "Aspirin",
    "APAP": "Acetaminophen",
    "HCTZ": "Hydrochlorothiazide",
    "BB": "Beta Blocker",
    "ACE-I": "ACE Inhibitor",
    "ACEI": "ACE Inhibitor",
    "ARB": "Angiotensin Receptor Blocker",
    "PPI": "Proton Pump Inhibitor",
    "NSAID": "Non-Steroidal Anti-Inflammatory Drug",
    "SSRI": "Selective Serotonin Reuptake Inhibitor",
    "SNRI": "Serotonin-Norepinephrine Reuptake Inhibitor",
    "TCA": "Tricyclic Antidepressant",
    "MDI": "Metered Dose Inhaler",
    "OTC": "Over The Counter",
    
    # Lab Tests & Imaging
    "CBC": "Complete Blood Count",
    "CMP": "Comprehensive Metabolic Panel",
    "BMP": "Basic Metabolic Panel",
    "HbA1c": "Hemoglobin A1c",
    "A1c": "Hemoglobin A1c",
    "TSH": "Thyroid Stimulating Hormone",
    "LFT": "Liver Function Test",
    "PT": "Prothrombin Time",
    "INR": "International Normalized Ratio",
    "ESR": "Erythrocyte Sedimentation Rate",
    "CRP": "C-Reactive Protein",
    "ABG": "Arterial Blood Gas",
    "EKG": "Electrocardiogram",
    "ECG": "Electrocardiogram",
    "CXR": "Chest X-Ray",
    "MRI": "Magnetic Resonance Imaging",
    "CT": "Computed Tomography",
    "US": "Ultrasound",
    "EEG": "Electroencephalogram",
    "EMG": "Electromyogram",
    
    # General Medical & Professional
    "Pt": "Patient",
    "Hx": "History",
    "Dx": "Diagnosis",
    "Tx": "Treatment",
    "Rx": "Prescription",
    "Sx": "Symptoms",
    "F/U": "Follow Up",
    "ROS": "Review of Systems",
    "PMH": "Past Medical History",
    "PSH": "Past Surgical History",
    "FH": "Family History",
    "SH": "Social History",
    "NKDA": "No Known Drug Allergies",
    "DNR": "Do Not Resuscitate",
    "DNI": "Do Not Intubate",
    "NPO": "Nothing by Mouth",
    "PRN": "As Needed",
    "BID": "Twice Daily",
    "TID": "Three Times Daily",
    "QID": "Four Times Daily",
    "QD": "Daily",
    "QHS": "At Bedtime",
    "PO": "By Mouth",
    "IV": "Intravenous",
    "IM": "Intramuscular",
    "SQ": "Subcutaneous",
    "SL": "Sublingual"
}

# Contexts for disambiguation
CONTEXT_DEPENDENT_ABBREV = {
    "RA": {
        "default": "Rheumatoid Arthritis",
        "cardiac|heart|atrium|valve|ventricle": "Right Atrium"
    },
    "MS": {
        "default": "Multiple Sclerosis",
        "heart|murmur|valve|mitral|stenosis": "Mitral Stenosis"
    },
    "CP": {
        "default": "Chest Pain",
        "cerebral|brain|developmental|palsy": "Cerebral Palsy"
    },
    "PE": {
        "default": "Pulmonary Embolism",
        "physical|exam|assessment": "Physical Examination"
    }
}

def expand_abbreviations(text: str, context_aware: bool = True) -> str:
    """
    Expand medical abbreviations in text.
    Handles word boundaries and preserves case where appropriate.
    """
    if not text:
        return ""

    # Sort abbreviation keys by length descending to match longer ones first (e.g., T2DM before DM)
    sorted_abbrs = sorted(MEDICAL_ABBREVIATIONS.keys(), key=len, reverse=True)
    
    result_text = text
    
    for abbr in sorted_abbrs:
        # Create regex for standalone word
        pattern = re.compile(rf'\b({re.escape(abbr)})\b', re.IGNORECASE)
        
        matches = list(pattern.finditer(result_text))
        # Process from end to start to maintain index validity
        for match in reversed(matches):
            matched_text = match.group(0)
            start, end = match.span()
            
            # Determine expansion
            expansion = None
            if context_aware and abbr in CONTEXT_DEPENDENT_ABBREV:
                # Look at surrounding context (100 chars before and after)
                ctx_start = max(0, start - 100)
                ctx_end = min(len(result_text), end + 100)
                context_snippet = result_text[ctx_start:ctx_end].lower()
                
                expansion = CONTEXT_DEPENDENT_ABBREV[abbr]["default"]
                for trigger, replacement in CONTEXT_DEPENDENT_ABBREV[abbr].items():
                    if trigger == "default":
                        continue
                    if re.search(trigger, context_snippet):
                        expansion = replacement
                        break
            else:
                expansion = MEDICAL_ABBREVIATIONS[abbr]

            # Match case of original
            if matched_text.isupper():
                final_expansion = f"{expansion} ({matched_text})"
            elif matched_text[0].isupper():
                # Title case or Pt
                final_expansion = f"{expansion} ({matched_text})"
            else:
                # lowercase
                final_expansion = f"{expansion.lower()} ({matched_text})"

            result_text = result_text[:start] + final_expansion + result_text[end:]

    return result_text

def detect_abbreviations(text: str) -> List[Dict[str, Any]]:
    """
    Detect all abbreviations in text and return metadata.
    """
    if not text:
        return []

    detected = []
    sorted_abbrs = sorted(MEDICAL_ABBREVIATIONS.keys(), key=len, reverse=True)
    
    # We iterate over unique occurrences found in text to avoid duplications in results
    for abbr in sorted_abbrs:
        pattern = re.compile(rf'\b({re.escape(abbr)})\b', re.IGNORECASE)
        for match in pattern.finditer(text):
            start, end = match.span()
            matched_text = match.group(0)
            
            # Simple context window for display
            ctx_start = max(0, start - 20)
            ctx_end = min(len(text), end + 20)
            context_window = text[ctx_start:ctx_end].replace('\n', ' ')

            # Basic expansion for detection info
            expansion = MEDICAL_ABBREVIATIONS[abbr]
            if abbr in CONTEXT_DEPENDENT_ABBREV:
                 # Re-run limited context check for detection
                ctx_wide_start = max(0, start - 100)
                ctx_wide_end = min(len(text), end + 100)
                context_snippet = text[ctx_wide_start:ctx_wide_end].lower()
                
                expansion = CONTEXT_DEPENDENT_ABBREV[abbr]["default"]
                for trigger, replacement in CONTEXT_DEPENDENT_ABBREV[abbr].items():
                    if trigger == "default":
                        continue
                    if re.search(trigger, context_snippet):
                        expansion = replacement
                        break

            detected.append({
                "abbreviation": matched_text,
                "position": start,
                "end": end,
                "expansion": expansion,
                "confidence": 0.95, # High confidence for dictionary matches
                "context": f"...{context_window}..."
            })
            
    # Sort by position
    detected.sort(key=lambda x: x["position"])
    return detected
