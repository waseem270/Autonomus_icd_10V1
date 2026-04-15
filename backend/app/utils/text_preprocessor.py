import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def segment_sentences(text: str) -> List[Dict[str, Any]]:
    """
    Segment text into sentences with metadata (Regex-based).
    """
    if not text:
        return []
    
    # Simple regex for sentence splitting
    # Splitting by common sentence endings followed by whitespace and capital letter
    # This is a fallback since spaCy is not available
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    
    sentences = []
    current_pos = 0
    for i, sent_text in enumerate(raw_sentences):
        clean_sent = sent_text.strip()
        if not clean_sent:
            continue
            
        start_idx = text.find(clean_sent, current_pos)
        if start_idx == -1:
            start_idx = current_pos
            
        sentences.append({
            "sentence_number": i + 1,
            "text": clean_sent,
            "start_char": start_idx,
            "end_char": start_idx + len(clean_sent),
            "length": len(clean_sent)
        })
        current_pos = start_idx + len(clean_sent)
        
    return sentences

def remove_noise(text: str) -> str:
    """
    Remove common noise from medical documents like page numbers, footers, etc.
    """
    if not text:
        return ""

    # 1. Remove Page numbers (e.g., Page 1, Page 1 of 10, 1 of 5)
    text = re.sub(r'(?i)Page \d+(?: of \d+)?', '', text)
    text = re.sub(r'\b\d+ of \d+\b', '', text)

    # 2. Remove common header/footer junk phrases
    noise_patterns = [
        r'Confidential Property of.*',
        r'Electronic Health Record.*',
        r'Printed on: \d{1,2}/\d{1,2}/\d{2,4}.*',
        r'DO NOT DUPLICATE',
        r'All Rights Reserved'
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # 3. Clean headers with dates (e.g., 10/25/2023 10:30 AM)
    text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(?:\s*[APM]{2})?\b', '', text)

    # 4. Remove standalone special characters that aren't medical symbols
    # Keep: °%/-+. (important for vitals/vols)
    text = re.sub(r'(?<![0-9])[#$*=_<>\|](?![0-9])', ' ', text)

    return text

def normalize_whitespace(text: str) -> str:
    """
    Normalize spaces and newlines.
    """
    if not text:
        return ""

    # Multiple spaces -> single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove leading/trailing whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    
    # Multiple newlines -> double newline (preserve paragraphs)
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def fix_line_breaks(text: str) -> str:
    """
    Fix broken lines from PDF extraction (hyphenation and punctuation rules).
    """
    if not text:
        return ""

    # 1. Join hyphenated words at line breaks
    # Example: "dia-\nbetes" -> "diabetes"
    text = re.sub(r'(\w+)-\n\s*(\w+)', r'\1\2', text)

    # 2. Join lines that don't end with sentence-ending punctuation 
    # but the next line starts with a lowercase letter (likely a continuation)
    lines = text.split('\n')
    fixed_lines = []
    
    i = 0
    while i < len(lines):
        current_line = lines[i].strip()
        if not current_line:
            fixed_lines.append("")
            i += 1
            continue
            
        # Check if we should join with the next line
        if i + 1 < len(lines):
            next_line = lines[i+1].strip()
            if next_line:
                # If current line doesn't end in [.!?:] and next line starts with lowercase
                if not re.search(r'[.!?:]$', current_line) and re.match(r'[a-z]', next_line):
                    current_line = current_line + " " + next_line
                    i += 1 # Skip next line
        
        fixed_lines.append(current_line)
        i += 1

    return '\n'.join(fixed_lines)

def preprocess_medical_text(text: str) -> str:
    """
    Complete preprocessing pipeline for medical text.
    """
    text = remove_noise(text)
    text = fix_line_breaks(text)
    text = normalize_whitespace(text)
    return text
