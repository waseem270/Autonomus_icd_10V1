import logging
import asyncio
import re
import pdfplumber
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from typing import Dict, List, Any
from ..core.config import settings
from ..utils.pdf_detector import detect_pdf_type

from ..utils.text_preprocessor import preprocess_medical_text, remove_noise, normalize_whitespace

logger = logging.getLogger(__name__)

# Configure Tesseract path if provided
if settings.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD

class TextExtractionService:
    """
    Service to handle text extraction from both digital and scanned PDFs.
    Uses pdfplumber for digital and Tesseract OCR for scanned content.
    """

    def __init__(self):
        self.ocr_language = settings.OCR_LANGUAGE
        self.ocr_config = settings.OCR_CONFIG
        self.poppler_path = settings.PDF2IMAGE_POPPLLER_PATH if settings.PDF2IMAGE_POPPLLER_PATH else None

    async def extract_text(self, pdf_path: str) -> Dict[str, Any]:
        """
        Main entry point for text extraction. Detects PDF type and routes appropriately.
        """
        logger.info(f"Starting text extraction for: {pdf_path}")
        
        try:
            # 1. Detect PDF type
            detection_result = detect_pdf_type(pdf_path)
            method = "digital" if detection_result["type"] == "digital" else "ocr"
            
            # 2. Extract based on type
            if method == "digital":
                result = await self.extract_from_digital_pdf(pdf_path)
            else:
                result = await self.extract_from_scanned_pdf(pdf_path)
            
            # 3. Light post-processing: remove noise but preserve line breaks so
            #    the downstream section detector can use structural headers.
            #    fix_line_breaks() is applied later inside text_structuring AFTER
            #    section detection has run.
            cleaned_text = remove_noise(result["raw_text"])
            cleaned_text = normalize_whitespace(cleaned_text)
            result["raw_text"] = cleaned_text
            
            # Update individual pages text as well
            for page in result["pages"]:
                page["text"] = normalize_whitespace(remove_noise(page["text"]))
            
            # 4. Calculate quality score
            result["quality_score"] = self.calculate_quality_score(cleaned_text, method)
            
            logger.info(f"Extraction complete for {pdf_path}. Method: {method}, Quality: {result['quality_score']:.2f}")
            return result

        except Exception as e:
            logger.error(f"Text extraction failed for {pdf_path}: {str(e)}", exc_info=True)
            raise

    async def extract_from_digital_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from a digital (text-based) PDF.
        """
        logger.info(f"Using digital extraction for: {pdf_path}")
        pages_content = []
        full_text = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Extract text preserving layout as much as possible
                    text = page.extract_text(layout=True) or ""
                    pages_content.append({
                        "page_num": i + 1,
                        "text": text
                    })
                    full_text.append(text)
                
                return {
                    "raw_text": "\n\n".join(full_text),
                    "page_count": len(pdf.pages),
                    "extraction_method": "digital",
                    "confidence_score": 1.0, 
                    "pages": pages_content
                }
        except Exception as e:
            logger.error(f"Digital extraction failed: {str(e)}")
            raise

    async def extract_from_scanned_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from a scanned PDF using OCR.
        """
        logger.info(f"Using OCR extraction for: {pdf_path}")
        pages_content = []
        full_text = []
        
        try:
            images = await asyncio.to_thread(
                convert_from_path, 
                pdf_path, 
                poppler_path=self.poppler_path
            )
            
            for i, image in enumerate(images):
                text = await asyncio.to_thread(
                    pytesseract.image_to_string,
                    image,
                    lang=self.ocr_language,
                    config=self.ocr_config
                )
                
                pages_content.append({
                    "page_num": i + 1,
                    "text": text,
                    "ocr_confidence": 0.85 
                })
                full_text.append(text)

            return {
                "raw_text": "\n\n".join(full_text),
                "page_count": len(images),
                "extraction_method": "ocr",
                "confidence_score": 0.85,
                "pages": pages_content
            }
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            raise

    def calculate_quality_score(self, text: str, extraction_method: str) -> float:
        """
        Calculate a quality score (0.0 to 1.0) for the extracted text.
        """
        if not text:
            return 0.0
            
        score = 1.0 if extraction_method == "digital" else 0.8
        
        # Penalty for too many special characters (likely garbage OCR)
        special_chars = len(re.findall(r'[^a-zA-Z0-9\s,\.\(\)\[\]\-\/]', text))
        total_chars = len(text)
        if total_chars > 0:
            special_ratio = special_chars / total_chars
            if special_ratio > 0.1:
                score -= 0.3
            elif special_ratio > 0.05:
                score -= 0.1
        
        # Bonus for medical keywords
        medical_keywords = ['patient', 'diagnosis', 'clinical', 'history', 'assessment', 'treatment', 'medication']
        found_keywords = sum(1 for kw in medical_keywords if kw.lower() in text.lower())
        if found_keywords >= 3:
            score += 0.1
            
        return max(0.0, min(1.0, score))


# Singleton instance
text_extraction_service = TextExtractionService()
