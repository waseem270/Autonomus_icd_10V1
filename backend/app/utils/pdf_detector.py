import logging
import pdfplumber
from pathlib import Path
from typing import Dict, Any

# Configure logger
logger = logging.getLogger(__name__)

def detect_pdf_type(pdf_path: str) -> Dict[str, Any]:
    """
    Detect if a PDF is digital, scanned, or hybrid by analyzing extractable text.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        dict: Metadata containing type, page_count, and extraction stats.
    """
    path = Path(pdf_path)
    if not path.exists():
        logger.error(f"File not found: {pdf_path}")
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    results = {
        "type": "unknown",
        "page_count": 0,
        "extractable_chars": 0,
        "avg_chars_per_page": 0.0,
        "confidence": 0.0,
        "needs_ocr": False
    }

    try:
        with pdfplumber.open(pdf_path) as pdf:
            results["page_count"] = len(pdf.pages)
            
            # Look at first 3 pages to determine type
            check_pages = pdf.pages[:3]
            page_stats = []
            total_chars = 0

            for page in check_pages:
                text = page.extract_text() or ""
                char_count = len(text.strip())
                page_stats.append(char_count)
                total_chars += char_count

            num_checked = len(check_pages)
            if num_checked > 0:
                avg_chars = total_chars / num_checked
                results["extractable_chars"] = total_chars
                results["avg_chars_per_page"] = avg_chars
                
                # Logic to determine type
                # 1. All pages have significant text -> Digital
                # 2. All pages have very little/no text -> Scanned
                # 3. Some have text, some don't -> Hybrid
                
                digital_pages = sum(1 for c in page_stats if c > 100)
                scanned_pages = sum(1 for c in page_stats if c <= 100)

                if digital_pages == num_checked:
                    results["type"] = "digital"
                    results["confidence"] = 0.95
                    results["needs_ocr"] = False
                elif scanned_pages == num_checked:
                    results["type"] = "scanned"
                    results["confidence"] = 0.90
                    results["needs_ocr"] = True
                else:
                    results["type"] = "hybrid"
                    results["confidence"] = 0.70
                    results["needs_ocr"] = True

                logger.info(f"PDF Detection: {path.name} is {results['type']} (Avg chars: {avg_chars:.2f})")
            
            return results

    except pdfplumber.pdfparser.PDFSyntaxError:
        logger.error(f"Malformed PDF: {pdf_path}")
        raise ValueError(f"The file {pdf_path} is a malformed or corrupted PDF.")
    except Exception as e:
        # Check for password protection/permission error
        if "password" in str(e).lower() or "decryption" in str(e).lower():
            logger.error(f"Encrypted PDF: {pdf_path}")
            raise PermissionError(f"The PDF {pdf_path} is password protected or encrypted.")
        
        logger.exception(f"Unexpected error processing {pdf_path}")
        raise

if __name__ == "__main__":
    # Quick test if run directly
    import sys
    if len(sys.argv) > 1:
        try:
            print(detect_pdf_type(sys.argv[1]))
        except Exception as e:
            print(f"Error: {e}")
