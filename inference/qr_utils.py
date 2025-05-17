import io
import re
import zipfile
from typing import Dict, List, Tuple, Union

import filetype
import numpy as np
import qreader
import requests
import tldextract
from PIL import Image
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes


class QRDocumentProcessor:
    """
    Class for handling QR code detection and document processing.
    Provides utilities for analyzing documents with QR codes and determining
    if they're potentially forged based on domain information.
    """
    
    def __init__(self, api_key: str = "pWlQYx9XMaP3400OhIdJwququbOeel63", trusted_domains: List[str] = ["gov.in"]):
        """
        Initialize the QR code processor with API key and trusted domains.
        
        Args:
            api_key: API key for the domain information service
            trusted_domains: List of domains considered trusted/legitimate
        """
        self.qr_reader = qreader.QReader(model_size='l')
        self.api_key = api_key
        self.trusted_domains = trusted_domains
    
    def detect_qr_code(self, image: Image.Image) -> Tuple[bool, Union[Dict, str, None]]:
        """
        Detect and decode QR codes in an image.
        
        Args:
            image: PIL Image to scan for QR codes
            
        Returns:
            Tuple of (found_qr_code, qr_data)
        """
        img_array = np.array(image)
        qr_codes = self.qr_reader.detect_and_decode(image=img_array)
        qr_codes = [code for code in qr_codes if code is not None]
        
        if not qr_codes:
            return False, None
        
        qr_data = qr_codes[0].lower()
        
        # Handle URLs by getting domain info
        if re.match(r'http[s]?://', qr_data):
            return True, self.get_domain_info(qr_data)
        
        return True, qr_data
    
    def get_domain_info(self, url: str) -> Dict:
        """
        Get domain information using the API Layer service.
        
        Args:
            url: URL to extract domain from and analyze
            
        Returns:
            Dictionary with domain information
        """
        main_domain = self._extract_main_domain(url)
        api_url = f"https://api.apilayer.com/whois/query?domain={main_domain}"
        
        headers = {"apikey": self.api_key}
        
        try:
            response = requests.get(api_url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json()
            return {"error": f"API error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}
    
    def _extract_main_domain(self, url: str) -> str:
        """Extract the main domain from a URL."""
        ext = tldextract.extract(url)
        return f"{ext.domain}.{ext.suffix}"
    
    def analyze_forgery(self, qr_data: Dict) -> Tuple[str, List]:
        """
        Analyze QR data to determine if document might be forged.
        
        Args:
            qr_data: QR data with domain information
            
        Returns:
            Tuple of (forgery_status, qr_values)
        """
        try:
            if qr_data and isinstance(qr_data, dict) and 'result' in qr_data and 'domain_name' in qr_data['result']:
                domain = qr_data['result']["domain_name"]
                if isinstance(domain, list):
                    domain = domain[0]
                
                # Check if domain is in trusted domains list
                is_trusted = any(trusted in domain for trusted in self.trusted_domains)
                forgery_status = "not forged" if is_trusted else "forged"
                return forgery_status, [qr_data]
        except Exception:
            pass
            
        return "not applicable", [qr_data] if qr_data else []
    
    def process_pdf_page(self, image: Image.Image, page_num: int) -> Dict:
        """
        Process a single PDF page image.
        
        Args:
            image: PIL Image of the PDF page
            page_num: Page number
            
        Returns:
            Dictionary with page analysis results
        """
        qr_found, qr_data = self.detect_qr_code(image)
        forgery_status = "not applicable"
        qr_values = []
        
        if qr_found:
            forgery_status, qr_values = self.analyze_forgery(qr_data)
            
        return {
            "page_number": page_num,
            "qr_codes": qr_values,
            "forged_status": forgery_status
        }
    
    def process_image(self, image: Image.Image) -> Dict:
        """
        Process a single image for QR codes.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dictionary with QR code analysis results
        """
        qr_found, qr_data = self.detect_qr_code(image)
        forgery_status = "not applicable" 
        qr_values = []
        
        if qr_found:
            forgery_status, qr_values = self.analyze_forgery(qr_data)
            
        return {
            "qr_codes": qr_values,
            "forged_status": forgery_status
        }


# Create a singleton instance for module-level access
qr_processor = QRDocumentProcessor()


def detect_file_type(file_bytes: bytes) -> str:
    """
    Detect file type from bytes content.
    
    Args:
        file_bytes: Raw file bytes
        
    Returns:
        MIME type string
    """
    kind = filetype.guess(file_bytes)
    if kind is None:
        raise ValueError("Could not determine file type")
    return kind.mime


def process_pdf_file(file_bytes: bytes, filename: str) -> Dict:
    """
    Process a PDF file for QR codes on each page.
    
    Args:
        file_bytes: Raw PDF file content
        filename: Name of the file
        
    Returns:
        Dictionary with processing results
    """
    pdf_reader = PdfReader(io.BytesIO(file_bytes))
    pdf_images = convert_from_bytes(file_bytes, dpi=300)
    pages_content = []

    for idx, (page, image) in enumerate(zip(pdf_reader.pages, pdf_images), start=1):
        page_result = qr_processor.process_pdf_page(image, idx)
        pages_content.append(page_result)

    return {
        "file_name": filename,
        "file_type": "pdf",
        "pages": pages_content
    }


def process_image_file(file_bytes: bytes, filename: str) -> Dict:
    """
    Process an image file for QR codes.
    Supports both single images and multi-page TIFF files.
    
    Args:
        file_bytes: Raw image file content
        filename: Name of the file
        
    Returns:
        Dictionary with processing results
    """
    image = Image.open(io.BytesIO(file_bytes))
    
    # Handle multi-page TIFF files
    if image.format == "TIFF" and hasattr(image, "n_frames") and image.n_frames > 1:
        pages_content = []
        
        for idx in range(image.n_frames):
            image.seek(idx)
            # Convert to RGB to ensure consistency
            image_frame = image.convert('RGB')
            page_result = qr_processor.process_pdf_page(image_frame, idx + 1)
            pages_content.append(page_result)
            
        return {
            "file_name": filename,
            "file_type": "tiff",
            "pages": pages_content
        }
    
    # Handle single-page images
    else:
        result = qr_processor.process_image(image)
        return {
            "file_name": filename,
            "file_type": "image",
            **result
        }


def process_zip_file(file_bytes: bytes) -> List[Dict]:
    """
    Process a ZIP archive containing PDFs and/or images.
    
    Args:
        file_bytes: Raw ZIP file content
        
    Returns:
        List of dictionaries with processing results for each file
    """
    results = []
    
    with zipfile.ZipFile(io.BytesIO(file_bytes), 'r') as zip_file:
        # Filter out directories and other non-file entries
        extracted_files = [f for f in zip_file.namelist() if "." in f and not f.startswith("__MACOSX")]

        for extracted_file_name in extracted_files:
            try:
                with zip_file.open(extracted_file_name) as extracted_file:
                    extracted_file_bytes = extracted_file.read()
                    
                    try:
                        extracted_mime_type = detect_file_type(extracted_file_bytes)
                    except ValueError:
                        # Skip files with unknown types
                        continue
                    
                    # Process based on file type
                    if extracted_mime_type == "application/pdf":
                        result = process_pdf_file(extracted_file_bytes, extracted_file_name)
                        results.append(result)
                    
                    elif extracted_mime_type.startswith("image/"):
                        result = process_image_file(extracted_file_bytes, extracted_file_name)
                        results.append(result)
            except Exception as e:
                # Add error info for this file but continue processing others
                results.append({
                    "file_name": extracted_file_name,
                    "file_type": "error",
                    "error": str(e)
                })
    
    return results