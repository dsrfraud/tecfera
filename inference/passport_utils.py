import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import the custom modules (adjust imports as needed for your project structure)
from inference import custom_ocr, passport_mrz_extraction, genai_validation, mrz_validation


class PassportProcessor:
    """
    Class for handling passport image processing, data extraction, and validation.
    """
    
    def __init__(self):
        """Initialize the passport processor with necessary components."""
        # Component instances will be initialized on first use
        self._ocr = None
        self._mrz_extractor = None
        self._genai_validator = None
        self._mrz_validator = None
    
    @property
    def ocr(self):
        """Lazy-load OCR component."""
        if self._ocr is None:
            self._ocr = custom_ocr
        return self._ocr
    
    @property
    def mrz_extractor(self):
        """Lazy-load MRZ extractor component."""
        if self._mrz_extractor is None:
            self._mrz_extractor = passport_mrz_extraction
        return self._mrz_extractor
    
    @property
    def genai_validator(self):
        """Lazy-load GenAI validator component."""
        if self._genai_validator is None:
            self._genai_validator = genai_validation
        return self._genai_validator
    
    @property
    def mrz_validator(self):
        """Lazy-load MRZ validator component."""
        if self._mrz_validator is None:
            self._mrz_validator = mrz_validation
        return self._mrz_validator
    
    def extract_text_from_image(self, image: np.ndarray) -> str:
        """
        Extract text from an image using OCR.
        
        Args:
            image: OpenCV image array
            
        Returns:
            Extracted text as a string
        """
        ocr_result = self.ocr.ocr.ocr(image)
        if ocr_result:
            return self.ocr.get_text_data(ocr_result)
        return ""
    
    def extract_mrz_data(self, image: np.ndarray) -> str:
        """
        Extract MRZ (Machine Readable Zone) data from passport image.
        
        Args:
            image: OpenCV image array
            
        Returns:
            Extracted MRZ text or empty string if not found
        """
        mrz_roi = self.mrz_extractor.get_mrz_area(image)
        if mrz_roi is not None and mrz_roi.size > 0:
            mrz_ocr_result = self.ocr.ocr.ocr(mrz_roi)
            return self.mrz_extractor.get_text_data(mrz_ocr_result)
        return ""
    
    def process_passport_front(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process front side of passport image.
        
        Args:
            image: OpenCV image array of passport front
            
        Returns:
            Dictionary with extracted passport data and validation status
        """
        # Extract text using OCR
        ocr_text = self.extract_text_from_image(image)        
        # Extract MRZ data
        mrz_text = self.extract_mrz_data(image)
        
        # Process extracted data using GenAI
        passport_data_json = self.genai_validator.extract_passport_front_data(ocr_text, mrz_text)
        
        try:
            # Convert to proper JSON
            passport_data = json.loads(passport_data_json)
            
            # Validate the extracted data
            validation_result = self.mrz_validator.verify_passport_details(passport_data)
            passport_data["match_status"] = validation_result
            
            return passport_data
        except json.JSONDecodeError:
            # Handle case where GenAI returns malformed JSON
            return {
                "error": "Failed to parse passport data",
                "raw_text": ocr_text[:200],  # Include a sample of raw text for debugging
                "status": "error"
            }
    
    def process_passport_back(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process back side of passport image.
        
        Args:
            image: OpenCV image array of passport back
            
        Returns:
            Dictionary with extracted passport data
        """
        # Extract text using OCR
        ocr_text = self.extract_text_from_image(image)
        
        # Process extracted data using GenAI
        passport_data_json = self.genai_validator.extract_passport_back_data(ocr_text)
        
        try:
            # Convert to proper JSON
            passport_data = json.loads(passport_data_json)
            return passport_data
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse passport data",
                "raw_text": ocr_text[:200],  # Include a sample of raw text for debugging
                "status": "error"
            }


# Create a singleton instance
passport_processor = PassportProcessor()


def extract_passport_data(
    image: np.ndarray, 
    is_front: bool = True, 
    include_validation: bool = True
) -> Dict[str, Any]:
    """
    Extract data from passport image (front or back).
    
    Args:
        image: OpenCV image array
        is_front: Whether this is the front side of passport
        include_validation: Whether to include validation data
        
    Returns:
        Dictionary with extracted passport data
    """
    if is_front:
        return passport_processor.process_passport_front(image)
    else:
        return passport_processor.process_passport_back(image)


def make_api_request(url: str, params: Dict = None, headers: Dict = None, 
                    method: str = "POST", timeout: int = 30, delay: int = 1, 
                    retries: int = 3) -> Optional[requests.Response]:
    """
    Make API request with retry logic.
    
    Args:
        url: API endpoint URL
        params: Request parameters
        headers: Request headers
        method: HTTP method (GET, POST, etc.)
        timeout: Request timeout in seconds
        delay: Delay between retries in seconds
        retries: Number of retry attempts
        
    Returns:
        Response object or None if all attempts fail
    """
    # Set up retry strategy
    retry_strategy = Retry(
        total=retries,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        backoff_factor=delay
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    
    with requests.Session() as session:
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        # Default headers if none provided
        if headers is None:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Content-Type": "application/x-www-form-urlencoded"
            }
        
        try:
            if method.upper() == "GET":
                response = session.get(url, params=params, headers=headers, timeout=timeout)
            else:  # POST
                response = session.post(url, data=params, headers=headers, timeout=timeout)
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {str(e)}")
            return None


def get_parsed_output(response: requests.Response) -> Dict[str, Any]:
    """
    Parse HTML response from passport status API.
    
    Args:
        response: Response object from API request
        
    Returns:
        Dictionary with parsed data
    """
    try:
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract status information (adjust selectors based on actual HTML structure)
        status_table = soup.find('table', class_='statusTable')
        
        if not status_table:
            return {"status": "Not found", "message": "No status information available"}
        
        # Parse table rows into dictionary
        result = {}
        rows = status_table.find_all('tr')
        
        for row in rows:
            columns = row.find_all('td')
            if len(columns) >= 2:
                key = columns[0].text.strip().replace(':', '')
                value = columns[1].text.strip()
                result[key] = value
        
        return result
    
    except Exception as e:
        return {"status": "error", "message": f"Failed to parse response: {str(e)}"}


def validate_passport_details(passport_data: Dict[str, Any]) -> Dict[str, bool]:
    """
    Wrapper for passport validation function.
    
    Args:
        passport_data: Dictionary with passport data
        
    Returns:
        Dictionary with validation results
    """
    return passport_processor.mrz_validator.verify_passport_details(passport_data)