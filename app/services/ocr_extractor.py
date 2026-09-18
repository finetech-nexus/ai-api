"""
PaddleOCR Extractor - PRODUCTION REWRITE
Enterprise-grade context-aware extraction with validation
Author: Senior OCR Developer
Version: 2.0
"""

from paddleocr import PaddleOCR
import re
import cv2
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
from datetime import datetime

from configs.config import config
from utils.logger import get_logger

logger = get_logger(__name__, log_file="ocr_extractor.log")


class DocumentType(str, Enum):
    """Supported ID document types."""
    DRIVERS_LICENSE = "drivers_license"
    NATIONAL_ID = "national_id"
    PASSPORT = "passport"
    RESIDENCE_PERMIT = "residence_permit"
    VOTER_ID = "voter_id"
    ID_CARD = "id_card"
    PAN_CARD = "pan_card"
    UNKNOWN = "unknown"


@dataclass
class MRZData:
    """Machine Readable Zone (MRZ) parsed data."""
    document_type: Optional[str] = None
    document_code: Optional[str] = None
    issuing_country: Optional[str] = None
    surname: Optional[str] = None
    given_names: Optional[str] = None
    passport_number: Optional[str] = None
    nationality: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    expiry_date: Optional[str] = None
    personal_number: Optional[str] = None
    check_digit: Optional[str] = None
    valid: bool = False


@dataclass
class OCRResult:
    """Comprehensive structured OCR result for API response."""
    document_type: str
    confidence: float
    
    # Core Personal Information
    full_name: Optional[str] = None
    surname: Optional[str] = None
    given_names: Optional[str] = None
    date_of_birth: Optional[str] = None
    nationality: Optional[str] = None
    
    # Document Information
    document_number: Optional[str] = None
    document_code: Optional[str] = None
    issuing_country: Optional[str] = None
    issuing_authority: Optional[str] = None
    
    # Dates
    issue_date: Optional[str] = None
    expiry_date: Optional[str] = None
    
    # Personal Details
    place_of_birth: Optional[str] = None
    address: Optional[str] = None
    gender: Optional[str] = None
    height: Optional[str] = None
    eye_color: Optional[str] = None
    
    # Additional Fields
    curp: Optional[str] = None
    registro: Optional[str] = None
    localidad: Optional[str] = None
    
    # MRZ Data
    mrz_data: Optional[Dict[str, Any]] = None
    
    # Metadata
    extracted_text: str = ""
    field_confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response, omit None values."""
        result = {}
        for k, v in asdict(self).items():
            if v is not None and v != {} and v != "":
                result[k] = v
        return result


class MRZParser:
    """Parse Machine Readable Zone from passports and IDs."""
    
    COUNTRY_CODES = {
        "CHE": "Switzerland", "DEU": "Germany", "ESP": "Spain", "PRT": "Portugal",
        "FRA": "France", "ITA": "Italy", "AUT": "Austria", "NLD": "Netherlands",
        "BEL": "Belgium", "LUX": "Luxembourg", "POL": "Poland", "USA": "United States",
        "CAN": "Canada", "GBR": "United Kingdom", "IRL": "Ireland", "MEX": "Mexico",
    }
    
    @staticmethod
    def clean_mrz_line(line: str) -> str:
        """Clean and normalize MRZ line."""
        line = line.upper().replace(" ", "")
        line = line.replace("O", "0").replace("I", "1").replace("S", "5")
        return line
    
    @staticmethod
    def extract_mrz_dates(date_str: str) -> Optional[str]:
        """Convert MRZ date format (YYMMDD) to DD.MM.YYYY."""
        if len(date_str) != 6 or not date_str.isdigit():
            return None
        
        try:
            yy, mm, dd = int(date_str[:2]), int(date_str[2:4]), int(date_str[4:6])
            yyyy = 2000 + yy if yy <= 30 else 1900 + yy
            
            if not (1 <= mm <= 12) or not (1 <= dd <= 31):
                return None
            
            datetime(yyyy, mm, dd)
            return f"{dd:02d}.{mm:02d}.{yyyy}"
        except (ValueError, TypeError):
            return None
    
    @classmethod
    def parse_passport_mrz(cls, mrz_lines: List[str]) -> MRZData:
        """Parse 2-line passport MRZ format (TD-3) or 3-line ID format (TD-1)."""
        if len(mrz_lines) < 2:
            return MRZData()
        
        line1 = cls.clean_mrz_line(mrz_lines[0])
        line2 = cls.clean_mrz_line(mrz_lines[1])
        
        data = MRZData()
        
        try:
            if len(line1) >= 44:
                data.document_type = line1[0]
                data.document_code = line1[0:2]
                data.issuing_country = line1[2:5].strip("<")
                
                name_section = line1[5:44].rstrip("<")
                if "<<" in name_section:
                    parts = name_section.split("<<", 1)
                    data.surname = parts[0].replace("<", " ").strip()
                    data.given_names = parts[1].replace("<", " ").strip() if len(parts) > 1 else ""
            
            if len(line2) >= 44:
                data.passport_number = line2[:9].rstrip("<")
                data.nationality = line2[10:13].strip("<")
                data.date_of_birth = cls.extract_mrz_dates(line2[13:19])
                data.gender = line2[20] if line2[20] in ["M", "F"] else None
                data.expiry_date = cls.extract_mrz_dates(line2[21:27])
                data.valid = True
        
        except (IndexError, ValueError) as e:
            logger.debug(f"MRZ parse error: {e}")
            data.valid = False
        
        return data


class ContextualExtractor:
    """Context-aware field extraction using spatial and semantic analysis."""
    
    @staticmethod
    def is_header_text(text: str) -> bool:
        """Check if text is a document header/label."""
        headers = {
            "passport", "passeporte", "pasaporte", "passeport", "reisepass",
            "identity", "identidad", "identidade", "documento", "document",
            "national", "nacional", "republic", "republica", "kingdom", "reino",
            "spain", "españa", "espana", "portugal", "germany", "deutschland",
            "switzerland", "schweiz", "suisse", "card", "carte", "karte",
            "driving", "license", "licencia", "führerschein"
        }
        return any(h in text.lower() for h in headers)
    
    @staticmethod
    def is_field_label(text: str) -> bool:
        """Check if text is a field label."""
        labels = {
            "nombre", "name", "nom", "nome", "nachname", "surname", "apellido",
            "numero", "number", "nummer", "numéro", "soporte", "support",
            "fecha", "date", "datum", "data", "sexo", "sex", "gender",
            "nacionalidad", "nationality", "ciudadano", "citizen",
            "valido", "valid", "expiry", "caducidad", "expedición"
        }
        return any(label in text.lower() for label in labels)
    
    @staticmethod
    def extract_name_from_detections(detections: List[Dict], country_hint: str = None) -> Optional[str]:
        """Extract name using multi-strategy approach."""
        # Strategy 0: Look for specific passport name patterns like "Schweizer Sample"
        for idx, det in enumerate(detections):
            text = det["text"].strip()
            # Pattern for passport names (Title Case with dot)
            if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+\.$', text):
                name = text.rstrip('.')
                logger.debug(f"Found passport name pattern: '{name}'")
                return name
        
        # Strategy 1: Look for "Name:" label pattern (highest priority)
        for idx, det in enumerate(detections):
            text = det["text"].strip()
            # Skip signature fields
            if "signature" in text.lower() or ("sign" in text.lower() and len(text) < 20):
                continue
            # Pattern: "Name: AMAN SIGROHA" or "Name AMAN SIGROHA" or "41H/Name FIRST NAME MIDDLE NAME"
            name_match = re.search(r'(?:^|\s)(?:Name|NAME|Nombre|NOME|Surname)[:\s×/]+([A-Z][A-Z\s]{2,40})', text, re.IGNORECASE)
            if name_match:
                name = name_match.group(1).strip()
                # Skip placeholder text like "FIRST NAME MIDDLE NAME SURNAME"
                if any(placeholder in name.upper() for placeholder in 
                       ["FIRST NAME", "MIDDLE NAME", "SURNAME", "PLACEHOLDER", "SAMPLE", "EXAMPLE"]):
                    continue
                # Skip signature fields
                if "signature" in name.lower() or "sign" in name.lower():
                    continue
                # Clean up: remove extra spaces, validate it's a name
                name = re.sub(r'\s+', ' ', name)
                if len(name) >= 3 and len(name.split()) <= 5:
                    # Exclude common false positives
                    if not any(word.lower() in ['department', 'transport', 'issued', 'by', 'licence', 'license', 'signature', 'sign',
                                                 'govt', 'government', 'india', 'income', 'tax', 'permanent', 'account', 'number', 'card'] 
                              for word in name.split()):
                        # Skip government/institutional phrases
                        if not any(phrase in name.upper() for phrase in ['GOVT OF', 'GOVERNMENT OF', 'INCOME TAX']):
                            logger.debug(f"Found name via label pattern: '{name}'")
                            return name
        
        # Strategy 2: Look for name after "Name:" in next detection
        for idx in range(len(detections) - 1):
            text = detections[idx]["text"].strip()
            # Skip if current text is signature field
            if "signature" in text.lower():
                continue
            # PAN card pattern: "41H/Name" or "Name / नाम" or just "Name"
            if re.search(r'(?:^|\s)(?:Name|NAME|Nombre|NOME|नाम)[:\s/]*$', text, re.IGNORECASE) or \
               re.search(r'[0-9A-Z]+/Name', text, re.IGNORECASE):
                # Next detection might be the name
                next_text = detections[idx + 1]["text"].strip()
                # Skip placeholder text like "FIRST NAME MIDDLE NAME SURNAME"
                if any(placeholder in next_text.upper() for placeholder in 
                       ["FIRST NAME", "MIDDLE NAME", "SURNAME", "PLACEHOLDER", "SAMPLE", "EXAMPLE"]):
                    continue
                # Skip signature fields
                if "signature" in next_text.lower() or "sign" in next_text.lower():
                    continue
                # Skip government/institutional text
                if any(word in next_text.lower() for word in ['govt', 'government', 'india', 'income', 'tax', 'department', 'permanent', 'account']):
                    continue
                if any(phrase in next_text.upper() for phrase in ['GOVT OF', 'GOVERNMENT OF', 'INCOME TAX']):
                    continue
                if len(next_text) >= 3 and len(next_text) <= 50:
                    words = next_text.split()
                    if 2 <= len(words) <= 5:
                        alpha_ratio = sum(c.isalpha() or c.isspace() or c in "-'." for c in next_text) / len(next_text)
                        if alpha_ratio >= 0.85 and not re.search(r'\d', next_text):
                            logger.debug(f"Found name after label: '{next_text}'")
                            return next_text
        
        # Strategy 3: Fallback to original scoring method
        candidates = []
        for idx, det in enumerate(detections):
            text = det["text"].strip()
            
            # Skip headers and labels
            if ContextualExtractor.is_header_text(text) or ContextualExtractor.is_field_label(text):
                continue
            
            # Skip signature fields
            if "signature" in text.lower() or ("sign" in text.lower() and len(text) < 20):
                continue
            
            # Skip if contains department/transport keywords (common false positives)
            if any(word in text.lower() for word in ['department', 'transport', 'issued', 'licence', 'license', 
                                                       'height', 'grondezza', 'stature', 'taille', 'grosse',
                                                       'geschlecht', 'sexe', 'sesso', 'blood', 'group',
                                                       'bern be', 'authority', 'autorite', 'behorde',
                                                       'lieu', 'origin', 'place', 'helmatort', 'luogo', 'signature', 'sign',
                                                       'govt', 'government', 'india', 'income', 'tax', 'permanent', 'account',
                                                       'number', 'card', 'holders', 'card holders']):
                continue
            
            # Skip government/institutional phrases
            if any(phrase in text.upper() for phrase in ['GOVT OF', 'GOVERNMENT OF', 'INCOME TAX', 'PERMANENT ACCOUNT',
                                                          'CARD HOLDERS', 'INCOME TAX DEPARTMENT']):
                continue
            
            # Skip short or long texts
            if len(text) < 4 or len(text) > 50:
                continue
            
            # Skip if contains numbers or special symbols
            if re.search(r'\d', text) or any(c in text for c in ['|', '·', '<<', '>>']):
                continue
            
            words = text.split()
            
            # Name must be 2-5 words
            if not (2 <= len(words) <= 5):
                continue
            
            # Must be mostly alphabetic
            alpha_ratio = sum(c.isalpha() or c.isspace() or c in "-'." for c in text) / len(text)
            if alpha_ratio < 0.85:
                continue
            
            # Score based on position (names typically in middle-upper section)
            position_score = 1.0 - (abs(idx - len(detections) * 0.35) / len(detections))
            confidence = det["confidence"]
            
            # Prefer Title Case or reasonable capitalization
            capitalization_score = 0.3 if text.istitle() or (text[0].isupper() and any(c.islower() for c in text)) else 0.1
            
            total_score = (confidence * 0.5) + (position_score * 0.3) + capitalization_score
            
            candidates.append((text, total_score))
            logger.debug(f"Name candidate: '{text}' (score: {total_score:.2f})")
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None

    @staticmethod
    def extract_all_dates(detections: List[Dict]) -> List[Tuple[str, datetime, int]]:
        """Extract ALL dates with their positions."""
        dates = []
        
        for idx, det in enumerate(detections):
            text = det["text"]
            
            # Skip header lines
            if ContextualExtractor.is_header_text(text):
                continue
            
            # Pattern: DD MM YYYY
            for match in re.finditer(r'\b(\d{1,2})\s+(\d{1,2})\s+(\d{4})\b', text):
                dd, mm, yyyy = match.groups()
                try:
                    dt = datetime(int(yyyy), int(mm), int(dd))
                    formatted = f"{int(dd):02d}.{int(mm):02d}.{yyyy}"
                    dates.append((formatted, dt, idx))
                except ValueError:
                    continue
            
            # Pattern: DD.MM.YYYY or DD/MM/YYYY or DD-MM-YYYY
            for match in re.finditer(r'\b(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})\b', text):
                dd, mm, yyyy = match.groups()
                try:
                    dt = datetime(int(yyyy), int(mm), int(dd))
                    formatted = f"{int(dd):02d}.{int(mm):02d}.{yyyy}"
                    dates.append((formatted, dt, idx))
                except ValueError:
                    continue
            
            # Pattern: DD.MM.YY (2-digit year)
            for match in re.finditer(r'\b(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2})\b', text):
                dd, mm, yy = match.groups()
                yyyy = int(f"19{yy}") if int(yy) > 50 else int(f"20{yy}")
                try:
                    dt = datetime(yyyy, int(mm), int(dd))
                    formatted = f"{int(dd):02d}.{int(mm):02d}.{yyyy}"
                    dates.append((formatted, dt, idx))
                except ValueError:
                    continue
        
        return dates

    @staticmethod
    def select_date_by_type(all_dates: List[Tuple[str, datetime, int]], date_type: str) -> Optional[str]:
        """Select appropriate date based on type."""
        if not all_dates:
            return None
        
        # Sort by position (earlier in document = higher priority for DOB/Issue, later = Expiry)
        if date_type == "birth":
            # Birth dates: past, typically before 2015 (extended to include more recent births)
            valid = [(d, dt, i) for d, dt, i in all_dates if 1920 <= dt.year <= 2015]
            if valid:
                valid.sort(key=lambda x: x[2])  # Earlier position
                return valid[0][0]
        
        elif date_type == "issue":
            # Issue dates: recent past, 2000-now
            valid = [(d, dt, i) for d, dt, i in all_dates if 2000 <= dt.year <= datetime.now().year]
            if valid:
                valid.sort(key=lambda x: (abs(x[1].year - 2015), x[2]))  # Prefer ~2015

                return valid[0][0]
        
        elif date_type == "expiry":
            # Expiry dates: future or recent, 2020-2050
            valid = [(d, dt, i) for d, dt, i in all_dates if 2020 <= dt.year <= 2050]
            if valid:
                valid.sort(key=lambda x: -x[2])  # Later position
                return valid[0][0]
        
        return None


class PaddleOCRExtractor:
    """Production-grade PaddleOCR extractor with context-aware extraction."""

    def __init__(self, languages: Optional[List[str]] = None):
        if languages is None:
            config_langs = config.get("models", "ocr", "languages", default=["en"])
            lang_map = {"en": "en", "es": "es", "de": "german", "pt": "portuguese", "fr": "french"}
            languages = [lang_map.get(lang[:2], "en") for lang in config_langs]
        
        self.languages = languages
        primary_lang = languages[0] if languages else "en"
        
        logger.info(f"Initializing PaddleOCR with language: {primary_lang}")
        
        try:
            use_gpu = config.get("models", "ocr", "gpu", default=False)
            self.ocr = PaddleOCR(
                lang=primary_lang,
                device="gpu" if use_gpu else "cpu",
                use_angle_cls=True,
                det_db_thresh=0.2,
                det_db_box_thresh=0.4,
                rec_batch_num=6,
                # show_log=False
            )
            logger.info(f"✅ PaddleOCR initialized (device={'gpu' if use_gpu else 'cpu'})")
        except (TypeError, ValueError):
            self.ocr = PaddleOCR(
                lang=primary_lang,
                use_gpu=False,
                use_angle_cls=True,
                # show_log=False,
                det_db_thresh=0.2,
                det_db_box_thresh=0.4,
                rec_batch_num=6
            )
            logger.info("✅ PaddleOCR initialized (legacy mode)")
        
        self.extractor = ContextualExtractor()

    def extract_text_paddle(self, image: np.ndarray) -> Tuple[str, List[Dict[str, Any]], float]:
        """Extract text using PaddleOCR."""
        try:
            # Ensure BGR format (3 channels)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
            # PaddleOCR uses the .ocr() method
            result = self.ocr.ocr(image)
            
            if not result or not isinstance(result, list) or len(result) == 0:
                return "", [], 0.0
            
            # PaddleOCR/PaddleX returns different formats depending on version
            # Newer versions return OCRResult objects, older versions return lists
            detections = []
            rec_texts = []
            rec_scores = []
            
            # Handle OCRResult object (PaddleX format)
            if result and len(result) > 0 and hasattr(result[0], '__dict__'):
                ocr_obj = result[0]
                
                # Log available attributes to understand the structure
                logger.info(f"OCRResult attributes: {dir(ocr_obj)}")
                
                # Try different attribute names that PaddleX might use
                if hasattr(ocr_obj, 'rec_text') and hasattr(ocr_obj, 'rec_score'):
                    # Single result format
                    rec_texts = [ocr_obj.rec_text] if ocr_obj.rec_text else []
                    rec_scores = [ocr_obj.rec_score] if ocr_obj.rec_score else []
                    rec_polys = [ocr_obj.dt_poly] if hasattr(ocr_obj, 'dt_poly') else [None]
                    
                    for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
                        detections.append({
                            "text": text,
                            "confidence": score,
                            "bbox": poly
                        })
                elif hasattr(ocr_obj, 'rec_texts') and hasattr(ocr_obj, 'rec_scores'):
                    # Multiple results format
                    rec_texts = ocr_obj.rec_texts if ocr_obj.rec_texts else []
                    rec_scores = ocr_obj.rec_scores if ocr_obj.rec_scores else []
                    rec_polys = ocr_obj.dt_polys if hasattr(ocr_obj, 'dt_polys') else [None] * len(rec_texts)
                    
                    for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
                        detections.append({
                            "text": text,
                            "confidence": score,
                            "bbox": poly
                        })
                elif hasattr(ocr_obj, 'json'):
                    # JSON format - parse the dictionary
                    json_data = ocr_obj.json if callable(ocr_obj.json) else ocr_obj.json
                    if isinstance(json_data, dict):
                        logger.info(f"OCRResult JSON keys: {json_data.keys()}")
                        logger.info(f"OCRResult JSON full content: {json_data}")
                        # Extract data from the 'res' key
                        if 'res' in json_data and isinstance(json_data['res'], dict):
                            res_dict = json_data['res']
                            
                            # PaddleX format: res is a dict with rec_texts, rec_scores, rec_polys
                            if 'rec_texts' in res_dict and 'rec_scores' in res_dict:
                                rec_texts = res_dict['rec_texts'] if res_dict['rec_texts'] else []
                                rec_scores = res_dict['rec_scores'] if res_dict['rec_scores'] else []
                                rec_polys = res_dict.get('rec_polys', res_dict.get('dt_polys', [None] * len(rec_texts)))
                                
                                # Zip them together to create detections
                                for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
                                    detections.append({
                                        "text": text,
                                        "confidence": score,
                                        "bbox": poly
                                    })
            # Handle nested list structure (older PaddleOCR format)
            elif result and isinstance(result, list):
                ocr_results = result[0] if result and isinstance(result[0], list) else result
                
                for line in ocr_results:
                    if line and isinstance(line, (list, tuple)) and len(line) >= 2:
                        bbox = line[0]  # polygon coordinates
                        text_info = line[1]  # (text, confidence)
                        if text_info and isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text = text_info[0]
                            confidence = text_info[1]
                            
                            detections.append({
                                "text": text,
                                "confidence": confidence,
                                "bbox": bbox
                            })
                            rec_texts.append(text)
                            rec_scores.append(confidence)
            
            full_text = "\n".join(rec_texts) if rec_texts else ""
            avg_conf = np.mean(rec_scores) if rec_scores else 0.0
            
            logger.info(f"✓ Extracted {len(rec_texts)} text regions (conf: {avg_conf:.1%})")
            
            return full_text, detections, avg_conf
        
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}", exc_info=True)
            return "", [], 0.0

    def detect_document_type(self, text: str) -> DocumentType:
        """Detect document type."""
        text_upper = text.upper()
        
        # PAN Card (Indian Permanent Account Number)
        if any(w in text_upper for w in ["PERMANENT ACCOUNT NUMBER", "PAN", "INCOME TAX DEPARTMENT", "GOVT OF INDIA"]):
            return DocumentType.PAN_CARD
        
        if any(w in text_upper for w in ["PASSPORT", "PASSEPORT", "PASSAPORTE", "REISEPASS"]):
            return DocumentType.PASSPORT
        
        if any(w in text_upper for w in ["DRIVING", "DRIVER", "FÜHRERSCHEIN", "LICENCIA"]):
            return DocumentType.DRIVERS_LICENSE
        
        if any(w in text_upper for w in ["DNI", "NATIONAL", "PERSONALAUSWEIS", "CIDADAO"]):
            return DocumentType.NATIONAL_ID
        
        if any(w in text_upper for w in ["IDENTITY", "ID CARD", "CARTÃO"]):
            return DocumentType.ID_CARD
        
        return DocumentType.UNKNOWN

    def extract_document_number(self, detections: List[Dict], full_text: str, doc_type: DocumentType) -> Optional[str]:
        """Extract document number with country-specific patterns."""
        text_clean = full_text.replace(" ", "").upper()
        
        # Indian PAN Card: 5 letters + 4 digits + 1 letter (e.g., AAAAA1234A)
        if doc_type == DocumentType.PAN_CARD or "PERMANENT ACCOUNT NUMBER" in full_text.upper() or "PAN" in full_text.upper():
            # Pattern: 5 uppercase letters, 4 digits, 1 uppercase letter
            match = re.search(r'\b([A-Z]{5}\d{4}[A-Z])\b', full_text.replace(" ", "").upper())
            if match:
                return match.group(1)
            # Also check in individual detections
            for det in detections:
                text = det["text"].replace(" ", "").upper()
                match = re.search(r'\b([A-Z]{5}\d{4}[A-Z])\b', text)
                if match:
                    return match.group(1)
        
        # Indian Driving License: DL1 20220166923 or DL-01-2022-0166923
        if "INDIAN" in text_clean or "DRIVING" in text_clean or "LICENCE" in text_clean:
            # Pattern: DL followed by numbers (with or without spaces, handle newlines)
            # First try to find DL and number in same detection
            for det in detections:
                text = det["text"].replace('\n', ' ').replace('\r', ' ')  # Normalize whitespace
                match = re.search(r'\b(DL[0-9]?\s+[0-9]{8,12})\b', text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            
            # If not found, try to find DL in one detection and number in next
            for idx in range(len(detections) - 1):
                text1 = detections[idx]["text"].strip()
                text2 = detections[idx + 1]["text"].strip()
                dl_match = re.search(r'\b(DL[0-9]?)\b', text1, re.IGNORECASE)
                num_match = re.search(r'\b([0-9]{8,12})\b', text2)
                if dl_match and num_match:
                    return f"{dl_match.group(1)} {num_match.group(1)}"
            
            # Fallback: search in full text
            match = re.search(r'\b(DL[0-9]?\s+[0-9]{8,12})\b', full_text.replace('\n', ' '), re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Spanish DNI: 99999999R
        if "ESPAÑA" in text_clean or "DNI" in text_clean:
            match = re.search(r'(\d{8}[A-Z])', text_clean)
            if match:
                return match.group(1)
        
        # Portuguese: 42634925 (with optional suffix)
        if "PORTUGAL" in text_clean or "PORTUGUESA" in text_clean:
            for det in detections:
                text = det["text"].replace(" ", "")
                # Look for 8-9 digit number
                match = re.search(r'(\d{8,9})', text)
                if match and not re.search(r'\d{2}[.\-/]\d{2}[.\-/]\d', text):
                    # Not a date
                    num = match.group(1)
                    # Check if it's followed by check digits
                    full_match = re.search(rf'{num}\s*\d?\s*[A-Z]{{2,4}}', full_text, re.IGNORECASE)
                    if full_match:
                        return full_match.group(0)
                    return num
        
        # Swiss/German: SOA00A92 or C012345678
        if "SCHWEIZ" in text_clean or "SUISSE" in text_clean or "DEUTSCHLAND" in text_clean:
            # Skip sample text
            for det in detections:
                text = det["text"].upper()
                if "SAMPLE" in text or "DATADANA" in text or "MUSTERMANN" in text:
                    continue
                
                # Pattern: 3 letters + alphanumeric
                match = re.search(r'\b([A-Z]{1,3}\d[A-Z0-9]{4,7})\b', text)
                if match:
                    return match.group(1)
                
                # Pattern: Letter + 9 digits
                match = re.search(r'\b([A-Z]\d{9})\b', text)
                if match:
                    return match.group(1)
        
        return None

    def extract_nationality(self, full_text: str) -> Optional[str]:
        """Extract nationality with priority order."""
        text_upper = full_text.upper()
        
        # Priority order: check longer/more specific patterns first
        priority_patterns = [
            (r'\bSCHWEIZ\b', "Switzerland"),
            (r'\bSUISSE\b', "Switzerland"),
            (r'\bSVIZZERA\b', "Switzerland"),
            (r'\bSVIZRA\b', "Switzerland"),
            (r'\bSWITZERLAND\b', "Switzerland"),
            (r'\bINDIA\b', "Indian"),
            (r'\bINDIAN\b', "Indian"),
            (r'\bDEUTSCHLAND\b', "Germany"),
            (r'\bESPAÑA\b', "Spain"),
            (r'\bESPANA\b', "Spain"),
            (r'\bPORTUGAL\b', "Portugal"),
            # Then check 3-letter codes
            (r'\bCHE\b', "Switzerland"),
            (r'\bIND\b', "Indian"),
            (r'\bESP\b', "Spain"),
            (r'\bPRT\b', "Portugal"),
            (r'\bDEU\b', "Germany"),
        ]
        
        for pattern, country in priority_patterns:
            if re.search(pattern, text_upper):
                return country
        
        return None

    def extract_date_from_label(self, detections: List[Dict], labels: List[str]) -> Optional[str]:
        """Extract date from label patterns like 'Date of Birth: 10-02-2003'."""
        for det in detections:
            text = det["text"].strip()
            for label in labels:
                # Pattern: "Date of Birth: 10-02-2003" or "Date of Birth 10-02-2003" or "Date ofBirth10-02-2003" (no space)
                # More flexible pattern that handles missing spaces between label and date
                # Escape special regex chars but allow flexible spacing
                label_escaped = re.escape(label).replace(r'\ ', r'\s*')  # Allow flexible spaces in label
                pattern = rf'(?:{label_escaped})[:\s]*(\d{{1,2}}[.\-/]\d{{1,2}}[.\-/]\d{{2,4}})'
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    date_str = match.group(1)
                    # Try to parse and format
                    try:
                        # Try DD-MM-YYYY format
                        parts = re.split(r'[.\-/]', date_str)
                        if len(parts) == 3:
                            if len(parts[2]) == 2:
                                parts[2] = f"20{parts[2]}" if int(parts[2]) < 50 else f"19{parts[2]}"
                            return f"{int(parts[0]):02d}.{int(parts[1]):02d}.{parts[2]}"
                    except:
                        pass
                    return date_str
                
                # Also try pattern where label might be split: "Date ofBirth10-02-2003"
                # Match "Date" followed by "of" (with optional space) followed by "Birth" (with optional space) followed by date
                flexible_pattern = r'(?:Date\s+of\s*Birth|Date\s*of\s*Birth|DateofBirth)[:\s]*(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4})'
                match = re.search(flexible_pattern, text, re.IGNORECASE)
                if match:
                    date_str = match.group(1)
                    try:
                        parts = re.split(r'[.\-/]', date_str)
                        if len(parts) == 3:
                            if len(parts[2]) == 2:
                                parts[2] = f"20{parts[2]}" if int(parts[2]) < 50 else f"19{parts[2]}"
                            return f"{int(parts[0]):02d}.{int(parts[1]):02d}.{parts[2]}"
                    except:
                        pass
                    return date_str
        
        # Also check if label is in one detection and date in next
        for idx in range(len(detections) - 1):
            text = detections[idx]["text"].strip()
            for label in labels:
                if re.search(rf'^{re.escape(label)}[:\s]*$', text, re.IGNORECASE):
                    next_text = detections[idx + 1]["text"].strip()
                    # Check if next text is a date
                    date_match = re.search(r'(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4})', next_text)
                    if date_match:
                        return date_match.group(1)
        return None
    
    def extract_address(self, detections: List[Dict], full_text: str) -> Optional[str]:
        """Extract address from document."""
        # Look for "Address:" label and collect all following address parts
        for idx, det in enumerate(detections):
            text = det["text"].strip()
            # Pattern: "Address:" or "Address:House NO-28," (handle missing space)
            addr_match = re.search(r'(?:Address|ADDRESS|Dirección|Morada)[:\s]+(.+)', text, re.IGNORECASE)
            if addr_match:
                # Start with the first part found in same line
                address_parts = [addr_match.group(1).strip()]
                
                # Collect following detections that are part of the address
                for j in range(idx + 1, min(idx + 10, len(detections))):
                    next_text = detections[j]["text"].strip()
                    # Clean up: remove leading dots/commas/spaces
                    next_text = re.sub(r'^[.,\s]+', '', next_text)
                    
                    # Stop if we hit another major label
                    if re.search(r'^(?:Name|Date|Number|Blood|Organ|Son|Daughter|Wife|Husband|Father|Mother)', next_text, re.IGNORECASE):
                        break
                    
                    # Include if it looks like an address part (has some content)
                    if len(next_text) > 2 and not re.match(r'^[.,\s]+$', next_text):
                        address_parts.append(next_text)
                
                # Join parts with space
                full_addr = " ".join(address_parts)
                # Clean up multiple spaces and trailing punctuation
                full_addr = re.sub(r'\s+', ' ', full_addr)
                full_addr = re.sub(r'[.,\s]+$', '', full_addr)
                if len(full_addr) > 5:
                    return full_addr
        
        # Look for address after "Address:" label in next detections
        for idx in range(len(detections) - 1):
            text = detections[idx]["text"].strip()
            if re.search(r'^(?:Address|ADDRESS|Dirección|Morada)[:\s]*$', text, re.IGNORECASE):
                # Collect following detections until we hit another label or end
                address_parts = []
                for j in range(idx + 1, min(idx + 10, len(detections))):
                    next_text = detections[j]["text"].strip()
                    # Clean up: remove leading dots/commas
                    next_text = re.sub(r'^[.,\s]+', '', next_text)
                    # Stop if we hit another label
                    if re.search(r'^(?:Name|Date|Number|Blood|Organ|Son|Daughter|Wife|Husband|Father|Mother)', next_text, re.IGNORECASE):
                        break
                    if len(next_text) > 2 and not re.match(r'^[.,\s]+$', next_text):
                        address_parts.append(next_text)
                if address_parts:
                    # Join and clean up
                    full_addr = " ".join(address_parts)
                    full_addr = re.sub(r'\s+', ' ', full_addr)
                    full_addr = re.sub(r'[.,\s]+$', '', full_addr)
                    return full_addr
        
        return None

    def extract_gender(self, detections: List[Dict], full_text: str) -> Optional[str]:
        """Extract gender with improved detection."""
        text_upper = full_text.upper()
        
        # Strategy 1: Check full_text for pattern "| F |" or "| M |" (most reliable for passports)
        if re.search(r'\|\s*F\s*\|', text_upper):
            logger.info("✓ Found gender F (pipe pattern in full text)")
            return "F"
        if re.search(r'\|\s*M\s*\|', text_upper):
            logger.info("✓ Found gender M (pipe pattern in full text)")
            return "M"
        
        # Strategy 2: Look for explicit gender fields
        for det in detections:
            text = det["text"].upper()
            # Pattern: "SEXO: F" or "SEX: M" or "Geschlecht · Sexe • Sesso"
            match = re.search(r'\b(SEX|SEXO|GENDER|GESCHLECHT|SESSO)[:\s·•×]+(M|F)\b', text)
            if match:
                logger.info(f"✓ Found gender via label: {match.group(2)}")
                return match.group(2)
        
        # Strategy 3: Check ALL detections for standalone M/F
        for idx, det in enumerate(detections):
            line = det["text"].strip()
            line_upper = line.upper()
            
            # Exact match for single letter F or M
            if line_upper in ['F', 'M']:
                logger.info(f"✓ Found gender (single letter at idx {idx}): {line_upper}")
                return line_upper
            
            # Very short lines with just F or M
            if len(line) <= 3 and line_upper in ['F', 'M', 'F.', 'M.']:
                logger.info(f"✓ Found gender (short line at idx {idx}): {line_upper}")
                return line_upper.rstrip('.')
            
            # Check for patterns like "F | 170" or "M | 180" (gender with height)
            if re.search(r'\bF\b\s*\|\s*\d{2,3}', line_upper):
                logger.info(f"✓ Found gender F (with height at idx {idx}): {line}")
                return "F"
            if re.search(r'\bM\b\s*\|\s*\d{2,3}', line_upper):
                logger.info(f"✓ Found gender M (with height at idx {idx}): {line}")
                return "M"
            
            # Check for F or M as isolated character (not part of a word)
            if re.search(r'\bF\b.*\d{3}', line_upper) or re.search(r'\d{3}.*\bF\b', line_upper):
                logger.info(f"✓ Found gender F (near number at idx {idx}): {line}")
                return "F"
            if re.search(r'\bM\b.*\d{3}', line_upper) or re.search(r'\d{3}.*\bM\b', line_upper):
                logger.info(f"✓ Found gender M (near number at idx {idx}): {line}")
                return "M"
        
        logger.info("⚠ No gender found in detections")
        return None

    def extract_structured(self, image: np.ndarray, confidence_threshold: float = 0.3) -> OCRResult:
        """Main extraction method with context-aware processing."""
        logger.info("=" * 60)
        logger.info("Starting context-aware OCR extraction...")
        
        # Extract text
        full_text, detections, avg_conf = self.extract_text_paddle(image)
        
        if not detections:
            logger.warning("⚠ No text extracted")
            return OCRResult(document_type=DocumentType.UNKNOWN.value, confidence=0.0, extracted_text="")
        
        logger.info(f"✓ Extracted {len(detections)} text regions")
        
        # Detect document type
        doc_type = self.detect_document_type(full_text)
        logger.info(f"✓ Document type: {doc_type.value}")
        
        # Extract all dates first
        all_dates = self.extractor.extract_all_dates(detections)
        logger.info(f"✓ Found {len(all_dates)} date candidates")
        
        # Extract fields
        full_name = self.extractor.extract_name_from_detections(detections)
        document_number = self.extract_document_number(detections, full_text, doc_type)
        nationality = self.extract_nationality(full_text)
        
        # Try to extract date of birth from "Date of Birth:" label first
        # Handle variations: "Date of Birth", "Date ofBirth", "DateofBirth", etc.
        date_of_birth = self.extract_date_from_label(detections, [
            "Date of Birth", "Date ofBirth", "DateofBirth", "DOB", 
            "Date of birth", "Birth Date", "Date of Birth"
        ])
        if not date_of_birth:
            date_of_birth = self.extractor.select_date_by_type(all_dates, "birth")
        
        issue_date = self.extractor.select_date_by_type(all_dates, "issue")
        expiry_date = self.extractor.select_date_by_type(all_dates, "expiry")
        gender = self.extract_gender(detections, full_text)
        address = self.extract_address(detections, full_text)
        
        # Parse MRZ if available
        mrz_lines = [d["text"] for d in detections if len(d["text"].replace(" ", "").replace("<", "")) >= 28 and 
                     re.match(r'^[A-Z0-9<\s]+$', d["text"].upper())]
        logger.info(f"✓ Found {len(mrz_lines)} potential MRZ lines")
        if mrz_lines:
            for i, line in enumerate(mrz_lines):
                logger.info(f"  MRZ line {i+1}: {line[:50]}...")
        mrz_data = MRZParser.parse_passport_mrz(mrz_lines) if mrz_lines else MRZData()
        
        # MRZ overrides (more reliable)
        if mrz_data.valid:
            logger.info("✓ MRZ validated - using MRZ data")
            if mrz_data.surname and mrz_data.given_names:
                full_name = f"{mrz_data.given_names} {mrz_data.surname}".strip()
            if mrz_data.date_of_birth:
                date_of_birth = mrz_data.date_of_birth
            if mrz_data.expiry_date:
                expiry_date = mrz_data.expiry_date
            if mrz_data.passport_number:
                document_number = mrz_data.passport_number
            if mrz_data.gender:
                gender = mrz_data.gender
            if mrz_data.nationality:
                nationality = mrz_data.nationality
        
        # Log results
        logger.info("Extraction results:")
        if full_name:
            logger.info(f"  ✓ Name: {full_name}")
        if document_number:
            logger.info(f"  ✓ Doc Number: {document_number}")
        if date_of_birth:
            logger.info(f"  ✓ DOB: {date_of_birth}")
        if issue_date:
            logger.info(f"  ✓ Issue: {issue_date}")
        if expiry_date:
            logger.info(f"  ✓ Expiry: {expiry_date}")
        if nationality:
            logger.info(f"  ✓ Nationality: {nationality}")
        if gender:
            logger.info(f"  ✓ Gender: {gender}")
        
        # Build result
        result = OCRResult(
            document_type=doc_type.value,
            confidence=avg_conf,
            full_name=full_name,
            document_number=document_number,
            nationality=nationality,
            date_of_birth=date_of_birth,
            issue_date=issue_date,
            expiry_date=expiry_date,
            gender=gender,
            address=address,
            mrz_data=asdict(mrz_data) if mrz_data.valid else None,
            extracted_text=" | ".join([d["text"] for d in detections])  # Include all detections
        )
        
        # Calculate confidence
        critical_fields = [full_name, document_number, date_of_birth, nationality, gender]
        extracted_count = sum(1 for f in critical_fields if f)
        result.confidence = (avg_conf * 0.4) + (extracted_count / 5 * 0.6)
        
        logger.info(f"✓ Overall confidence: {result.confidence:.1%}")
        logger.info(f"✓ Extracted {extracted_count}/5 critical fields")
        logger.info("=" * 60)
        
        return result


# Singleton pattern
_ocr_instance: Optional[PaddleOCRExtractor] = None
_ocr_lock = threading.Lock()


def get_ocr_extractor() -> PaddleOCRExtractor:
    """Thread-safe singleton getter."""
    global _ocr_instance
    if _ocr_instance is None:
        with _ocr_lock:
            if _ocr_instance is None:
                _ocr_instance = PaddleOCRExtractor(
                    languages=config.get("models", "ocr", "languages", default=["en"]),
                )
                logger.info("✓ OCR extractor singleton created")
    return _ocr_instance


def reset_ocr_extractor() -> None:
    """Reset singleton (for testing)."""
    global _ocr_instance
    _ocr_instance = None
    logger.info("OCR extractor singleton reset")


# Backward compatibility
OCRExtractor = PaddleOCRExtractor
