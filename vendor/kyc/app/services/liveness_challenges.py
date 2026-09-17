# app/services/liveness_challenges.py

"""
Liveness Challenge Generation and Validation.
Handles challenge generation, validation, and security features.
Adapted from face_liveness_detection-Anti-spoofing/questions.py
Enhanced with security features (HMAC, nonce, timestamp).
"""

import random
import hashlib
import hmac
import time
import uuid
import threading
from typing import Dict, Optional, List, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta

from configs.config import config
from utils.logger import get_logger

logger = get_logger(__name__, log_file="liveness.log")


class ChallengeType(str, Enum):
    """Types of liveness challenges."""
    BLINK = "blink"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    # Note: Emotion-based challenges (smile, surprise, angry) are skipped for MVP
    # due to TensorFlow 1.x model incompatibility


class ChallengeStatus(str, Enum):
    """Challenge validation status."""
    PASS = "pass"
    FAIL = "fail"
    PENDING = "pending"
    EXPIRED = "expired"
    INVALID = "invalid"


class LivenessChallenge:
    """Represents a liveness challenge session (can contain multiple tasks)."""
    
    def __init__(
        self,
        challenge_id: str,
        challenge_type: ChallengeType = None,  # For backward compatibility
        question_text: str = None,  # For backward compatibility
        challenge_types: List[ChallengeType] = None,  # NEW: Multiple challenges
        question_texts: List[str] = None,  # NEW: Multiple questions
        timestamp: float = None,
        nonce: str = None,
        signature: Optional[str] = None,
        expires_in: int = 30  # seconds
    ):
        self.challenge_id = challenge_id
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.nonce = nonce if nonce is not None else uuid.uuid4().hex
        self.signature = signature
        self.expires_at = self.timestamp + expires_in
        self.status = ChallengeStatus.PENDING
        
        # Support both single and multiple challenges
        if challenge_types is not None and question_texts is not None:
            # Multi-challenge mode
            self.challenge_types = challenge_types
            self.question_texts = question_texts
            # For backward compatibility
            self.challenge_type = challenge_types[0] if challenge_types else None
            self.question_text = question_texts[0] if question_texts else ""
        else:
            # Single challenge mode (backward compatibility)
            self.challenge_type = challenge_type
            self.question_text = question_text
            self.challenge_types = [challenge_type] if challenge_type else []
            self.question_texts = [question_text] if question_text else []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert challenge to dictionary for API response."""
        # Multi-challenge response
        if len(self.challenge_types) > 1:
            return {
                "challenge_id": self.challenge_id,
                "challenge_types": [ct.value for ct in self.challenge_types],
                "questions": self.question_texts,
                "instructions": [self._get_instruction(ct) for ct in self.challenge_types],
                "timestamp": self.timestamp,
                "expires_at": self.expires_at,
                "nonce": self.nonce,
                "signature": self.signature,
                "multi_challenge": True
            }
        else:
            # Single challenge response (backward compatibility)
            return {
                "challenge_id": self.challenge_id,
                "challenge_type": self.challenge_type.value if self.challenge_type else None,
                "question": self.question_text,
                "instruction": self._get_instruction(self.challenge_type),
                "timestamp": self.timestamp,
                "expires_at": self.expires_at,
                "nonce": self.nonce,
                "signature": self.signature,
                "multi_challenge": False
            }
    
    def _get_instruction(self, challenge_type: ChallengeType = None) -> str:
        """Get user-friendly instruction text."""
        if challenge_type is None:
            challenge_type = self.challenge_type
        
        instructions = {
            ChallengeType.BLINK: "Blink your eyes once",
            ChallengeType.TURN_LEFT: "Turn your face to the left",
            ChallengeType.TURN_RIGHT: "Turn your face to the right"
        }
        return instructions.get(challenge_type, self.question_text if hasattr(self, 'question_text') else "")
    
    def is_expired(self) -> bool:
        """Check if challenge has expired."""
        return time.time() > self.expires_at
    
    def verify_signature(self, secret_key: str) -> bool:
        """Verify HMAC signature of challenge."""
        if not self.signature:
            return False
        
        expected_signature = self._generate_signature(secret_key)
        return hmac.compare_digest(self.signature, expected_signature)
    
    def _generate_signature(self, secret_key: str) -> str:
        """Generate HMAC signature for challenge."""
        # Include all challenge types in signature
        challenge_str = ",".join([ct.value for ct in self.challenge_types])
        message = f"{self.challenge_id}:{challenge_str}:{self.timestamp}:{self.nonce}"
        signature = hmac.new(
            secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature


class ChallengeGenerator:
    """
    Generates and manages liveness challenges.
    """
    
    # Question bank - excluding emotion challenges for MVP
    CHALLENGE_TYPES = [
        ChallengeType.BLINK,
        ChallengeType.TURN_LEFT,
        ChallengeType.TURN_RIGHT
    ]
    
    QUESTION_TEXTS = {
        ChallengeType.BLINK: "blink eyes",
        ChallengeType.TURN_LEFT: "turn face left",
        ChallengeType.TURN_RIGHT: "turn face right"
    }
    
    def __init__(self, secret_key: Optional[str] = None, expires_in: Optional[int] = None):
        """
        Initialize challenge generator.
        
        Args:
            secret_key: Secret key for HMAC signing (uses config if None)
            expires_in: Challenge expiration time in seconds (uses config if None)
        """
        if secret_key is None:
            secret_key = config.get("liveness", "security", "hmac_secret", default="change-me-in-production")
        
        if expires_in is None:
            expires_in = config.get("liveness", "challenge", "expires_in", default=120)
        
        self.secret_key = secret_key
        self.expires_in = expires_in
        
        # In-memory challenge storage (for validation)
        # In production, consider Redis or database
        self._active_challenges: Dict[str, LivenessChallenge] = {}
        
        logger.info(f"ChallengeGenerator initialized (expires_in: {expires_in}s)")
    
    def generate_challenge(self, challenge_type: Optional[ChallengeType] = None, num_challenges: int = 2, challenge_types: Optional[List[ChallengeType]] = None) -> LivenessChallenge:
        """
        Generate a new liveness challenge (single or multi-challenge).
        
        Args:
            challenge_type: Specific challenge type for single challenge, or None for random
            num_challenges: Number of challenges to generate (default: 2 for multi-challenge)
            challenge_types: Explicit sequence (e.g. turn_left then turn_right)
        """
        challenge_id = str(uuid.uuid4())
        timestamp = time.time()
        nonce = uuid.uuid4().hex

        if challenge_types:
            selected_types = list(challenge_types)
            question_texts = [self.QUESTION_TEXTS[ct] for ct in selected_types]
            challenge = LivenessChallenge(
                challenge_id=challenge_id,
                challenge_types=selected_types,
                question_texts=question_texts,
                timestamp=timestamp,
                nonce=nonce,
                expires_in=self.expires_in
            )
            logger.info(f"Generated requested challenge: {[ct.value for ct in selected_types]} (ID: {challenge_id})")
        elif num_challenges > 1:
            selected_types = random.sample(self.CHALLENGE_TYPES, min(num_challenges, len(self.CHALLENGE_TYPES)))
            question_texts = [self.QUESTION_TEXTS[ct] for ct in selected_types]
            
            challenge = LivenessChallenge(
                challenge_id=challenge_id,
                challenge_types=selected_types,
                question_texts=question_texts,
                timestamp=timestamp,
                nonce=nonce,
                expires_in=self.expires_in
            )
            
            logger.info(f"Generated multi-challenge: {[ct.value for ct in selected_types]} (ID: {challenge_id})")
        else:
            # Single challenge mode (backward compatibility)
            if challenge_type is None:
                challenge_type = random.choice(self.CHALLENGE_TYPES)
            
            challenge = LivenessChallenge(
                challenge_id=challenge_id,
                challenge_type=challenge_type,
                question_text=self.QUESTION_TEXTS[challenge_type],
                timestamp=timestamp,
                nonce=nonce,
                expires_in=self.expires_in
            )
            
            logger.info(f"Generated challenge: {challenge_type.value} (ID: {challenge_id})")
        
        # Generate signature
        challenge.signature = challenge._generate_signature(self.secret_key)
        
        # Store for validation
        self._active_challenges[challenge_id] = challenge
        
        return challenge
    
    def generate_multiple(self, count: int, allow_duplicates: bool = False) -> List[LivenessChallenge]:
        """
        Generate multiple challenges.
        
        Args:
            count: Number of challenges to generate
            allow_duplicates: If False, ensures unique challenge types
        
        Returns:
            List of LivenessChallenge objects
        """
        challenges = []
        used_types = set()
        
        for _ in range(count):
            if not allow_duplicates and len(used_types) >= len(self.CHALLENGE_TYPES):
                # Reset if we've used all types
                used_types.clear()
            
            challenge_type = None
            while challenge_type is None or (not allow_duplicates and challenge_type in used_types):
                challenge_type = random.choice(self.CHALLENGE_TYPES)
            
            if not allow_duplicates:
                used_types.add(challenge_type)
            
            challenge = self.generate_challenge(challenge_type)
            challenges.append(challenge)
        
        return challenges
    
    def validate_challenge(self, challenge_id: str) -> Tuple[bool, Optional[LivenessChallenge]]:
        """
        Validate and retrieve challenge.
        
        Args:
            challenge_id: Challenge ID to validate
        
        Returns:
            Tuple of (is_valid, challenge)
        """
        challenge = self._active_challenges.get(challenge_id)
        
        if challenge is None:
            logger.warning(f"Challenge not found: {challenge_id}")
            return False, None
        
        if challenge.is_expired():
            logger.warning(f"Challenge expired: {challenge_id}")
            self._active_challenges.pop(challenge_id, None)
            return False, None
        
        if not challenge.verify_signature(self.secret_key):
            logger.warning(f"Challenge signature invalid: {challenge_id}")
            return False, None
        
        return True, challenge
    
    def validate_response(
        self,
        challenge_id: str,
        detection_results: Dict[str, Any]
    ) -> Tuple[ChallengeStatus, Optional[str]]:
        """
        Validate challenge response based on detection results.
        
        Args:
            challenge_id: Challenge ID
            detection_results: Dictionary with:
                - blinks: int (number of blinks detected)
                - orientation: str ('left', 'right', or None)
                - orientations: List[str] (all orientations detected in sequence)
                - face_detected: bool
        
        Returns:
            Tuple of (status, message)
        """
        is_valid, challenge = self.validate_challenge(challenge_id)
        
        if not is_valid or challenge is None:
            return ChallengeStatus.INVALID, "Challenge not found or expired"
        
        if challenge.is_expired():
            self._active_challenges.pop(challenge_id, None)
            return ChallengeStatus.EXPIRED, "Challenge expired"
        
        # Multi-challenge validation
        if len(challenge.challenge_types) > 1:
            return self._validate_multi_challenge(challenge, detection_results)
        
        # Single challenge validation (backward compatibility)
        return self._validate_single_challenge(challenge, detection_results)
    
    def _validate_single_challenge(
        self,
        challenge: LivenessChallenge,
        detection_results: Dict[str, Any]
    ) -> Tuple[ChallengeStatus, Optional[str]]:
        """Validate a single challenge."""
        # Validate based on challenge type
        if challenge.challenge_type == ChallengeType.BLINK:
            blinks = detection_results.get("blinks", 0)
            if blinks >= 1:
                challenge.status = ChallengeStatus.PASS
                self._active_challenges.pop(challenge.challenge_id, None)
                return ChallengeStatus.PASS, "Blink detected successfully"
            else:
                return ChallengeStatus.FAIL, "No blink detected"
        
        elif challenge.challenge_type == ChallengeType.TURN_LEFT:
            orientation = detection_results.get("orientation")
            if orientation == "left":
                challenge.status = ChallengeStatus.PASS
                self._active_challenges.pop(challenge.challenge_id, None)
                return ChallengeStatus.PASS, "Left orientation detected"
            else:
                return ChallengeStatus.FAIL, f"Expected left, got {orientation}"
        
        elif challenge.challenge_type == ChallengeType.TURN_RIGHT:
            orientation = detection_results.get("orientation")
            if orientation == "right":
                challenge.status = ChallengeStatus.PASS
                self._active_challenges.pop(challenge.challenge_id, None)
                return ChallengeStatus.PASS, "Right orientation detected"
            else:
                return ChallengeStatus.FAIL, f"Expected right, got {orientation}"
        
        return ChallengeStatus.INVALID, "Unknown challenge type"
    
    def _validate_multi_challenge(
        self,
        challenge: LivenessChallenge,
        detection_results: Dict[str, Any]
    ) -> Tuple[ChallengeStatus, Optional[str]]:
        """
        Validate multiple challenges completed in a single video.
        All challenges must be completed successfully.
        """
        blinks = detection_results.get("blinks", 0)
        orientations = detection_results.get("orientations", [])
        
        # Track which challenges were completed
        completed_challenges = []
        failed_challenges = []
        
        for challenge_type in challenge.challenge_types:
            if challenge_type == ChallengeType.BLINK:
                if blinks >= 1:
                    completed_challenges.append("blink")
                    logger.info(f"✓ Blink challenge completed ({blinks} blinks detected)")
                else:
                    failed_challenges.append("blink (no blink detected)")
                    logger.warning(f"✗ Blink challenge failed (0 blinks)")
            
            elif challenge_type == ChallengeType.TURN_LEFT:
                # Check if "left" appears in orientations
                if "left" in orientations:
                    completed_challenges.append("turn left")
                    logger.info(f"✓ Turn left challenge completed")
                else:
                    failed_challenges.append("turn left (not detected)")
                    logger.warning(f"✗ Turn left challenge failed")
            
            elif challenge_type == ChallengeType.TURN_RIGHT:
                # Check if "right" appears in orientations
                if "right" in orientations:
                    completed_challenges.append("turn right")
                    logger.info(f"✓ Turn right challenge completed")
                else:
                    failed_challenges.append("turn right (not detected)")
                    logger.warning(f"✗ Turn right challenge failed")
        
        # All challenges must be completed
        if len(completed_challenges) == len(challenge.challenge_types):
            challenge.status = ChallengeStatus.PASS
            self._active_challenges.pop(challenge.challenge_id, None)
            message = f"All challenges completed: {', '.join(completed_challenges)}"
            logger.info(f"✅ Multi-challenge PASSED: {message}")
            return ChallengeStatus.PASS, message
        else:
            # Some challenges failed
            message = f"Completed: {', '.join(completed_challenges) if completed_challenges else 'none'}. Failed: {', '.join(failed_challenges)}"
            logger.warning(f"❌ Multi-challenge FAILED: {message}")
            return ChallengeStatus.FAIL, message
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired challenges.
        
        Returns:
            Number of challenges removed
        """
        current_time = time.time()
        expired_ids = [
            cid for cid, challenge in self._active_challenges.items()
            if challenge.expires_at < current_time
        ]
        
        for cid in expired_ids:
            self._active_challenges.pop(cid, None)
        
        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired challenges")
        
        return len(expired_ids)
    
    def get_challenge_count(self) -> int:
        """Get number of active challenges."""
        return len(self._active_challenges)


# ============================================================================
# Singleton Pattern
# ============================================================================

_generator_instance: Optional[ChallengeGenerator] = None
_generator_lock = threading.Lock()


def get_challenge_generator() -> ChallengeGenerator:
    """Get or create singleton ChallengeGenerator instance. Thread-safe."""
    global _generator_instance
    
    if _generator_instance is None:
        with _generator_lock:
            if _generator_instance is None:
                _generator_instance = ChallengeGenerator()
                logger.info("ChallengeGenerator singleton created")
    
    return _generator_instance


# ============================================================================
# Helper Functions (for backward compatibility with original repo)
# ============================================================================

def question_bank(index: int) -> str:
    """
    Get question text by index (backward compatibility).
    
    Args:
        index: Question index (0-2 for MVP: blink, turn_left, turn_right)
    
    Returns:
        Question text string
    """
    questions = [
        "blink eyes",
        "turn face left",
        "turn face right"
    ]
    
    if 0 <= index < len(questions):
        return questions[index]
    else:
        raise ValueError(f"Invalid question index: {index}")


def challenge_result(
    question: str,
    detection_results: Dict[str, Any],
    blinks_up: int = 0
) -> str:
    """
    Validate challenge result (backward compatibility).
    
    Args:
        question: Question text
        detection_results: Detection results dictionary
        blinks_up: Number of blinks detected (1 = blink, 0 = no blink)
    
    Returns:
        "pass" or "fail"
    """
    if question == "blink eyes":
        if blinks_up >= 1:
            return "pass"
        else:
            return "fail"
    
    elif question == "turn face left":
        orientation = detection_results.get("orientation", [])
        if isinstance(orientation, list):
            orientation = orientation[0] if len(orientation) > 0 else None
        if orientation == "left":
            return "pass"
        else:
            return "fail"
    
    elif question == "turn face right":
        orientation = detection_results.get("orientation", [])
        if isinstance(orientation, list):
            orientation = orientation[0] if len(orientation) > 0 else None
        if orientation == "right":
            return "pass"
        else:
            return "fail"
    
    return "fail"

