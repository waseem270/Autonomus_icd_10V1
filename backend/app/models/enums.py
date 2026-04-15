from enum import Enum

class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class TemporalStatus(str, Enum):
    CURRENT = "current"
    HISTORY = "history"
    FAMILY_HISTORY = "family_history"
    RULED_OUT = "ruled_out"

class MappingMethod(str, Enum):
    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    LLM_RANKED = "llm_ranked"
    MANUAL = "manual"

class MappingStatus(str, Enum):
    PENDING = "pending"
    AUTO_ASSIGNED = "auto_assigned"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    MANUAL_REVIEW = "manual_review"
    NOT_FOUND = "not_found"

class RejectionReason(str, Enum):
    NEGATED = "negated"
    NO_MEAT = "no_meat"
    TEMPORAL_INVALID = "temporal_invalid"
    LOW_CONFIDENCE = "low_confidence"
    ICD_NOT_FOUND = "icd_not_found"
