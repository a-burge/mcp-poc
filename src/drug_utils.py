"""Drug name normalization and detection utilities."""
import re
import unicodedata
from typing import List


def normalize(s: str) -> str:
    """
    Normalize text for drug name matching.
    
    Converts to lowercase, removes accents/diacritics, strips non-alphanumeric
    characters, and collapses whitespace. This enables fuzzy matching of drug
    names with variations in accents, spacing, and punctuation.
    
    Examples:
        "Íbúfen" -> "ibufen"
        "Íbúfen 200 mg" -> "ibufen 200 mg"
        "Ibúfen_200mg_SmPC" -> "ibufen 200mg smpc"
    
    Args:
        s: Input string to normalize
        
    Returns:
        Normalized string (lowercase, no accents, alphanumeric + spaces only)
    """
    if not s:
        return ""
    
    # Convert to lowercase
    s = s.lower()
    
    # Remove accents/diacritics using NFKD normalization
    # NFKD decomposes characters (e.g., 'í' -> 'i' + combining mark)
    # Then encode/decode to ASCII ignores the combining marks
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("utf-8")
    
    # Replace non-alphanumeric characters with spaces
    s = re.sub(r"[^a-z0-9]+", " ", s)
    
    # Collapse multiple spaces and strip
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def detect_medications(question: str, known_drugs: List[str]) -> List[str]:
    """
    Detect which medications are mentioned in a question.
    
    Uses normalized token matching to find drug mentions even with variations
    in accents, spacing, or formatting. Matches if any token from a drug_id
    appears in the normalized question.
    
    Examples:
        detect_medications("Hverjar eru frábendingar fyrir Íbúfen?", ["Íbúfen_200mg_SmPC"])
        -> ["Íbúfen_200mg_SmPC"]
        
        detect_medications("Íbúfen og Panodil", ["Íbúfen_200mg_SmPC", "Panodil_SmPC"])
        -> ["Íbúfen_200mg_SmPC", "Panodil_SmPC"]
    
    Args:
        question: User's question text
        known_drugs: List of drug_id strings from vector store
        
    Returns:
        List of matching drug_id strings (preserves original casing/format)
    """
    if not question or not known_drugs:
        return []
    
    norm_question = normalize(question)
    matches = []
    
    for drug_id in known_drugs:
        norm_drug = normalize(drug_id)
        
        # Split drug_id into tokens (e.g., "ibufen 200mg smpc" -> ["ibufen", "200mg", "smpc"])
        tokens = norm_drug.split()
        
        # Match if any token appears in the normalized question
        # This handles partial matches (e.g., "Íbúfen" matches "Íbúfen_200mg_SmPC")
        if any(token in norm_question for token in tokens if len(token) > 2):  # Ignore very short tokens
            matches.append(drug_id)
    
    return matches
