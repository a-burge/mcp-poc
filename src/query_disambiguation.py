"""Query disambiguation for medication queries."""
import difflib
import logging
from typing import List, Optional, Dict, Any

from src.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

# Generic tokens that are too common to be reliable brand identifiers
BAD_TOKENS = {"pro", "nor", "micro", "mono", "vet", "plus"}

# Lazy import to avoid circular dependencies
_ingredients_manager = None


import unicodedata
import re

def _norm(s: str) -> str:
    """Normalize text: lowercase, strip diacritics and punctuation, collapse spaces."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).lower()
    s = "".join(c for c in s if not unicodedata.combining(c))
    # remove punctuation
    s = re.sub(r"[^\w\s]", " ", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_icelandic(s: str) -> str:
    if not s:
        return ""
    # 1) Unicode normalization
    s = unicodedata.normalize("NFKC", s)
    # 2) Lowercase
    s = s.lower()
    return s

def strip_diacritics(s: str) -> str:
    # Optional: make matching accent-insensitive
    nfkd_form = unicodedata.normalize('NFKD', s)
    return "".join(c for c in nfkd_form if not unicodedata.combining(c))

def _get_ingredients_manager():
    """Get IngredientsManager instance (lazy import)."""
    global _ingredients_manager
    if _ingredients_manager is None:
        try:
            from src.ingredients_manager import IngredientsManager
            _ingredients_manager = IngredientsManager()
        except Exception as e:
            logger.warning(f"Could not load IngredientsManager: {e}")
            return None
    return _ingredients_manager

def detect_medications_in_query(query: str, available_medications: List[str]) -> List[str]:
    """
    High-precision brand detection using word-boundary matching.
    
    Uses strict matching criteria to avoid false positives:
    - Minimum token length of 4 characters
    - Word-boundary regex matching (not substring)
    - Denylist for generic tokens
    
    Args:
        query: User query text
        available_medications: List of available medication names
        
    Returns:
        List of matching medication names
    """
    q = _norm(query)
    matches = []

    logger.debug(f"Query normalized: {repr(q)}")

    for med in available_medications:
        base = _norm(med)
        token = base.split("_")[0]

        # Skip short tokens (minimum 4 characters for reliability)
        if not token or len(token) < 4:
            logger.debug(f"Skipping short token: {repr(token)} from {repr(med)}")
            continue
        
        # Skip generic tokens that are too common
        if token in BAD_TOKENS:
            logger.debug(f"Skipping generic token: {repr(token)} from {repr(med)}")
            continue

        # Use word-boundary regex matching instead of substring matching
        # This prevents "pro" from matching inside "ibuprofein"
        pattern = rf"\b{re.escape(token)}\b"
        if re.search(pattern, q):
            matches.append(med)
            logger.debug(f"Matched: {repr(med)} via token {repr(token)}")

    logger.debug(f"Final matches: {matches}")
    return list(set(matches))

def detect_active_ingredients_in_query(query: str, ingredients_manager: Optional[Any] = None) -> List[str]:
    """
    High-precision ingredient detection with fuzzy matching for typos.
    
    Uses both exact word-boundary matching and fuzzy matching to handle:
    - Exact matches: "ibuprofen" matches "ibuprofen"
    - Typos: "ibuprofein" matches "ibuprofen" (similarity >= 0.8)
    - Short variations: edit distance <= 1 for short words
    
    Args:
        query: User query text
        ingredients_manager: Optional IngredientsManager instance
        
    Returns:
        List of detected ingredient names (INN names)
    """
    if ingredients_manager is None:
        ingredients_manager = _get_ingredients_manager()
    if not ingredients_manager:
        return []

    q = _norm(query)
    # Tokenize query into words for fuzzy matching
    query_words = q.split()
    
    detected = []
    for ing_key, ing_data in ingredients_manager.ingredients.items():
        inn = ing_data.get("inn_name", "")
        if not inn:
            continue
        inn_norm = _norm(inn)
        
        # Skip short ingredient names (minimum 4 characters)
        if len(inn_norm) < 4:
            continue
        
        # First, try exact word-boundary matching (high precision)
        pattern = rf"\b{re.escape(inn_norm)}\b"
        if re.search(pattern, q):
            detected.append(inn)
            logger.debug(f"Exact match: {repr(inn)} in query")
            continue
        
        # If no exact match, try fuzzy matching for typos
        # Only match if similarity is high enough (>= 0.8)
        for word in query_words:
            if len(word) < 4:
                continue
            
            # Compute similarity ratio using SequenceMatcher
            similarity = difflib.SequenceMatcher(None, inn_norm, word).ratio()
            
            # Match if similarity >= 0.8 (catches common typos like "ibuprofein" -> "ibuprofen")
            if similarity >= 0.8:
                detected.append(inn)
                logger.debug(
                    f"Fuzzy match: {repr(inn)} (norm: {repr(inn_norm)}) "
                    f"matched {repr(word)} (similarity: {similarity:.2f})"
                )
                break  # Found a match for this ingredient, move to next
    
    # Unique preserve order
    out = []
    seen = set()
    for x in detected:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def find_matching_medications(
    query: str,
    vector_store_manager: VectorStoreManager,
    use_ingredients: bool = True
) -> List[str]:
    """
    Find medications that match the query.
    
    Now includes ingredient-based matching for queries like "Hvað er Ibuprofen?"
    where the user asks about an ingredient name rather than a brand name.
    
    Args:
        query: User query text
        vector_store_manager: VectorStoreManager instance
        use_ingredients: If True, also search by ingredient names
        
    Returns:
        List of matching medication names
    """
    # Get all available medications
    available_medications = vector_store_manager.get_unique_medications()
    
    if not available_medications:
        return []
    
    # First, try brand name matching
    matches = detect_medications_in_query(query, available_medications)
    
    # If no brand matches and ingredients are enabled, try ingredient matching
    if not matches and use_ingredients:
        ingredients_manager = _get_ingredients_manager()
        if ingredients_manager:
            # Find drugs by ingredient name in query
            ingredient_matches = ingredients_manager.find_drugs_by_ingredient_query(query)
            
            # Filter to only drugs that exist in vector store
            if ingredient_matches:
                # Normalize drug names for comparison
                available_normalized = {
                    strip_diacritics(normalize_icelandic(m)).replace("_smpc", ""): m
                    for m in available_medications
                }
                
                for ing_drug in ingredient_matches:
                    ing_normalized = strip_diacritics(normalize_icelandic(ing_drug))
                    # Try exact match
                    if ing_drug in available_medications:
                        matches.append(ing_drug)
                    # Try normalized match
                    elif ing_normalized in available_normalized:
                        matches.append(available_normalized[ing_normalized])
                    # Try partial match
                    else:
                        for available in available_medications:
                            if (ing_normalized in normalize_icelandic(available) or 
                                normalize_icelandic(available) in ing_normalized):
                                matches.append(available)
                                break
                
                logger.info(f"Found {len(matches)} drugs via ingredient matching")
    
    return list(set(matches))  # Remove duplicates


def generate_clarification_prompt(matching_medications: List[str]) -> str:
    """
    Generate clarification prompt for ambiguous queries.
    
    Args:
        matching_medications: List of matching medication names
        
    Returns:
        Clarification prompt text in Icelandic
    """
    if len(matching_medications) == 0:
        return ""
    
    if len(matching_medications) == 1:
        return f"Spurningin tengist líklega: **{matching_medications[0]}**"
    
    # Multiple matches - generate clarification
    medication_list = "\n".join([f"- {med}" for med in matching_medications])
    
    prompt = f"""Spurningin gæti tengst nokkrum lyfjum. Vinsamlegast veldu hvaða lyf þú vilt spyrja um:

{medication_list}

Sláðu inn númerið eða nafnið á lyfinu sem þú vilt spyrja um."""
    
    return prompt


def should_disambiguate(
    query: str,
    vector_store_manager: VectorStoreManager
) -> Dict[str, Any]:
    """
    Determine if query needs disambiguation.
    
    Args:
        query: User query text
        vector_store_manager: VectorStoreManager instance
        
    Returns:
        Dictionary with:
        - needs_disambiguation: bool
        - matching_medications: List[str]
        - clarification_prompt: str
    """
    matching_medications = find_matching_medications(query, vector_store_manager)
    
    needs_disambiguation = len(matching_medications) > 1
    
    clarification_prompt = ""
    if needs_disambiguation:
        clarification_prompt = generate_clarification_prompt(matching_medications)
    elif len(matching_medications) == 1:
        clarification_prompt = generate_clarification_prompt(matching_medications)
    
    return {
        "needs_disambiguation": needs_disambiguation,
        "matching_medications": matching_medications,
        "clarification_prompt": clarification_prompt,
        "selected_medication": matching_medications[0] if len(matching_medications) == 1 else None,
    }
