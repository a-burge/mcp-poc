"""Query disambiguation for medication queries."""
import logging
from typing import List, Optional, Dict, Any

from src.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


def detect_medications_in_query(query: str, available_medications: List[str]) -> List[str]:
    """
    Detect which medications are mentioned in the query.
    
    Uses simple pattern matching to find medication names in the query.
    
    Args:
        query: User query text
        available_medications: List of available medication names
        
    Returns:
        List of medication names found in query
    """
    query_lower = query.lower()
    matches = []
    
    for medication in available_medications:
        medication_lower = medication.lower()
        # Check if medication name appears in query
        if medication_lower in query_lower:
            matches.append(medication)
        # Also check for partial matches (words in medication name)
        medication_words = medication_lower.split()
        if len(medication_words) > 1:
            # Check if any significant word from medication appears
            for word in medication_words:
                if len(word) > 3 and word in query_lower:
                    matches.append(medication)
                    break
    
    return list(set(matches))  # Remove duplicates


def find_matching_medications(
    query: str,
    vector_store_manager: VectorStoreManager
) -> List[str]:
    """
    Find medications that match the query.
    
    Args:
        query: User query text
        vector_store_manager: VectorStoreManager instance
        
    Returns:
        List of matching medication names
    """
    # Get all available medications
    available_medications = vector_store_manager.get_unique_medications()
    
    if not available_medications:
        return []
    
    # Detect medications mentioned in query
    matches = detect_medications_in_query(query, available_medications)
    
    return matches


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
