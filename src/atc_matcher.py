"""Drug-to-ATC code matching service."""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from src.drug_utils import normalize
from config import Config

logger = logging.getLogger(__name__)


class ATCMatcher:
    """Matches drugs to ATC codes using name and active ingredient matching."""
    
    def __init__(self, atc_index_path: Optional[Path] = None):
        """
        Initialize ATC matcher.
        
        Args:
            atc_index_path: Path to ATC index JSON file
        """
        if atc_index_path is None:
            atc_index_path = Config.ATC_INDEX_PATH
        
        self.atc_index_path = atc_index_path
        self.atc_index: Dict[str, Any] = {}
        self.drug_mappings: Dict[str, List[str]] = {}
        self._load_atc_index()
    
    def _load_atc_index(self) -> None:
        """Load ATC index from JSON file."""
        if not self.atc_index_path.exists():
            logger.warning(f"ATC index not found at {self.atc_index_path}")
            return
        
        try:
            with open(self.atc_index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.atc_index = data.get("hierarchy", {})
                self.drug_mappings = data.get("drug_mappings", {})
            logger.info(f"Loaded ATC index with {len(self.atc_index)} categories")
        except Exception as e:
            logger.error(f"Error loading ATC index: {e}", exc_info=True)
    
    def match_drug_to_atc(
        self,
        drug_id: str,
        structured_data: Dict[str, Any]
    ) -> List[Tuple[str, float, str]]:
        """
        Match a drug to ATC codes using name and active ingredient matching.
        
        Args:
            drug_id: Drug identifier (e.g., "Activelle_SmPC")
            structured_data: Structured JSON data for the drug
            
        Returns:
            List of tuples: (atc_code, confidence, matched_by)
            - atc_code: ATC code string
            - confidence: Confidence score 0.0-1.0
            - matched_by: "name", "ingredient", or "both"
        """
        matches = []
        
        # Extract drug name from structured data
        drug_name = self._extract_drug_name(structured_data)
        
        # Extract active ingredients
        active_ingredients = self._extract_active_ingredients(structured_data)
        
        # Try name matching first
        name_matches = self._match_by_name(drug_id, drug_name)
        matches.extend(name_matches)
        
        # Try ingredient matching
        ingredient_matches = self._match_by_ingredients(active_ingredients)
        matches.extend(ingredient_matches)
        
        # Deduplicate and sort by confidence
        unique_matches = self._deduplicate_matches(matches)
        unique_matches.sort(key=lambda x: x[1], reverse=True)
        
        return unique_matches
    
    def _extract_drug_name(self, structured_data: Dict[str, Any]) -> Optional[str]:
        """Extract drug name from structured data."""
        # Try section 1 (name)
        sections = structured_data.get("sections", {})
        section_1 = sections.get("1")
        
        if section_1:
            text = section_1.get("text", "")
            # Extract drug name from text (remove formatting)
            # Example: "**Activelle 1 mg/0,5 mg filmuhúðaðar töflur. **"
            # Extract "Activelle"
            text = text.replace("**", "").strip()
            # Take first word/phrase before numbers or "mg"
            parts = re.split(r'\d+|mg|filmuhúðaðar|töflur', text, flags=re.IGNORECASE)
            if parts:
                name = parts[0].strip()
                if name:
                    return name
        
        # Fallback to drug_id
        drug_id = structured_data.get("drug_id", "")
        if drug_id:
            # Remove "_SmPC" suffix if present
            name = drug_id.replace("_SmPC", "").replace("_", " ")
            return name
        
        return None
    
    def _extract_active_ingredients(self, structured_data: Dict[str, Any]) -> List[str]:
        """Extract active ingredients from structured data."""
        ingredients = []
        sections = structured_data.get("sections", {})
        
        # Try ingredients_summary first
        ingredients_summary = sections.get("ingredients_summary")
        if ingredients_summary:
            text = ingredients_summary.get("text", "")
            # Extract active ingredients from summary
            # Look for "Virk efni:" section
            if "Virk efni:" in text or "virk efni" in text.lower():
                # Extract lines after "Virk efni:"
                lines = text.split('\n')
                found_marker = False
                for line in lines:
                    if "virk efni" in line.lower():
                        found_marker = True
                        continue
                    if found_marker and line.strip():
                        # Clean up ingredient name
                        ingredient = self._clean_ingredient_name(line)
                        if ingredient:
                            ingredients.append(ingredient)
        
        # Fallback to section 2 (composition)
        if not ingredients:
            section_2 = sections.get("2")
            if section_2:
                text = section_2.get("text", "")
                # Extract active ingredients before "Hjálparefni"
                lines = text.split('\n')
                for line in lines:
                    if "hjálparefni" in line.lower():
                        break
                    # Extract ingredient names
                    ingredient = self._extract_ingredient_from_line(line)
                    if ingredient:
                        ingredients.append(ingredient)
        
        return ingredients
    
    def _clean_ingredient_name(self, text: str) -> Optional[str]:
        """Clean and extract ingredient name from text."""
        # Remove markdown formatting
        text = re.sub(r'\*\*', '', text)
        text = text.strip()
        
        # Remove common prefixes and suffixes
        text = re.sub(r'^[-•]\s*', '', text)
        text = re.sub(r'\s*\(.*?\)', '', text)  # Remove parenthetical content
        text = re.sub(r'\d+[.,]\d*\s*(mg|g|ml|%)', '', text, flags=re.IGNORECASE)
        
        # Take first meaningful word/phrase
        words = text.split()
        if words:
            # Skip very short words
            meaningful_words = [w for w in words if len(w) > 3]
            if meaningful_words:
                return meaningful_words[0]
        
        return None
    
    def _extract_ingredient_from_line(self, line: str) -> Optional[str]:
        """Extract ingredient name from a line of text."""
        # Pattern: "X mg af [ingredient]" or "[ingredient] (X mg)"
        match = re.search(r'(?:mg|g)\s+af\s+([a-záéíóúýþæö]+)', line.lower())
        if match:
            return match.group(1).strip()
        
        # Alternative: extract word before "mg" or numbers
        cleaned = re.sub(r'\d+[.,]\d*\s*(mg|g|ml|%)', '', line, flags=re.IGNORECASE)
        cleaned = re.sub(r'\([^)]*\)', '', cleaned)
        cleaned = re.sub(r'\s+(jafngildir|sem|er|innan|í|af)', '', cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip('.,;:()[]')
        
        words = cleaned.split()
        for word in words:
            if len(word) > 4 and word.lower() not in ['innihaldslýsing', 'innihald']:
                return word
        
        return None
    
    def _match_by_name(
        self,
        drug_id: str,
        drug_name: Optional[str]
    ) -> List[Tuple[str, float, str]]:
        """Match drug by name."""
        matches = []
        
        if not drug_name:
            return matches
        
        norm_drug_name = normalize(drug_name)
        norm_drug_id = normalize(drug_id)
        
        # Search in drug_mappings
        for mapped_name, atc_codes in self.drug_mappings.items():
            norm_mapped = normalize(mapped_name)
            
            # Exact match
            if norm_drug_name == norm_mapped or norm_drug_id == norm_mapped:
                for atc_code in atc_codes:
                    matches.append((atc_code, 1.0, "name"))
            
            # Partial match (drug name contains mapped name or vice versa)
            elif norm_drug_name in norm_mapped or norm_mapped in norm_drug_name:
                confidence = min(len(norm_drug_name), len(norm_mapped)) / max(len(norm_drug_name), len(norm_mapped))
                for atc_code in atc_codes:
                    matches.append((atc_code, confidence * 0.8, "name"))
        
        return matches
    
    def _match_by_ingredients(
        self,
        active_ingredients: List[str]
    ) -> List[Tuple[str, float, str]]:
        """Match drug by active ingredients."""
        matches = []
        
        if not active_ingredients:
            return matches
        
        # Normalize ingredients
        norm_ingredients = [normalize(ing) for ing in active_ingredients]
        
        # Search in drug_mappings (assuming ingredient names might be in mappings)
        # This is a simplified approach - in practice, you'd need a separate
        # ingredient-to-ATC mapping or search in the ATC hierarchy descriptions
        
        # For now, return empty - can be enhanced with ingredient database
        return matches
    
    def _deduplicate_matches(
        self,
        matches: List[Tuple[str, float, str]]
    ) -> List[Tuple[str, float, str]]:
        """Deduplicate matches, keeping highest confidence for each ATC code."""
        seen = {}
        
        for atc_code, confidence, matched_by in matches:
            if atc_code not in seen:
                seen[atc_code] = (atc_code, confidence, matched_by)
            else:
                # Keep match with higher confidence
                _, existing_conf, existing_by = seen[atc_code]
                if confidence > existing_conf:
                    # If both matches exist, mark as "both"
                    if matched_by != existing_by:
                        matched_by = "both"
                    seen[atc_code] = (atc_code, confidence, matched_by)
        
        return list(seen.values())


def match_drug_to_atc(
    drug_id: str,
    structured_data: Dict[str, Any],
    atc_index_path: Optional[Path] = None
) -> List[Tuple[str, float, str]]:
    """
    Convenience function to match a drug to ATC codes.
    
    Args:
        drug_id: Drug identifier
        structured_data: Structured JSON data
        atc_index_path: Optional path to ATC index
        
    Returns:
        List of (atc_code, confidence, matched_by) tuples
    """
    matcher = ATCMatcher(atc_index_path)
    return matcher.match_drug_to_atc(drug_id, structured_data)
