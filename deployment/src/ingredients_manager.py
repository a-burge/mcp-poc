"""Ingredients data manager for accessing ingredient-to-drug mappings and generating RAG context."""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

from config import Config
from src.query_disambiguation import normalize_icelandic, strip_diacritics

logger = logging.getLogger(__name__)


class IngredientsManager:
    """Manages ingredients data access and provides utilities for RAG integration."""
    
    def __init__(
        self,
        ingredients_index_path: Optional[Path] = None
    ):
        """
        Initialize Ingredients manager.
        
        Args:
            ingredients_index_path: Path to ingredients index JSON file
        """
        if ingredients_index_path is None:
            ingredients_index_path = Config.INGREDIENTS_INDEX_PATH
        
        self.ingredients_index_path = ingredients_index_path
        self.ingredients: Dict[str, Any] = {}
        self.drug_to_ingredients: Dict[str, List[str]] = {}
        self._load_data()
    
    def _load_data(self) -> None:
        """Load ingredients index."""
        if self.ingredients_index_path.exists():
            try:
                with open(self.ingredients_index_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.ingredients = data.get("ingredients", {})
                    self.drug_to_ingredients = data.get("drug_to_ingredients", {})
                
                # Normalize ingredient keys so lookups are consistent
                normalized_index = {}
                for key, data in self.ingredients.items():
                    inn_name = data.get("inn_name") or key
                    norm = self._norm_ingredient_name(inn_name)
                    normalized_index[norm] = data
                self.ingredients = normalized_index
                
                # Also normalize drug_to_ingredients
                normalized_map = {}
                for drug, ing_list in self.drug_to_ingredients.items():
                    norm_ing_list = [self._norm_ingredient_name(x) for x in ing_list if x]
                    normalized_map[drug] = norm_ing_list
                self.drug_to_ingredients = normalized_map
                
                logger.info(
                    f"Loaded ingredients index with {len(self.ingredients)} ingredients "
                    f"and {len(self.drug_to_ingredients)} drug mappings"
                )
            except Exception as e:
                logger.error(f"Error loading ingredients index: {e}", exc_info=True)
        else:
            logger.warning(f"Ingredients index not found at {self.ingredients_index_path}")
    
    def get_ingredients_for_drug(self, drug_id: str) -> List[str]:
        """
        Get active ingredients (INN names) for a drug.
        
        Args:
            drug_id: Drug identifier (brand name)
            
        Returns:
            List of ingredient names (INN names)
        """
        # Normalize drug_id for lookup (remove common suffixes)
        normalized_id = self._normalize_drug_id(drug_id)
        
        # Try direct lookup
        ingredients = self.drug_to_ingredients.get(drug_id, [])
        if ingredients:
            return ingredients
        
        # Try normalized lookup
        ingredients = self.drug_to_ingredients.get(normalized_id, [])
        if ingredients:
            return ingredients
        
        # Try partial matching with consistent normalization
        norm = normalized_id.lower()
        for mapped_drug, mapped_ingredients in self.drug_to_ingredients.items():
            mapped = self._normalize_drug_id(mapped_drug).lower()
            if norm == mapped or norm in mapped or mapped in norm:
                return mapped_ingredients
        
        return []
    
    def get_drugs_by_ingredient(self, ingredient_name: str) -> List[str]:
        """
        Get all drugs containing a specific active ingredient.
        
        Args:
            ingredient_name: Ingredient name (INN name)
            
        Returns:
            List of drug names (brand names)
        """
        # Normalize ingredient name for lookup
        norm_ing = self._norm_ingredient_name(ingredient_name)
        
        matching_drugs = []
        
        # Search in ingredients dict (now normalized)
        for ing_key, ing_data in self.ingredients.items():
            inn_name = ing_data.get("inn_name", "")
            inn_norm = self._norm_ingredient_name(inn_name)
            
            if norm_ing == inn_norm or norm_ing in inn_norm or inn_norm in norm_ing:
                drugs = ing_data.get("drugs", {})
                matching_drugs.extend(list(drugs.keys()))
        
        # Also check reverse mapping (now normalized)
        for drug, ingredients in self.drug_to_ingredients.items():
            for ing in ingredients:
                if norm_ing == ing or norm_ing in ing or ing in norm_ing:
                    matching_drugs.append(drug)
        
        return sorted(set(matching_drugs))
    
    def get_generic_alternatives(self, drug_id: str, max_results: int = 10) -> List[str]:
        """
        Get generic alternatives (same active ingredient, different brand).
        
        Args:
            drug_id: Drug identifier
            max_results: Maximum number of alternatives to return
            
        Returns:
            List of alternative drug names
        """
        # Get ingredients for this drug
        ingredients = self.get_ingredients_for_drug(drug_id)
        
        if not ingredients:
            return []
        
        # Find all drugs with same ingredients
        alternatives = []
        for ingredient in ingredients:
            drugs_with_ingredient = self.get_drugs_by_ingredient(ingredient)
            # Exclude the original drug
            alternatives.extend([
                drug for drug in drugs_with_ingredient 
                if drug.lower() != drug_id.lower()
            ])
        
        # Remove duplicates and limit
        alternatives = list(set(alternatives))[:max_results]
        return alternatives
    
    def format_ingredients_context_for_rag(
        self,
        drug_id: str,
        include_alternatives: bool = True
    ) -> str:
        """
        Format ingredients information as context for RAG prompts.
        
        Args:
            drug_id: Drug identifier
            include_alternatives: If True, include generic alternatives
            
        Returns:
            Formatted string with ingredients context
        """
        ingredients = self.get_ingredients_for_drug(drug_id)
        
        if not ingredients:
            return ""
        
        context_parts = []
        context_parts.append(f"Virk innihaldsefni fyrir {drug_id}:")
        
        for ingredient in ingredients:
            context_parts.append(f"  - {ingredient}")
        
        # Include generic alternatives if requested
        if include_alternatives:
            alternatives = self.get_generic_alternatives(drug_id, max_results=5)
            if alternatives:
                context_parts.append(f"\nAðrar vörumerki með sama virka innihaldsefni:")
                for alt_drug in alternatives:
                    context_parts.append(f"  - {alt_drug}")
        
        return "\n".join(context_parts)
    
    def find_drugs_by_ingredient_query(self, query: str) -> List[str]:
        """
        Find drugs by matching ingredient name in query.
        
        Useful for queries like "Hvað er Ibuprofen?" where user asks about
        ingredient name rather than brand name.
        
        Args:
            query: User query text
            
        Returns:
            List of matching drug names
        """
        query_lower = query.lower()
        matching_drugs = []
        
        # Search for ingredient names in query
        for ing_key, ing_data in self.ingredients.items():
            inn_name = ing_data.get("inn_name", "")
            inn_lower = inn_name.lower()
            
            # Check if ingredient name appears in query
            if inn_lower in query_lower or query_lower in inn_lower:
                # Get all drugs with this ingredient
                drugs = self.get_drugs_by_ingredient(inn_name)
                matching_drugs.extend(drugs)
        
        return list(set(matching_drugs))  # Remove duplicates
    
    def _normalize_drug_id(self, drug_id: str) -> str:
        """
        Normalize drug ID for lookup.
        
        Mirrors the normalization pattern from _normalize_brand_name() in rag_chain_langgraph.py:
        - Applies normalize_icelandic() (NFKC normalization + lowercase)
        - Applies strip_diacritics() (NFKD normalization + remove accents)
        - Removes SmPC suffixes
        - Removes strength patterns (e.g., _400mg, _200mg, _mg/mL, etc.)
        - Normalizes delimiters (underscores → spaces) for consistent matching
        
        This ensures consistent matching with the RAG pipeline normalization.
        
        Args:
            drug_id: Drug identifier
            
        Returns:
            Normalized drug identifier (lowercase, no accents, no suffixes)
        """
        if not drug_id:
            return drug_id
        
        # Apply full normalization chain to match RAG pipeline
        normalized = normalize_icelandic(drug_id)  # NFKC + lowercase
        normalized = strip_diacritics(normalized)  # Remove accents
        
        # Remove document type suffixes (normalized is already lowercase at this point)
        for suffix in ["_smpc", "_smpc_smpc"]:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
        
        # Remove strength patterns (e.g., "_200mg", "_50ml", "_mg/mL", "_IU", etc.)
        # Handles: mg, g, ml, µg, mcg, iu, iu/ml, mg/ml, mg/mL, mg/g, %, and decimal numbers
        normalized = re.sub(
            r'_(\d+(\.\d+)?\s*(mg|g|ml|µg|mcg|iu|iu/ml|mg/ml|mg/mL|mg/g|%))',
            '',
            normalized,
            flags=re.IGNORECASE
        )
        
        # Normalize delimiters: convert underscores to spaces for consistent matching
        # This fixes mismatch between vector store (underscores) and ingredients index (spaces)
        # e.g., "dicloxacillin_bluefish" → "dicloxacillin bluefish"
        normalized = normalized.replace("_", " ")
        
        return normalized.strip()
    
    def _norm_ingredient_name(self, name: str) -> str:
        """
        Normalize ingredient names consistently with INN conventions.
        
        Uses the same normalization pipeline as drug IDs plus:
        - Icelandic→INN character mappings (k→c, s[ei]→c[ei], þ→th)
        - Light INN-specific cleanup for Icelandic inflections (ini/inum endings)
        
        This enables matching:
        - díklófenak → diclofenac
        - parasetamól → paracetamol
        
        Args:
            name: Ingredient name (INN name)
            
        Returns:
            Normalized ingredient name
        """
        if not name:
            return ""
        
        # Reuse existing normalization utilities
        s = normalize_icelandic(name)        # NFKC + lowercase
        s = strip_diacritics(s)              # remove accents
        
        # Remove punctuation
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        
        # Apply Icelandic → INN character normalization
        # k → c (diklofenak → diclofenak → diclofenac)
        s = s.replace('k', 'c')
        # Soft c: s before e or i → c (parasetamol → paracetamol)
        s = re.sub(r's([ei])', r'c\1', s)
        # þ → th (already stripped by diacritics, but handle if present)
        s = s.replace('þ', 'th')
        
        # Common Icelandic inflections for antibiotic INNs
        # Example: diklóxacillín → dicloxacillin
        #          diklóxacillíni → dicloxacillin
        #          diklóxacillínum → dicloxacillin
        s = re.sub(r"(in|ini|inum)$", "", s)
        
        return s


def get_ingredients_manager() -> IngredientsManager:
    """Get a singleton IngredientsManager instance."""
    if not hasattr(get_ingredients_manager, '_instance'):
        get_ingredients_manager._instance = IngredientsManager()
    return get_ingredients_manager._instance

