"""ATC data manager for accessing ATC hierarchy and generating RAG context."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from config import Config

logger = logging.getLogger(__name__)


class ATCManager:
    """Manages ATC data access and provides utilities for RAG integration."""
    
    def __init__(
        self,
        atc_index_path: Optional[Path] = None,
        drug_mappings_path: Optional[Path] = None
    ):
        """
        Initialize ATC manager.
        
        Args:
            atc_index_path: Path to ATC index JSON file
            drug_mappings_path: Path to drug-to-ATC mappings JSON file
        """
        if atc_index_path is None:
            atc_index_path = Config.ATC_INDEX_PATH
        if drug_mappings_path is None:
            drug_mappings_path = Config.DRUG_ATC_MAPPINGS_PATH
        
        self.atc_index_path = atc_index_path
        self.drug_mappings_path = drug_mappings_path
        self.atc_index: Dict[str, Any] = {}
        self.drug_mappings: Dict[str, Dict[str, Any]] = {}
        self._load_data()
    
    def _load_data(self) -> None:
        """Load ATC index and drug mappings."""
        # Load ATC index
        if self.atc_index_path.exists():
            try:
                with open(self.atc_index_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.atc_index = data.get("hierarchy", {})
                logger.info(f"Loaded ATC index with {len(self.atc_index)} categories")
            except Exception as e:
                logger.error(f"Error loading ATC index: {e}", exc_info=True)
        else:
            logger.warning(f"ATC index not found at {self.atc_index_path}")
        
        # Load drug mappings
        if self.drug_mappings_path.exists():
            try:
                with open(self.drug_mappings_path, 'r', encoding='utf-8') as f:
                    self.drug_mappings = json.load(f)
                logger.info(f"Loaded {len(self.drug_mappings)} drug-to-ATC mappings")
            except Exception as e:
                logger.error(f"Error loading drug mappings: {e}", exc_info=True)
        else:
            logger.warning(f"Drug mappings not found at {self.drug_mappings_path}")
    
    def get_atc_codes_for_drug(self, drug_id: str) -> List[str]:
        """
        Get ATC codes for a specific drug.
        
        Args:
            drug_id: Drug identifier
            
        Returns:
            List of ATC codes
        """
        mapping = self.drug_mappings.get(drug_id, {})
        return mapping.get("atc_codes", [])
    
    def get_drugs_by_atc(self, atc_code: str) -> List[str]:
        """
        Get all drugs with a specific ATC code.
        
        Args:
            atc_code: ATC code (can be partial, e.g., "A10BA" or full "A10BA02")
            
        Returns:
            List of drug_ids with matching ATC codes
        """
        matching_drugs = []
        
        for drug_id, mapping in self.drug_mappings.items():
            atc_codes = mapping.get("atc_codes", [])
            # Check if any ATC code starts with the given code (for partial matches)
            if any(code.startswith(atc_code) for code in atc_codes):
                matching_drugs.append(drug_id)
        
        return matching_drugs
    
    def get_atc_hierarchy_path(self, atc_code: str) -> List[Dict[str, str]]:
        """
        Get the full hierarchy path for an ATC code.
        
        Args:
            atc_code: ATC code (e.g., "A10BA02")
            
        Returns:
            List of dictionaries with 'code' and 'name' for each level
            Example: [
                {"code": "A", "name": "Alimentary tract and metabolism"},
                {"code": "A10", "name": "Drugs used in diabetes"},
                ...
            ]
        """
        path = []
        
        if not atc_code:
            return path
        
        # Extract level 1
        level1_code = atc_code[0] if len(atc_code) > 0 else None
        if level1_code and level1_code in self.atc_index:
            level1_data = self.atc_index[level1_code]
            path.append({
                "code": level1_code,
                "name": level1_data.get("name", "")
            })
        
        # Extract level 2 (2 digits)
        if len(atc_code) >= 3:
            level2_code = atc_code[:3]
            # Navigate through hierarchy to find level 2
            if level1_code and level1_code in self.atc_index:
                level2_data = self._find_level_in_hierarchy(
                    self.atc_index[level1_code],
                    level2_code,
                    level=2
                )
                if level2_data:
                    path.append({
                        "code": level2_code,
                        "name": level2_data.get("name", "")
                    })
        
        # Extract level 3 (1 letter)
        if len(atc_code) >= 4:
            level3_code = atc_code[:4]
            level3_data = self._find_level_in_hierarchy(
                self.atc_index.get(level1_code, {}),
                level3_code,
                level=3
            )
            if level3_data:
                path.append({
                    "code": level3_code,
                    "name": level3_data.get("name", "")
                })
        
        # Extract level 4 (2 letters)
        if len(atc_code) >= 6:
            level4_code = atc_code[:6]
            level4_data = self._find_level_in_hierarchy(
                self.atc_index.get(level1_code, {}),
                level4_code,
                level=4
            )
            if level4_data:
                path.append({
                    "code": level4_code,
                    "name": level4_data.get("name", "")
                })
        
        # Level 5 is the full code
        if len(atc_code) >= 7:
            level5_data = self._find_level_in_hierarchy(
                self.atc_index.get(level1_code, {}),
                atc_code,
                level=5
            )
            if level5_data:
                path.append({
                    "code": atc_code,
                    "name": level5_data.get("name", "")
                })
        
        return path
    
    def _find_level_in_hierarchy(
        self,
        parent_data: Dict[str, Any],
        target_code: str,
        level: int
    ) -> Optional[Dict[str, Any]]:
        """Recursively find a level in the hierarchy."""
        if not parent_data:
            return None
        
        # Check current level
        if parent_data.get("code") == target_code:
            return parent_data
        
        # Check nested levels
        level_key = f"level{level}"
        nested_levels = parent_data.get(level_key, {})
        
        if isinstance(nested_levels, dict):
            for code, data in nested_levels.items():
                if code == target_code:
                    return data
                # Recursively search deeper
                result = self._find_level_in_hierarchy(data, target_code, level)
                if result:
                    return result
        
        return None
    
    def format_atc_context_for_rag(
        self,
        drug_id: str,
        include_alternatives: bool = True
    ) -> str:
        """
        Format ATC information as context for RAG prompts.
        
        Args:
            drug_id: Drug identifier
            include_alternatives: If True, include alternative drugs in same category
            
        Returns:
            Formatted string with ATC context
        """
        atc_codes = self.get_atc_codes_for_drug(drug_id)
        
        if not atc_codes:
            return ""
        
        context_parts = []
        context_parts.append(f"ATC flokkun fyrir {drug_id}:")
        
        for atc_code in atc_codes:
            hierarchy_path = self.get_atc_hierarchy_path(atc_code)
            
            if hierarchy_path:
                # Format hierarchy path
                path_str = " > ".join([f"{level['code']} ({level['name']})" for level in hierarchy_path])
                context_parts.append(f"  - {atc_code}: {path_str}")
            else:
                context_parts.append(f"  - {atc_code}")
        
        # Include alternatives if requested
        if include_alternatives:
            alternatives = []
            for atc_code in atc_codes:
                # Get drugs with same ATC code (excluding current drug)
                same_atc_drugs = [
                    d for d in self.get_drugs_by_atc(atc_code)
                    if d != drug_id
                ]
                alternatives.extend(same_atc_drugs)
            
            if alternatives:
                # Remove duplicates
                alternatives = list(set(alternatives))
                context_parts.append(f"\nAðrar lyf með sama ATC flokk:")
                for alt_drug in alternatives[:10]:  # Limit to 10 alternatives
                    alt_atc = self.get_atc_codes_for_drug(alt_drug)
                    context_parts.append(f"  - {alt_drug} (ATC: {', '.join(alt_atc)})")
        
        return "\n".join(context_parts)
    
    def get_alternatives(self, drug_id: str, max_results: int = 10) -> List[str]:
        """
        Get alternative drugs in the same ATC category.
        
        Args:
            drug_id: Drug identifier
            max_results: Maximum number of alternatives to return
            
        Returns:
            List of alternative drug_ids
        """
        atc_codes = self.get_atc_codes_for_drug(drug_id)
        
        if not atc_codes:
            return []
        
        alternatives = []
        for atc_code in atc_codes:
            same_atc_drugs = [
                d for d in self.get_drugs_by_atc(atc_code)
                if d != drug_id
            ]
            alternatives.extend(same_atc_drugs)
        
        # Remove duplicates and limit results
        alternatives = list(set(alternatives))[:max_results]
        return alternatives


def get_atc_manager() -> ATCManager:
    """Get a singleton ATCManager instance."""
    if not hasattr(get_atc_manager, '_instance'):
        get_atc_manager._instance = ATCManager()
    return get_atc_manager._instance
