"""Deep analysis of differences between ATC index and structured data - improved matching."""
import json
import logging
import re
from pathlib import Path
from typing import Dict, Set, List, Any, Tuple
from collections import defaultdict

from config import Config
from src.drug_utils import normalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def strip_suffixes(drug_name: str) -> str:
    """
    Strip common suffixes from drug names for matching.
    
    Removes patterns like "_SmPC", "SmPC", "_Smpc", etc.
    """
    # Remove common suffixes
    suffixes = [
        r'_SmPC$',
        r'_Smpc$',
        r'_smpc$',
        r'SmPC$',
        r'Smpc$',
        r'_SmPC\.json$',
        r'_Smpc\.json$',
    ]
    
    result = drug_name
    for suffix in suffixes:
        result = re.sub(suffix, '', result, flags=re.IGNORECASE)
    
    return result.strip()


def normalize_for_matching(drug_name: str) -> str:
    """
    Normalize drug name for matching, stripping suffixes first.
    
    This allows matching "Actilyse" with "Actilyse_SmPC".
    """
    base_name = strip_suffixes(drug_name)
    return normalize(base_name)


def load_atc_drugs(atc_index_path: Path) -> Dict[str, Any]:
    """Load all drugs from ATC index."""
    logger.info(f"Loading ATC index from {atc_index_path}")
    
    with open(atc_index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    drug_mappings = data.get("drug_mappings", {})
    hierarchy = data.get("hierarchy", {})
    
    # Extract drugs from drug_mappings
    atc_drugs = set(drug_mappings.keys())
    
    # Also extract from hierarchy
    hierarchy_drugs = set()
    atc_codes_by_drug = defaultdict(set)
    
    def extract_drugs_from_level(level_data: Dict[str, Any], atc_code: str = "") -> None:
        """Recursively extract drug names from hierarchy."""
        if isinstance(level_data, dict):
            if "drugs" in level_data:
                for drug_name, drug_info in level_data["drugs"].items():
                    hierarchy_drugs.add(drug_name)
                    drug_atc = drug_info.get("atc_code", atc_code)
                    if drug_atc:
                        atc_codes_by_drug[drug_name].add(drug_atc)
            
            current_code = level_data.get("code", atc_code)
            for key, value in level_data.items():
                if key not in ["code", "name"]:
                    extract_drugs_from_level(value, current_code)
    
    for level1_data in hierarchy.values():
        extract_drugs_from_level(level1_data)
    
    all_atc_drugs = atc_drugs.union(hierarchy_drugs)
    
    # Create normalized versions for matching
    normalized_to_original = {}
    for drug in all_atc_drugs:
        norm = normalize_for_matching(drug)
        if norm not in normalized_to_original:
            normalized_to_original[norm] = []
        normalized_to_original[norm].append(drug)
    
    logger.info(f"Found {len(all_atc_drugs)} unique drugs in ATC index")
    
    return {
        "drugs": sorted(list(all_atc_drugs)),
        "normalized_map": normalized_to_original,
        "atc_codes": dict(atc_codes_by_drug),
        "drug_mappings": drug_mappings
    }


def load_structured_drugs(structured_dir: Path) -> Dict[str, Any]:
    """Load all drugs from structured data directory."""
    logger.info(f"Loading structured data from {structured_dir}")
    
    if not structured_dir.exists():
        return {
            "drugs": [],
            "normalized_map": {},
            "metadata": {}
        }
    
    json_files = list(structured_dir.glob("*_SmPC.json"))
    
    structured_drugs = []
    drug_metadata = {}
    normalized_to_original = {}
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            drug_id = data.get("drug_id", json_file.stem.replace("_SmPC", ""))
            structured_drugs.append(drug_id)
            
            # Store metadata
            drug_metadata[drug_id] = {
                "filename": json_file.name,
                "source_pdf": data.get("source_pdf", ""),
                "has_sections": "sections" in data and bool(data.get("sections")),
                "section_count": len(data.get("sections", {})),
                "atc_codes": data.get("atc_codes", [])
            }
            
            # Create normalized version for matching (strip suffixes)
            norm = normalize_for_matching(drug_id)
            if norm not in normalized_to_original:
                normalized_to_original[norm] = []
            normalized_to_original[norm].append(drug_id)
            
        except Exception as e:
            logger.warning(f"Error reading {json_file.name}: {e}")
    
    logger.info(f"Found {len(structured_drugs)} unique drugs in structured data")
    
    return {
        "drugs": sorted(structured_drugs),
        "normalized_map": normalized_to_original,
        "metadata": drug_metadata
    }


def find_matches(
    atc_drugs: List[str],
    structured_drugs: List[str],
    atc_normalized: Dict[str, List[str]],
    structured_normalized: Dict[str, List[str]]
) -> Dict[str, Any]:
    """Find matches between ATC and structured drugs using improved normalization."""
    atc_set = set(atc_drugs)
    structured_set = set(structured_drugs)
    
    # Direct matches (exact)
    direct_matches = atc_set.intersection(structured_set)
    
    # Normalized matches
    atc_normalized_set = set(atc_normalized.keys())
    structured_normalized_set = set(structured_normalized.keys())
    normalized_intersection = atc_normalized_set.intersection(structured_normalized_set)
    
    # Build mapping of normalized matches
    normalized_matches = []
    matched_atc = set()
    matched_structured = set()
    
    for norm_key in normalized_intersection:
        atc_originals = atc_normalized[norm_key]
        structured_originals = structured_normalized[norm_key]
        
        # Check if any are direct matches
        direct_intersection = set(atc_originals) & structured_set
        if direct_intersection:
            matched_atc.update(direct_intersection)
            matched_structured.update(direct_intersection)
        else:
            # Fuzzy match - same normalized form
            for atc_drug in atc_originals:
                for struct_drug in structured_originals:
                    normalized_matches.append({
                        "atc": atc_drug,
                        "structured": struct_drug,
                        "normalized": norm_key
                    })
                    matched_atc.add(atc_drug)
                    matched_structured.add(struct_drug)
    
    # Add direct matches
    matched_atc.update(direct_matches)
    matched_structured.update(direct_matches)
    
    only_in_atc = atc_set - matched_atc
    only_in_structured = structured_set - matched_structured
    
    return {
        "direct_matches": sorted(list(direct_matches)),
        "normalized_matches": normalized_matches,
        "only_in_atc": sorted(list(only_in_atc)),
        "only_in_structured": sorted(list(only_in_structured)),
        "match_counts": {
            "direct": len(direct_matches),
            "normalized": len(normalized_matches),
            "total_matched_atc": len(matched_atc),
            "total_matched_structured": len(matched_structured),
            "only_in_atc": len(only_in_atc),
            "only_in_structured": len(only_in_structured)
        }
    }


def analyze_structured_only(
    only_in_structured: List[str],
    structured_metadata: Dict[str, Any],
    normalized_matches: List[Dict[str, str]]
) -> Dict[str, Any]:
    """Analyze why drugs are only in structured data."""
    analysis = {
        "veterinary_drugs": [],
        "has_atc_codes": [],
        "no_atc_codes": [],
        "likely_human_drugs": [],
        "naming_patterns": defaultdict(list),
        "matched_drugs": set()
    }
    
    # Track which drugs were matched
    for match in normalized_matches:
        analysis["matched_drugs"].add(match["structured"])
    
    for drug in only_in_structured:
        # Skip if this was matched
        if drug in analysis["matched_drugs"]:
            continue
            
        metadata = structured_metadata.get(drug, {})
        drug_lower = drug.lower()
        
        # Check if veterinary
        if "vet" in drug_lower or "veterinary" in drug_lower:
            analysis["veterinary_drugs"].append(drug)
        
        # Check ATC codes
        atc_codes = metadata.get("atc_codes", [])
        if atc_codes:
            analysis["has_atc_codes"].append({
                "drug": drug,
                "atc_codes": atc_codes
            })
        else:
            analysis["no_atc_codes"].append(drug)
        
        # Check naming patterns
        if "_SmPC" in drug or "SmPC" in drug:
            analysis["naming_patterns"]["has_smpc_suffix"].append(drug)
        if "_" in drug:
            analysis["naming_patterns"]["has_underscores"].append(drug)
        if "-" in drug:
            analysis["naming_patterns"]["has_dashes"].append(drug)
        if any(x in drug for x in ["Heilsa", "Lyfjaver", "Abacus", "Mylan", "Normon", "Alvogen"]):
            analysis["naming_patterns"]["has_distributor"].append(drug)
    
    # Likely human drugs (not veterinary, has sections, not matched)
    for drug in only_in_structured:
        if drug not in analysis["veterinary_drugs"] and drug not in analysis["matched_drugs"]:
            metadata = structured_metadata.get(drug, {})
            if metadata.get("has_sections"):
                analysis["likely_human_drugs"].append(drug)
    
    return analysis


def main() -> None:
    """Main analysis function."""
    logger.info("=" * 80)
    logger.info("DEEP ANALYSIS: ATC INDEX vs STRUCTURED DATA (Improved Matching)")
    logger.info("=" * 80)
    
    # Load data
    atc_data = load_atc_drugs(Config.ATC_INDEX_PATH)
    structured_data = load_structured_drugs(Config.STRUCTURED_DIR)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("BASIC COUNTS")
    logger.info("=" * 80)
    logger.info(f"ATC Index:           {len(atc_data['drugs']):>6} drugs")
    logger.info(f"Structured Data:    {len(structured_data['drugs']):>6} drugs")
    logger.info(f"Difference:         {len(structured_data['drugs']) - len(atc_data['drugs']):>6} more in structured")
    logger.info("")
    
    # Find matches
    logger.info("=" * 80)
    logger.info("MATCHING ANALYSIS (with suffix stripping)")
    logger.info("=" * 80)
    matches = find_matches(
        atc_data["drugs"],
        structured_data["drugs"],
        atc_data["normalized_map"],
        structured_data["normalized_map"]
    )
    
    logger.info(f"Direct matches (exact):        {matches['match_counts']['direct']:>6}")
    logger.info(f"Normalized matches (fuzzy):    {matches['match_counts']['normalized']:>6}")
    logger.info(f"Total ATC drugs matched:       {matches['match_counts']['total_matched_atc']:>6} ({matches['match_counts']['total_matched_atc']/len(atc_data['drugs'])*100:.1f}%)")
    logger.info(f"Total Structured drugs matched: {matches['match_counts']['total_matched_structured']:>6} ({matches['match_counts']['total_matched_structured']/len(structured_data['drugs'])*100:.1f}%)")
    logger.info(f"Only in ATC (unmatched):       {matches['match_counts']['only_in_atc']:>6}")
    logger.info(f"Only in Structured (unmatched): {matches['match_counts']['only_in_structured']:>6}")
    logger.info("")
    
    # Show sample matches
    if matches["normalized_matches"]:
        logger.info("Sample normalized matches (same drug, different names):")
        for match in matches["normalized_matches"][:30]:
            logger.info(f"  ATC: '{match['atc']:35}' <-> Structured: '{match['structured']}'")
        if len(matches["normalized_matches"]) > 30:
            logger.info(f"  ... and {len(matches['normalized_matches']) - 30} more")
        logger.info("")
    
    # Analyze why drugs are only in structured
    logger.info("=" * 80)
    logger.info("ANALYSIS: WHY DRUGS ARE ONLY IN STRUCTURED DATA")
    logger.info("=" * 80)
    structured_only_analysis = analyze_structured_only(
        matches["only_in_structured"],
        structured_data["metadata"],
        matches["normalized_matches"]
    )
    
    logger.info(f"Veterinary drugs:             {len(structured_only_analysis['veterinary_drugs']):>6}")
    logger.info(f"Has ATC codes in JSON:         {len(structured_only_analysis['has_atc_codes']):>6}")
    logger.info(f"No ATC codes in JSON:         {len(structured_only_analysis['no_atc_codes']):>6}")
    logger.info(f"Likely human drugs (unmatched): {len(structured_only_analysis['likely_human_drugs']):>6}")
    logger.info("")
    
    # Show samples
    if structured_only_analysis["veterinary_drugs"]:
        logger.info(f"Veterinary drugs ({len(structured_only_analysis['veterinary_drugs'])}):")
        for drug in sorted(structured_only_analysis["veterinary_drugs"])[:20]:
            logger.info(f"  - {drug}")
        if len(structured_only_analysis["veterinary_drugs"]) > 20:
            logger.info(f"  ... and {len(structured_only_analysis['veterinary_drugs']) - 20} more")
        logger.info("")
    
    # Drugs with ATC codes but not in ATC index
    if structured_only_analysis["has_atc_codes"]:
        logger.info(f"Drugs in structured data with ATC codes but not in ATC index ({len(structured_only_analysis['has_atc_codes'])}):")
        for item in structured_only_analysis["has_atc_codes"][:20]:
            logger.info(f"  - {item['drug']}: {item['atc_codes']}")
        if len(structured_only_analysis["has_atc_codes"]) > 20:
            logger.info(f"  ... and {len(structured_only_analysis['has_atc_codes']) - 20} more")
        logger.info("")
    
    # Sample unmatched human drugs
    if structured_only_analysis["likely_human_drugs"]:
        logger.info(f"Sample unmatched human drugs ({len(structured_only_analysis['likely_human_drugs'])}):")
        for drug in sorted(structured_only_analysis["likely_human_drugs"])[:30]:
            logger.info(f"  - {drug}")
        if len(structured_only_analysis["likely_human_drugs"]) > 30:
            logger.info(f"  ... and {len(structured_only_analysis['likely_human_drugs']) - 30} more")
        logger.info("")
    
    # Drugs only in ATC
    if matches["only_in_atc"]:
        logger.info(f"Drugs only in ATC index (not in structured data) ({len(matches['only_in_atc'])}):")
        for drug in matches["only_in_atc"][:30]:
            logger.info(f"  - {drug}")
        if len(matches["only_in_atc"]) > 30:
            logger.info(f"  ... and {len(matches['only_in_atc']) - 30} more")
        logger.info("")
    
    # Save detailed report
    report = {
        "summary": {
            "atc_count": len(atc_data["drugs"]),
            "structured_count": len(structured_data["drugs"]),
            "difference": len(structured_data["drugs"]) - len(atc_data["drugs"]),
            "matched_atc": matches["match_counts"]["total_matched_atc"],
            "matched_structured": matches["match_counts"]["total_matched_structured"],
            "unmatched_atc": matches["match_counts"]["only_in_atc"],
            "unmatched_structured": matches["match_counts"]["only_in_structured"]
        },
        "matching": matches,
        "structured_only_analysis": {
            "veterinary_count": len(structured_only_analysis["veterinary_drugs"]),
            "has_atc_codes_count": len(structured_only_analysis["has_atc_codes"]),
            "likely_human_drugs_count": len(structured_only_analysis["likely_human_drugs"]),
            "sample_veterinary": structured_only_analysis["veterinary_drugs"][:50],
            "sample_human": structured_only_analysis["likely_human_drugs"][:100]
        },
        "only_in_atc": matches["only_in_atc"][:100]
    }
    
    report_path = Path("atc_vs_structured_analysis_v2.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Detailed report saved to: {report_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

