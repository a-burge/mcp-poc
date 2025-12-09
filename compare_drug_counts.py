"""Compare drug counts between ATC index, vector store, and structured data."""
import json
import logging
from pathlib import Path
from typing import Dict, Set, List, Any

from config import Config
from src.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def count_drugs_in_atc_index(atc_index_path: Path) -> Dict[str, Any]:
    """
    Count unique drugs in ATC index.
    
    Args:
        atc_index_path: Path to atc_index.json
        
    Returns:
        Dictionary with count and drug names
    """
    logger.info(f"Loading ATC index from {atc_index_path}")
    
    with open(atc_index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract drugs from drug_mappings (most reliable source)
    drug_mappings = data.get("drug_mappings", {})
    atc_drugs = set(drug_mappings.keys())
    
    # Also extract drugs from hierarchy (in case some are missing from mappings)
    hierarchy = data.get("hierarchy", {})
    hierarchy_drugs = set()
    
    def extract_drugs_from_level(level_data: Dict[str, Any]) -> None:
        """Recursively extract drug names from hierarchy."""
        if isinstance(level_data, dict):
            # Check if this level has a "drugs" key
            if "drugs" in level_data:
                for drug_name in level_data["drugs"].keys():
                    hierarchy_drugs.add(drug_name)
            
            # Recursively process nested levels
            for key, value in level_data.items():
                if key not in ["code", "name"]:  # Skip metadata fields
                    extract_drugs_from_level(value)
    
    # Extract from all levels
    for level1_data in hierarchy.values():
        extract_drugs_from_level(level1_data)
    
    # Combine both sources
    all_atc_drugs = atc_drugs.union(hierarchy_drugs)
    
    logger.info(f"Found {len(drug_mappings)} drugs in drug_mappings")
    logger.info(f"Found {len(hierarchy_drugs)} drugs in hierarchy")
    logger.info(f"Total unique drugs in ATC index: {len(all_atc_drugs)}")
    
    return {
        "count": len(all_atc_drugs),
        "drugs": sorted(all_atc_drugs),
        "from_mappings": sorted(list(atc_drugs)),
        "from_hierarchy": sorted(list(hierarchy_drugs)),
        "mappings_count": len(drug_mappings),
        "hierarchy_count": len(hierarchy_drugs)
    }


def count_drugs_in_vector_store() -> Dict[str, Any]:
    """
    Count unique drugs in vector store.
    
    Returns:
        Dictionary with count and drug names
    """
    logger.info("Initializing vector store...")
    vector_store = VectorStoreManager()
    
    logger.info("Getting all drug IDs from vector store...")
    vector_store_drugs = vector_store.get_all_drug_ids()
    
    logger.info(f"Found {len(vector_store_drugs)} unique drugs in vector store")
    
    return {
        "count": len(vector_store_drugs),
        "drugs": sorted(vector_store_drugs)
    }


def count_drugs_in_structured_data(structured_dir: Path) -> Dict[str, Any]:
    """
    Count unique drugs in structured data directory.
    
    Args:
        structured_dir: Path to structured data directory
        
    Returns:
        Dictionary with count and drug names
    """
    logger.info(f"Scanning structured data directory: {structured_dir}")
    
    if not structured_dir.exists():
        logger.warning(f"Structured directory does not exist: {structured_dir}")
        return {
            "count": 0,
            "drugs": []
        }
    
    # Find all JSON files matching the pattern {drug_id}_SmPC.json
    json_files = list(structured_dir.glob("*_SmPC.json"))
    
    # Extract drug IDs from JSON files (more reliable than filename parsing)
    structured_drugs = set()
    drug_id_from_file = set()  # For comparison
    
    for json_file in json_files:
        # Method 1: Extract from filename (for comparison)
        filename_drug_id = json_file.stem.replace("_SmPC", "")
        drug_id_from_file.add(filename_drug_id)
        
        # Method 2: Read drug_id from JSON file (more accurate)
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                drug_id = data.get("drug_id", filename_drug_id)
                structured_drugs.add(drug_id)
        except Exception as e:
            logger.warning(f"Could not read drug_id from {json_file.name}: {e}")
            # Fallback to filename-based extraction
            structured_drugs.add(filename_drug_id)
    
    logger.info(f"Found {len(json_files)} JSON files")
    logger.info(f"Found {len(structured_drugs)} unique drugs in structured data (from JSON drug_id field)")
    logger.info(f"Found {len(drug_id_from_file)} unique drugs (from filename parsing)")
    
    return {
        "count": len(structured_drugs),
        "drugs": sorted(list(structured_drugs)),
        "json_files": len(json_files),
        "drugs_from_filename": sorted(list(drug_id_from_file))
    }


def find_discrepancies(
    atc_data: Dict[str, Any],
    vector_store_data: Dict[str, Any],
    structured_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Find discrepancies between the three sources.
    
    Args:
        atc_data: ATC index data
        vector_store_data: Vector store data
        structured_data: Structured data directory data
        
    Returns:
        Dictionary with discrepancy analysis
    """
    atc_drugs = set(atc_data["drugs"])
    vector_store_drugs = set(vector_store_data["drugs"])
    structured_drugs = set(structured_data["drugs"])
    
    # Find drugs in each source but not in others
    only_in_atc = atc_drugs - vector_store_drugs - structured_drugs
    only_in_vector_store = vector_store_drugs - atc_drugs - structured_drugs
    only_in_structured = structured_drugs - atc_drugs - vector_store_drugs
    
    # Find drugs in vector store but not in structured (should be same)
    in_vector_not_structured = vector_store_drugs - structured_drugs
    in_structured_not_vector = structured_drugs - vector_store_drugs
    
    # Find drugs in ATC but not indexed
    in_atc_not_indexed = atc_drugs - vector_store_drugs
    
    # Find drugs indexed but not in ATC
    indexed_not_in_atc = vector_store_drugs - atc_drugs
    
    return {
        "only_in_atc": sorted(list(only_in_atc)),
        "only_in_vector_store": sorted(list(only_in_vector_store)),
        "only_in_structured": sorted(list(only_in_structured)),
        "in_vector_not_structured": sorted(list(in_vector_not_structured)),
        "in_structured_not_vector": sorted(list(in_structured_not_vector)),
        "in_atc_not_indexed": sorted(list(in_atc_not_indexed)),
        "indexed_not_in_atc": sorted(list(indexed_not_in_atc)),
        "counts": {
            "only_in_atc": len(only_in_atc),
            "only_in_vector_store": len(only_in_vector_store),
            "only_in_structured": len(only_in_structured),
            "in_vector_not_structured": len(in_vector_not_structured),
            "in_structured_not_vector": len(in_structured_not_vector),
            "in_atc_not_indexed": len(in_atc_not_indexed),
            "indexed_not_in_atc": len(indexed_not_in_atc)
        }
    }


def main() -> None:
    """Main comparison function."""
    logger.info("=" * 80)
    logger.info("DRUG COUNT COMPARISON")
    logger.info("=" * 80)
    
    # Count drugs in ATC index
    atc_data = count_drugs_in_atc_index(Config.ATC_INDEX_PATH)
    
    # Count drugs in vector store
    vector_store_data = count_drugs_in_vector_store()
    
    # Count drugs in structured data
    structured_data = count_drugs_in_structured_data(Config.STRUCTURED_DIR)
    
    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"ATC Index:           {atc_data['count']:>6} drugs")
    logger.info(f"Vector Store:        {vector_store_data['count']:>6} drugs")
    logger.info(f"Structured Data:     {structured_data['count']:>6} drugs")
    logger.info("")
    
    # Find discrepancies
    discrepancies = find_discrepancies(atc_data, vector_store_data, structured_data)
    
    logger.info("=" * 80)
    logger.info("DISCREPANCIES")
    logger.info("=" * 80)
    logger.info(f"Drugs only in ATC index:                    {discrepancies['counts']['only_in_atc']:>6}")
    logger.info(f"Drugs only in vector store:                 {discrepancies['counts']['only_in_vector_store']:>6}")
    logger.info(f"Drugs only in structured data:               {discrepancies['counts']['only_in_structured']:>6}")
    logger.info(f"Drugs in vector store but not structured:   {discrepancies['counts']['in_vector_not_structured']:>6}")
    logger.info(f"Drugs in structured but not vector store:    {discrepancies['counts']['in_structured_not_vector']:>6}")
    logger.info(f"Drugs in ATC but not indexed:               {discrepancies['counts']['in_atc_not_indexed']:>6}")
    logger.info(f"Drugs indexed but not in ATC:               {discrepancies['counts']['indexed_not_in_atc']:>6}")
    logger.info("")
    
    # Print detailed lists if there are discrepancies
    if discrepancies['counts']['in_atc_not_indexed'] > 0:
        logger.info(f"Drugs in ATC but not indexed ({len(discrepancies['in_atc_not_indexed'])}):")
        for drug in discrepancies['in_atc_not_indexed'][:20]:  # Show first 20
            logger.info(f"  - {drug}")
        if len(discrepancies['in_atc_not_indexed']) > 20:
            logger.info(f"  ... and {len(discrepancies['in_atc_not_indexed']) - 20} more")
        logger.info("")
    
    if discrepancies['counts']['indexed_not_in_atc'] > 0:
        logger.info(f"Drugs indexed but not in ATC ({len(discrepancies['indexed_not_in_atc'])}):")
        for drug in discrepancies['indexed_not_in_atc'][:20]:  # Show first 20
            logger.info(f"  - {drug}")
        if len(discrepancies['indexed_not_in_atc']) > 20:
            logger.info(f"  ... and {len(discrepancies['indexed_not_in_atc']) - 20} more")
        logger.info("")
    
    if discrepancies['counts']['in_vector_not_structured'] > 0:
        logger.info(f"Drugs in vector store but not in structured data ({len(discrepancies['in_vector_not_structured'])}):")
        for drug in discrepancies['in_vector_not_structured'][:20]:  # Show first 20
            logger.info(f"  - {drug}")
        if len(discrepancies['in_vector_not_structured']) > 20:
            logger.info(f"  ... and {len(discrepancies['in_vector_not_structured']) - 20} more")
        logger.info("")
    
    if discrepancies['counts']['in_structured_not_vector'] > 0:
        logger.info(f"Drugs in structured data but not in vector store ({len(discrepancies['in_structured_not_vector'])}):")
        for drug in discrepancies['in_structured_not_vector'][:20]:  # Show first 20
            logger.info(f"  - {drug}")
        if len(discrepancies['in_structured_not_vector']) > 20:
            logger.info(f"  ... and {len(discrepancies['in_structured_not_vector']) - 20} more")
        logger.info("")
    
    # Save detailed report to file
    report = {
        "summary": {
            "atc_index_count": atc_data["count"],
            "vector_store_count": vector_store_data["count"],
            "structured_data_count": structured_data["count"]
        },
        "discrepancies": discrepancies,
        "atc_drugs": atc_data["drugs"],
        "vector_store_drugs": vector_store_data["drugs"],
        "structured_drugs": structured_data["drugs"]
    }
    
    report_path = Path("drug_count_comparison_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Detailed report saved to: {report_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

