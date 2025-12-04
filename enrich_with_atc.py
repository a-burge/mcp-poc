"""Script to enrich structured JSON files with ATC codes."""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from src.atc_matcher import ATCMatcher
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def enrich_json_file(
    json_path: Path,
    atc_matcher: ATCMatcher,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Enrich a single JSON file with ATC codes.
    
    Args:
        json_path: Path to structured JSON file
        atc_matcher: ATCMatcher instance
        dry_run: If True, don't save changes
        
    Returns:
        Dictionary with enrichment results
    """
    try:
        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        drug_id = data.get("drug_id", json_path.stem)
        
        # Check if already enriched
        if "atc_codes" in data:
            logger.info(f"{drug_id}: Already has ATC codes, skipping")
            return {
                "success": True,
                "drug_id": drug_id,
                "atc_codes": data.get("atc_codes", []),
                "skipped": True
            }
        
        # Match to ATC codes
        matches = atc_matcher.match_drug_to_atc(drug_id, data)
        
        if not matches:
            logger.warning(f"{drug_id}: No ATC codes found")
            return {
                "success": False,
                "drug_id": drug_id,
                "reason": "No ATC codes matched"
            }
        
        # Extract ATC codes (take top matches with confidence > 0.5)
        atc_codes = [
            code for code, confidence, _ in matches
            if confidence > 0.5
        ]
        
        if not atc_codes:
            logger.warning(f"{drug_id}: No high-confidence ATC matches")
            return {
                "success": False,
                "drug_id": drug_id,
                "reason": "No high-confidence matches"
            }
        
        # Add to JSON data
        data["atc_codes"] = atc_codes
        data["atc_matched_by"] = matches[0][2] if matches else "unknown"
        data["atc_confidence"] = matches[0][1] if matches else 0.0
        
        # Save if not dry run
        if not dry_run:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"{drug_id}: Added {len(atc_codes)} ATC codes")
        else:
            logger.info(f"{drug_id}: [DRY RUN] Would add {len(atc_codes)} ATC codes")
        
        return {
            "success": True,
            "drug_id": drug_id,
            "atc_codes": atc_codes,
            "matched_by": matches[0][2] if matches else "unknown",
            "confidence": matches[0][1] if matches else 0.0
        }
        
    except Exception as e:
        logger.error(f"Error enriching {json_path}: {e}", exc_info=True)
        return {
            "success": False,
            "drug_id": json_path.stem,
            "reason": str(e)
        }


def update_drug_mappings(
    enrichment_results: list,
    mappings_path: Optional[Path] = None
) -> None:
    """
    Update drug-to-ATC mappings file with enrichment results.
    
    Args:
        enrichment_results: List of enrichment result dictionaries
        mappings_path: Path to mappings file
    """
    if mappings_path is None:
        mappings_path = Config.DRUG_ATC_MAPPINGS_PATH
    
    Config.ensure_directories()
    
    # Load existing mappings
    existing_mappings = {}
    if mappings_path.exists():
        try:
            with open(mappings_path, 'r', encoding='utf-8') as f:
                existing_mappings = json.load(f)
        except Exception as e:
            logger.warning(f"Error loading existing mappings: {e}")
    
    # Update with new results
    for result in enrichment_results:
        if result.get("success") and not result.get("skipped"):
            drug_id = result["drug_id"]
            existing_mappings[drug_id] = {
                "atc_codes": result.get("atc_codes", []),
                "matched_by": result.get("matched_by", "unknown"),
                "confidence": result.get("confidence", 0.0)
            }
    
    # Save updated mappings
    with open(mappings_path, 'w', encoding='utf-8') as f:
        json.dump(existing_mappings, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Updated drug mappings file with {len(existing_mappings)} entries")


def main(
    json_path: Optional[Path] = None,
    dry_run: bool = False,
    update_mappings: bool = True
) -> None:
    """
    Main function to enrich JSON files with ATC codes.
    
    Args:
        json_path: Path to single JSON file (if None, processes all files)
        dry_run: If True, don't save changes
        update_mappings: If True, update drug mappings file
    """
    Config.ensure_directories()
    
    # Initialize matcher
    matcher = ATCMatcher()
    
    if json_path:
        # Process single file
        json_path = Path(json_path)
        if not json_path.exists():
            logger.error(f"JSON file not found: {json_path}")
            return
        
        result = enrich_json_file(json_path, matcher, dry_run=dry_run)
        
        if update_mappings and not dry_run:
            update_drug_mappings([result])
        
        print(f"\nEnrichment result:")
        print(f"  Drug ID: {result['drug_id']}")
        print(f"  Success: {result['success']}")
        if result.get('atc_codes'):
            print(f"  ATC Codes: {', '.join(result['atc_codes'])}")
            print(f"  Matched by: {result.get('matched_by', 'unknown')}")
            print(f"  Confidence: {result.get('confidence', 0.0):.2f}")
    else:
        # Process all files
        structured_dir = Config.STRUCTURED_DIR
        
        if not structured_dir.exists():
            logger.error(f"Structured directory not found: {structured_dir}")
            return
        
        json_files = sorted(list(structured_dir.glob("*_SmPC.json")))
        
        if not json_files:
            logger.warning(f"No JSON files found in {structured_dir}")
            return
        
        logger.info(f"Processing {len(json_files)} JSON files...")
        
        results = []
        for i, json_file in enumerate(json_files, 1):
            logger.info(f"[{i}/{len(json_files)}] Processing {json_file.name}...")
            result = enrich_json_file(json_file, matcher, dry_run=dry_run)
            results.append(result)
        
        # Summary
        successful = sum(1 for r in results if r.get("success") and not r.get("skipped"))
        skipped = sum(1 for r in results if r.get("skipped"))
        failed = len(results) - successful - skipped
        
        print(f"\nEnrichment summary:")
        print(f"  Total files: {len(results)}")
        print(f"  Successful: {successful}")
        print(f"  Skipped (already enriched): {skipped}")
        print(f"  Failed: {failed}")
        
        if update_mappings and not dry_run:
            update_drug_mappings(results)


if __name__ == "__main__":
    import sys
    
    dry_run = "--dry-run" in sys.argv
    
    # Check for single file argument
    json_path = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        json_path = Path(sys.argv[1])
    
    main(json_path=json_path, dry_run=dry_run)
