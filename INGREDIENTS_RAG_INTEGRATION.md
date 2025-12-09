# Ingredients Scraper Integration with RAG System

## Overview

The Ingredients scraper (`Samheitaskrá`) provides a **chemical composition** perspective that complements the ATC scraper's **therapeutic classification** perspective. Together, they enable richer, more comprehensive drug information retrieval and question answering.

## Data Structure from Ingredients Scraper

Based on the example provided, the scraped data structure will be:

```json
{
  "ingredients": {
    "Ibuprofenum INN": {
      "inn_name": "Ibuprofenum",
      "inn_code": "INN",
      "drugs": {
        "Alvofen Express": {
          "form_strength": "Mjúkt hylki / 400 mg",
          "documents": [
            {
              "type": "Fylgiseðill",
              "url": "...",
              "date": "21.3.2025"
            },
            {
              "type": "SmPC",
              "url": "...",
              "date": "21.3.2025"
            }
          ]
        },
        "Íbúfen": {
          "form_strength": "Filmuhúðuð tafla / 200 mg",
          "documents": [...]
        },
        "Nurofen Junior Appelsín": {
          "form_strength": "Mixtúra, dreifa / 40 mg/ml",
          "documents": [...]
        }
        // ... more brands
      }
    },
    "Paracetamolum INN": {
      // ... similar structure
    }
  },
  "drug_to_ingredients": {
    "Íbúfen": ["Ibuprofenum INN"],
    "Alvofen Express": ["Ibuprofenum INN"],
    "Parapró": ["Ibuprofenum INN", "Paracetamolum INN"]  // Combination drug
  },
  "scraped_at": "2024-12-04T23:19:31"
}
```

## Integration Points in RAG System

### 1. **IngredientsManager Class** (Similar to ATCManager)

Create `src/ingredients_manager.py` following the same pattern as `ATCManager`:

```python
class IngredientsManager:
    """Manages ingredients data access and provides utilities for RAG integration."""
    
    def __init__(self, ingredients_index_path: Optional[Path] = None):
        # Load ingredients data from JSON
        pass
    
    def get_ingredients_for_drug(self, drug_id: str) -> List[str]:
        """Get active ingredients (INN names) for a drug."""
        pass
    
    def get_drugs_by_ingredient(self, ingredient_name: str) -> List[str]:
        """Get all drugs containing a specific active ingredient."""
        pass
    
    def format_ingredients_context_for_rag(
        self, 
        drug_id: str, 
        include_alternatives: bool = True
    ) -> str:
        """Format ingredients information as context for RAG prompts."""
        pass
    
    def get_generic_alternatives(self, drug_id: str, max_results: int = 10) -> List[str]:
        """Get generic alternatives (same active ingredient, different brand)."""
        pass
```

### 2. **Enhanced Query Disambiguation**

**Current State**: `query_disambiguation.py` only matches against known drug names.

**Enhancement**: Add ingredient-based matching:

```python
def find_matching_medications(
    query: str,
    vector_store_manager: VectorStoreManager,
    ingredients_manager: IngredientsManager  # NEW
) -> List[str]:
    """
    Find medications that match the query.
    Now includes ingredient-based matching.
    """
    # Existing: brand name matching
    available_medications = vector_store_manager.get_unique_medications()
    matches = detect_medications_in_query(query, available_medications)
    
    # NEW: If no matches, try ingredient matching
    if not matches:
        # Check if query mentions an active ingredient
        query_lower = query.lower()
        for ingredient_name, ingredient_data in ingredients_manager.ingredients.items():
            inn_name = ingredient_data["inn_name"].lower()
            if inn_name in query_lower or query_lower in inn_name:
                # Find all drugs with this ingredient
                drugs = ingredients_manager.get_drugs_by_ingredient(ingredient_name)
                matches.extend(drugs)
    
    return list(set(matches))
```

**Use Case Example**:
- User query: "Hvað er Ibuprofen?"
- System finds: All brands containing Ibuprofenum (Íbúfen, Alvofen, Nurofen, etc.)
- System can: Either retrieve all brands, or ask user which brand they mean

### 3. **Enhanced Retrieval with Ingredient-Based Expansion**

**Current State**: Retrieval only searches by exact drug name.

**Enhancement**: Expand queries to include all brands with same active ingredient:

```python
# In rag_chain_langgraph.py or vector_store.py

def expand_query_with_ingredients(
    query: str,
    extracted_medication: Optional[str],
    ingredients_manager: IngredientsManager
) -> Dict[str, Any]:
    """
    Expand query to include all brands with same active ingredient.
    
    Returns:
        Dictionary with:
        - expanded_drug_ids: List of all relevant drug IDs
        - ingredient_info: Ingredient context string
    """
    if not extracted_medication:
        return {"expanded_drug_ids": [], "ingredient_info": ""}
    
    # Get active ingredients for the drug
    ingredients = ingredients_manager.get_ingredients_for_drug(extracted_medication)
    
    if not ingredients:
        return {"expanded_drug_ids": [extracted_medication], "ingredient_info": ""}
    
    # Find all drugs with same ingredients (generic alternatives)
    all_related_drugs = set([extracted_medication])
    ingredient_names = []
    
    for ingredient in ingredients:
        drugs_with_ingredient = ingredients_manager.get_drugs_by_ingredient(ingredient)
        all_related_drugs.update(drugs_with_ingredient)
        ingredient_names.append(ingredient)
    
    # Format ingredient context
    ingredient_info = f"\nVirk innihaldsefni: {', '.join(ingredient_names)}\n"
    ingredient_info += f"Aðrar vörumerki með sama virka efni: {', '.join(sorted(all_related_drugs - {extracted_medication}))}\n"
    
    return {
        "expanded_drug_ids": sorted(list(all_related_drugs)),
        "ingredient_info": ingredient_info
    }
```

**Use Case Example**:
- User asks about "Íbúfen"
- System expands to: ["Íbúfen", "Alvofen Express", "Alvofen Junior", "Ibetin", "Nurofen Junior Appelsín", ...]
- System retrieves documents from ALL brands with Ibuprofenum
- Provides comprehensive answer covering all formulations

### 4. **Enhanced Context Formatting in RAG**

**Current State**: `_format_context()` in `rag_chain_langgraph.py` adds ATC context.

**Enhancement**: Add ingredients context alongside ATC:

```python
def _format_context(
    self, 
    docs: List[Document], 
    include_atc: bool = True,
    include_ingredients: bool = True  # NEW
) -> str:
    """Format documents into context string for LLM prompt."""
    formatted_context_parts = []
    seen_drugs = set()
    
    for idx, doc in enumerate(docs, 1):
        metadata = doc.metadata
        drug_id = metadata.get("drug_id", metadata.get("medication_name", "Unknown"))
        section_num = metadata.get("section_number", "Unknown")
        section_title = metadata.get("section_title", metadata.get("section", "Unknown"))
        
        context_block = f"""
Kafli {section_num}: {section_title}
Lyf: {drug_id}
Innihald:
{doc.page_content}
"""
        formatted_context_parts.append(context_block)
        
        # Add ATC context (existing)
        if include_atc and drug_id not in seen_drugs and drug_id != "Unknown":
            atc_context = self.atc_manager.format_atc_context_for_rag(
                drug_id,
                include_alternatives=False
            )
            if atc_context:
                formatted_context_parts.append(f"ATC upplýsingar fyrir {drug_id}:\n{atc_context}")
        
        # NEW: Add ingredients context
        if include_ingredients and drug_id not in seen_drugs and drug_id != "Unknown":
            ingredients_context = self.ingredients_manager.format_ingredients_context_for_rag(
                drug_id,
                include_alternatives=False
            )
            if ingredients_context:
                formatted_context_parts.append(f"Virk innihaldsefni fyrir {drug_id}:\n{ingredients_context}")
        
        seen_drugs.add(drug_id)
    
    return "\n---\n".join(formatted_context_parts)
```

**Example Output**:
```
Kafli 4.1: Indications
Lyf: Íbúfen
Innihald:
Íbúfen er notað til að lækka...

ATC upplýsingar fyrir Íbúfen:
ATC flokkun fyrir Íbúfen:
  - M01AE01: M > M01 (Musculo-skeletal system > Antiinflammatory and antirheumatic products) > M01AE (Propionic acid derivatives) > M01AE01 (Ibuprofen)

Virk innihaldsefni fyrir Íbúfen:
Virk innihaldsefni: Ibuprofenum INN
Aðrar vörumerki með sama virka efni:
  - Alvofen Express (Mjúkt hylki / 400 mg)
  - Alvofen Junior (Mixtúra, dreifa / 40 mg/ml)
  - Ibetin (Filmuhúðuð tafla / 400 mg)
  - Nurofen Junior Appelsín (Mixtúra, dreifa / 40 mg/ml)
  ...
```

### 5. **Generic Alternative Finding**

**Current State**: `get_alternatives()` in `ATCManager` finds therapeutic alternatives (same ATC code).

**Enhancement**: Add ingredient-based generic alternatives:

```python
# In generation node of rag_chain_langgraph.py

# Check if query asks for alternatives
asks_for_alternatives = any(
    phrase in question_lower
    for phrase in ["valkostir", "aðrar lyf", "sambærileg lyf", "generics", "jafngildi"]
)

if asks_for_alternatives:
    extracted_medication = state.get("extracted_medication")
    if extracted_medication:
        # Therapeutic alternatives (same ATC - existing)
        atc_alternatives = self.atc_manager.get_alternatives(extracted_medication, max_results=5)
        
        # Generic alternatives (same ingredient - NEW)
        generic_alternatives = self.ingredients_manager.get_generic_alternatives(
            extracted_medication, 
            max_results=5
        )
        
        if atc_alternatives or generic_alternatives:
            alt_context = "\n\nValkostir:\n"
            
            if generic_alternatives:
                alt_context += "Jafngildi (sama virka efni, önnur vörumerki):\n"
                for alt_drug in generic_alternatives:
                    alt_context += f"- {alt_drug}\n"
            
            if atc_alternatives:
                alt_context += "\nSambærileg lyf (sama ATC flokk, önnur virk efni):\n"
                for alt_drug in atc_alternatives:
                    alt_atc = self.atc_manager.get_atc_codes_for_drug(alt_drug)
                    alt_context += f"- {alt_drug} (ATC: {', '.join(alt_atc) if alt_atc else 'Ekki þekkt'})\n"
            
            context += alt_context
```

**Use Case Example**:
- User: "Hvaða valkostir eru til fyrir Íbúfen?"
- System responds:
  - **Jafngildi** (same active ingredient): Alvofen, Ibetin, Nurofen, etc.
  - **Sambærileg lyf** (same therapeutic class, different ingredient): Other NSAIDs like Naproxen, Diclofenac

### 6. **Vector Store Metadata Enhancement**

**Current State**: Vector store documents have `drug_id` and `atc_codes` metadata.

**Enhancement**: Add `active_ingredients` metadata:

```python
# When ingesting documents, add ingredient metadata
def enrich_document_metadata(
    doc: Document,
    ingredients_manager: IngredientsManager
) -> Document:
    """Enrich document metadata with ingredient information."""
    drug_id = doc.metadata.get("drug_id")
    if drug_id:
        ingredients = ingredients_manager.get_ingredients_for_drug(drug_id)
        doc.metadata["active_ingredients"] = ingredients
    
    return doc
```

**Benefits**:
- Filter retrieval by active ingredient: `filter={"active_ingredients": {"$contains": "Ibuprofenum"}}`
- Find all documents for drugs with specific ingredient
- Cross-reference ATC + Ingredients in queries

### 7. **Combined ATC + Ingredients Retrieval**

Create a unified retrieval strategy that uses both:

```python
def get_retriever_by_ingredient_and_atc(
    self,
    ingredient_name: Optional[str] = None,
    atc_code: Optional[str] = None,
    top_k: Optional[int] = None
):
    """
    Get retriever filtered by both ingredient and ATC code.
    
    Useful for queries like: "NSAIDs containing Ibuprofen"
    """
    filters = {}
    
    if ingredient_name:
        filters["active_ingredients"] = {"$contains": ingredient_name}
    
    if atc_code:
        filters["atc_codes"] = {"$contains": atc_code}
    
    search_kwargs = {
        "k": top_k or Config.RETRIEVAL_TOP_K,
        "filter": filters
    }
    
    return self.vector_store.as_retriever(search_kwargs=search_kwargs)
```

## Complete Integration Flow Example

### Scenario: User asks "Hvað er Ibuprofen og hvaða lyf innihalda það?"

1. **Query Processing**:
   - Extract "Ibuprofen" from query
   - `ingredients_manager.get_drugs_by_ingredient("Ibuprofenum INN")` → ["Íbúfen", "Alvofen Express", ...]

2. **Retrieval Expansion**:
   - Expand query to retrieve documents from ALL brands with Ibuprofenum
   - Retrieve SmPC sections from: Íbúfen, Alvofen, Nurofen, etc.

3. **Context Formatting**:
   - Add ATC context: "M01AE01 - Antiinflammatory and antirheumatic products"
   - Add ingredients context: "Virk efni: Ibuprofenum INN. Aðrar vörumerki: Alvofen, Ibetin, Nurofen..."

4. **LLM Generation**:
   - LLM receives comprehensive context from multiple brands
   - Can provide unified answer about Ibuprofen across all formulations
   - Can list all available brands and their forms/strengths

5. **Response**:
   ```
   Ibuprofenum er virkt efni sem tilheyrir flokki sýklaeyðandi og bólgueyðandi lyfja (NSAIDs).
   
   Lyf sem innihalda Ibuprofenum á íslenskum markaði:
   - Íbúfen (200 mg, 400 mg, 600 mg taflur)
   - Alvofen Express (400 mg mjúkt hylki)
   - Alvofen Junior (40 mg/ml mixtúra)
   - Nurofen Junior (40 mg/ml mixtúra, ýmsar bragðtegundir)
   ...
   
   [Citations to relevant SmPC sections]
   ```

## Benefits Summary

1. **Better Query Understanding**: Handles ingredient names, not just brand names
2. **Comprehensive Retrieval**: Retrieves from all brands with same ingredient
3. **Generic Alternative Finding**: Identifies same-ingredient alternatives (cheaper options)
4. **Richer Context**: Combines therapeutic (ATC) + chemical (Ingredients) perspectives
5. **Cross-Referencing**: Can answer questions like "What other NSAIDs contain Ibuprofen?"
6. **Improved Disambiguation**: When user says "Ibuprofen", system knows all relevant brands
7. **Better Answers**: LLM can provide comprehensive information across all formulations

## Implementation Priority

1. **Phase 1**: Create `IngredientsManager` and load data
2. **Phase 2**: Add ingredient context to RAG prompts (similar to ATC)
3. **Phase 3**: Enhance query disambiguation with ingredient matching
4. **Phase 4**: Add ingredient-based retrieval expansion
5. **Phase 5**: Add generic alternative finding
6. **Phase 6**: Enhance vector store metadata with ingredients

## Files to Create/Modify

**New Files**:
- `src/ingredients_manager.py` - Similar to `atc_manager.py`
- `src/ingredients_scraper.py` - Similar to `atc_scraper.py`

**Modified Files**:
- `config.py` - Add `INGREDIENTS_INDEX_PATH`
- `src/rag_chain_langgraph.py` - Add ingredients context formatting
- `src/query_disambiguation.py` - Add ingredient-based matching
- `src/vector_store.py` - Add ingredient metadata support
- `enrich_with_atc.py` - Consider `enrich_with_ingredients.py` or combined enrichment

This integration will significantly enhance the RAG system's ability to understand and answer questions about medications from both therapeutic and chemical perspectives.

