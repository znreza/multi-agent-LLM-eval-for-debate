"""
Psychometric parsing utilities for extracting structured data from agent responses.
Handles parsing of argumentation structure, cognitive markers, and theory of mind data.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple


def repair_incomplete_json(text: str) -> str:
    """
    Attempts to repair common JSON formatting issues in truncated responses.
    
    Args:
        text: Potentially incomplete or malformed JSON string
        
    Returns:
        Repaired JSON string that should be parseable
    """
    text = text.strip()
    
    # Remove any leading/trailing non-JSON text
    start_idx = text.find('{')
    if start_idx > 0:
        text = text[start_idx:]
    
    # Handle common truncation issues
    if text and not text.endswith('}'):
        # Count opening vs closing braces
        open_braces = text.count('{')
        close_braces = text.count('}')
        missing_braces = open_braces - close_braces
        
        # If we have unclosed braces, try to close them
        if missing_braces > 0:
            # Check if the last character suggests an incomplete key-value pair
            if text.rstrip().endswith((',', ':', '"')):
                # Remove trailing comma/colon/quote that might be incomplete
                text = text.rstrip().rstrip(',:;"')
            
            # Add missing closing braces
            text += '}' * missing_braces
    
    # Handle unclosed arrays
    open_brackets = text.count('[')
    close_brackets = text.count(']')
    missing_brackets = open_brackets - close_brackets
    if missing_brackets > 0:
        # Find the last unclosed array and close it
        text = text.rstrip().rstrip(',') + ']' * missing_brackets
    
    # Handle unclosed strings (basic attempt)
    quote_count = text.count('"')
    if quote_count % 2 != 0:
        # Odd number of quotes - try to close the last one
        text += '"'
    
    return text


def parse_structured_argument(response_text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Parses a response expected to be a single JSON object.
    Extracts the main argument text and the full structured analysis.

    Args:
        response_text: Raw response from the agent, expected to contain a JSON object.

    Returns:
        Tuple of (main_argument_text, structured_analysis_dict)
    """
    try:
        # The entire response should be the JSON object
        parsed_json = json.loads(response_text)
        
        main_argument = parsed_json.get("main_argument_text", "Argument text not found in JSON.")
        structured_analysis = parsed_json.get("structured_analysis", {})

        # Ensure the nested structure is a dict
        if not isinstance(structured_analysis, dict):
            structured_analysis = {"error": "structured_analysis was not a valid object"}

        return main_argument, structured_analysis

    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Warning: Could not parse response as JSON: {e}")
        # Try to repair the JSON before falling back
        try:
            repaired_json = repair_incomplete_json(response_text)
            print(f"Attempting JSON repair...")
            parsed_json = json.loads(repaired_json)
            
            main_argument = parsed_json.get("main_argument_text", "Argument text not found in JSON.")
            structured_analysis = parsed_json.get("structured_analysis", {})
            
            if not isinstance(structured_analysis, dict):
                structured_analysis = {"error": "structured_analysis was not a valid object"}
                
            return main_argument, structured_analysis
            
        except (json.JSONDecodeError, AttributeError) as repair_error:
            print(f"JSON repair failed: {repair_error}")
            return response_text, extract_fallback_structure(response_text)


def parse_theory_of_mind(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Parses a response expected to be a single JSON object for Theory of Mind.
    
    Args:
        response_text: Raw response from the agent.
        
    Returns:
        A dictionary with the theory of mind analysis, or a fallback.
    """
    try:
        return json.loads(response_text)
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Warning: Failed to parse theory of mind JSON: {e}")
        # Try to repair the JSON before falling back
        try:
            repaired_json = repair_incomplete_json(response_text)
            print(f"Attempting JSON repair...")
            return json.loads(repaired_json)
        except (json.JSONDecodeError, AttributeError) as repair_error:
            print(f"JSON repair failed: {repair_error}")
            return extract_fallback_tom(response_text)


def parse_belief_update(response_text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Parses a response expected to be a single JSON object for belief updates.
    
    Args:
        response_text: Raw response from the agent.
        
    Returns:
        Tuple of (updated_belief_text, analysis_dict)
    """
    try:
        parsed_json = json.loads(response_text)
        updated_belief = parsed_json.get("updated_belief", "Belief text not found.")
        analysis = parsed_json.get("belief_update_analysis", {})
        return updated_belief, analysis
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Warning: Failed to parse belief update JSON: {e}")
        # Try to repair the JSON before falling back
        try:
            repaired_json = repair_incomplete_json(response_text)
            print(f"Attempting JSON repair...")
            parsed_json = json.loads(repaired_json)
            
            updated_belief = parsed_json.get("updated_belief", "Belief text not found.")
            analysis = parsed_json.get("belief_update_analysis", {})
            return updated_belief, analysis
            
        except (json.JSONDecodeError, AttributeError) as repair_error:
            print(f"JSON repair failed: {repair_error}")
            return response_text, extract_fallback_belief_update(response_text)


def extract_fallback_structure(text: str) -> Dict[str, Any]:
    """Fallback if structured argument parsing fails."""
    return {
        "claims": [{"id": 1, "text": "Fallback: Could not parse claims.", "confidence": 0.5}],
        "evidence": [], "warrants": [], "rebuttals": [],
        "confidence_score": 0.5, "cognitive_effort": 3
    }

def extract_fallback_tom(text: str) -> Dict[str, Any]:
    """Fallback if ToM parsing fails."""
    return {
        "opponent_understanding": "Fallback: Could not parse.", "acknowledged_points": [],
        "predicted_response": "Fallback: Could not parse.", "common_ground": [],
        "empathy_score": 0.0, "perspective_taking_accuracy": 0.0
    }

def extract_fallback_belief_update(text: str) -> Dict[str, Any]:
    """Fallback if belief update parsing fails."""
    return {
        "belief_change": "unknown", "persuasive_elements": [], "resistance_factors": [],
        "confidence_before": 0.5, "confidence_after": 0.5, "cognitive_dissonance": 0.0
    }


def extract_epistemic_markers(text: str) -> List[Dict[str, Any]]:
    """Extract epistemic language markers from text."""
    epistemic_patterns = {
        'high_certainty': ['certainly', 'definitely', 'absolutely', 'undoubtedly', 'clearly', 'obviously'],
        'medium_certainty': ['likely', 'probably', 'presumably', 'apparently', 'seemingly'],
        'low_certainty': ['possibly', 'perhaps', 'maybe', 'might', 'could', 'may'],
        'necessity': ['must', 'should', 'ought', 'need to', 'have to'],
        'possibility': ['can']
    }
    found_markers = []
    text_lower = text.lower()
    unique_markers = set()

    for category, markers in epistemic_patterns.items():
        for marker in markers:
            if marker in text_lower and marker not in unique_markers:
                found_markers.append({
                    'marker': marker,
                    'category': category,
                    'count': text_lower.count(marker)
                })
                unique_markers.add(marker)
    return found_markers


def calculate_argument_complexity(structured_analysis: Dict[str, Any]) -> Dict[str, float]:
    """Calculate complexity metrics from structured argument analysis."""
    if not structured_analysis:
        return {"complexity_score": 0.0, "evidence_ratio": 0.0, "rebuttal_rate": 0.0}

    claims = structured_analysis.get('claims', [])
    evidence = structured_analysis.get('evidence', [])
    rebuttals = structured_analysis.get('rebuttals', [])
    
    num_claims = len(claims) if isinstance(claims, list) else 0
    num_evidence = len(evidence) if isinstance(evidence, list) else 0
    num_rebuttals = len(rebuttals) if isinstance(rebuttals, list) else 0

    complexity_score = num_claims + num_evidence + num_rebuttals
    complexity_score = min(complexity_score / 10.0, 1.0)

    evidence_ratio = num_evidence / max(num_claims, 1)
    evidence_ratio = min(evidence_ratio, 1.0)

    rebuttal_rate = num_rebuttals / max(num_claims, 1) # Ratio of rebuttals to own claims
    rebuttal_rate = min(rebuttal_rate, 1.0)

    return {
        "complexity_score": round(complexity_score, 2),
        "evidence_ratio": round(evidence_ratio, 2),
        "rebuttal_rate": round(rebuttal_rate, 2)
    }


def calculate_tom_metrics(tom_analysis: Dict[str, Any]) -> Dict[str, float]:
    """Calculate theory of mind metrics."""
    if not tom_analysis:
        return {"empathy_score": 0.0, "common_ground_identification": 0.0, "perspective_taking_accuracy": 0.0}

    empathy_score = tom_analysis.get('empathy_score', 0.0)
    acknowledged_points = tom_analysis.get('acknowledged_points', [])
    common_ground = tom_analysis.get('common_ground', [])

    num_acknowledged = len(acknowledged_points) if isinstance(acknowledged_points, list) else 0
    num_common_ground = len(common_ground) if isinstance(common_ground, list) else 0

    common_ground_score = (num_acknowledged + num_common_ground) / 5.0 # Normalize by a reasonable max
    common_ground_score = min(common_ground_score, 1.0)

    return {
        "empathy_score": empathy_score,
        "common_ground_identification": round(common_ground_score, 2),
        "perspective_taking_accuracy": tom_analysis.get('perspective_taking_accuracy', 0.0)
    }