"""
Utility functions for the multi-agent debate system.
"""

import re


def remove_thinking_tags(text: str) -> str:
    """
    Remove <think>...</think> tags and their content from model responses.
    
    Args:
        text: Raw text from model that may contain thinking tags
        
    Returns:
        Cleaned text with thinking content removed
        
    Examples:
        remove_thinking_tags("<think>reasoning</think>actual answer") -> "actual answer"
        remove_thinking_tags("prefix<think>thought</think>middle<think>more</think>suffix") -> "prefixmiddlesuffix"
    """
    if not isinstance(text, str):
        return text
    
    # Remove all <think>...</think> blocks (non-greedy matching)
    # This handles multiple thinking blocks in a single response
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up any remaining unclosed <think> tags at the start/end
    cleaned = re.sub(r'^<think>.*$', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'^.*</think>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up extra whitespace that might be left behind
    cleaned = cleaned.strip()
    
    return cleaned


def clean_llm_response(response: str) -> str:
    """
    Clean LLM response by removing thinking tags and normalizing whitespace.
    
    Args:
        response: Raw response from LLM
        
    Returns:
        Cleaned response ready for use
    """
    if not response:
        return response
    
    # print("Raw response:", response)
    
    # Remove thinking tags
    cleaned = remove_thinking_tags(response)
    
    # Normalize whitespace (remove extra newlines/spaces but preserve structure)
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)  # Max 2 consecutive newlines
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)  # Normalize spaces/tabs
    cleaned = cleaned.strip()
    
    return cleaned


if __name__ == "__main__":
    # Test cases
    test_cases = [
        "<think>I need to analyze this carefully</think>The answer is yes.",
        "Before <think>reasoning here</think> middle <think>more thinking</think> end",
        "<think>unclosed thinking tag",
        "response without thinking</think>", 
        "<think>\nmultiline\nthinking\n</think>\nActual response here",
        "Clean response without any tags",
        ""
    ]
    
    print("Testing thinking tag removal:")
    for i, test in enumerate(test_cases, 1):
        result = remove_thinking_tags(test)
        print(f"{i}. '{test}' -> '{result}'")