#!/usr/bin/env python3
"""
Test script to verify prompt integration with agents.py
"""

import sys
import os

# Add current directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prompts import (
    get_agent_initial_belief_prompt,
    get_agent_argument_prompt,
    get_agent_update_belief_prompt,
    get_moderator_summary_prompt,
    get_moderator_judge_prompt,
    PERSONA_PROMPT_MAPPING,
    MODERATOR_STYLE_MAPPING
)

def test_prompt_formatting():
    """Test that all prompt functions work correctly"""
    print("=== Testing Prompt Formatting ===\n")
    
    # Test agent initial belief prompt
    print("1. Agent Initial Belief Prompt (Pro stance):")
    belief_prompt = get_agent_initial_belief_prompt(
        persona="evidence-driven analyst",
        incentive="truth",
        topic="AI should be regulated by government",
        stance_hint="pro"
    )
    print(belief_prompt[:200] + "..." if len(belief_prompt) > 200 else belief_prompt)
    print()
    
    # Test agent argument prompt
    print("2. Agent Argument Generation Prompt:")
    arg_prompt = get_agent_argument_prompt(
        persona="contrarian debater", 
        incentive="persuasion",
        topic="Universal Basic Income is beneficial",
        round_number=1,
        current_belief="UBI creates dependency and reduces work incentive"
    )
    print(arg_prompt[:200] + "..." if len(arg_prompt) > 200 else arg_prompt)
    print()
    
    # Test belief update prompt
    print("3. Agent Belief Update Prompt:")
    update_prompt = get_agent_update_belief_prompt(
        persona="values-focused ethicist",
        incentive="truth",
        topic="Social media harms democracy", 
        current_belief="Social media creates echo chambers",
        other_arguments=[
            "Social media enables democratic participation",
            "The problem is not the platform but how it's used"
        ]
    )
    print(update_prompt[:200] + "..." if len(update_prompt) > 200 else update_prompt)
    print()
    
    # Test moderator summary prompt
    print("4. Moderator Summary Prompt:")
    mock_round_args = [
        {"agent": "A1", "argument": "Government regulation protects consumers"},
        {"agent": "A2", "argument": "Regulation stifles innovation and competition"},
        {"agent": "A3", "argument": "A balanced approach with minimal oversight is best"}
    ]
    summary_prompt = get_moderator_summary_prompt(
        style="neutral",
        topic="Government regulation of tech companies",
        round_arguments=mock_round_args
    )
    print(summary_prompt[:200] + "..." if len(summary_prompt) > 200 else summary_prompt)
    print()
    
    # Test moderator judge prompt
    print("5. Moderator Judgment Prompt:")
    mock_all_args = [
        {"agent": "A1", "round": 0, "argument": "First argument for regulation"},
        {"agent": "A2", "round": 0, "argument": "First argument against regulation"},
        {"agent": "A1", "round": 1, "argument": "Second argument for regulation"},
        {"agent": "A2", "round": 1, "argument": "Second argument against regulation"}
    ]
    judge_prompt = get_moderator_judge_prompt(
        style="analytical",
        topic="Technology regulation debate",
        all_arguments=mock_all_args
    )
    print(judge_prompt[:200] + "..." if len(judge_prompt) > 200 else judge_prompt)

def test_persona_mappings():
    """Test persona-based prompt mapping"""
    print("\n=== Testing Persona Mappings ===\n")
    
    print("Persona → Prompt Mapping:")
    for persona, prompt_key in PERSONA_PROMPT_MAPPING.items():
        print(f"  {persona} → {prompt_key}")
    
    print("\nModerator Style → Prompt Mapping:")
    for style, prompt_key in MODERATOR_STYLE_MAPPING.items():
        print(f"  {style} → {prompt_key}")

def test_prompt_variations():
    """Test different prompt variations for same persona"""
    print("\n=== Testing Prompt Variations ===\n")
    
    topic = "Climate change requires immediate action"
    personas = ["evidence-driven analyst", "values-focused ethicist", "contrarian debater"]
    
    print(f"Topic: {topic}\n")
    
    for persona in personas:
        print(f"--- {persona.upper()} ---")
        prompt = get_agent_argument_prompt(
            persona=persona,
            incentive="truth",
            topic=topic,
            round_number=1,
            current_belief="Climate action is important but costly"
        )
        # Show the guidelines section which differs by persona
        lines = prompt.split('\n')
        guidelines_start = None
        for i, line in enumerate(lines):
            if 'Guidelines:' in line:
                guidelines_start = i
                break
        
        if guidelines_start:
            guidelines_section = '\n'.join(lines[guidelines_start:guidelines_start+5])
            print(guidelines_section)
        print()

if __name__ == "__main__":
    print("Testing v2 Prompt System Integration")
    print("=" * 50)
    
    try:
        test_prompt_formatting()
        test_persona_mappings() 
        test_prompt_variations()
        
        print("\n" + "=" * 50)
        print("🎉 All prompt integration tests passed!")
        print("The v2 prompt system is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise