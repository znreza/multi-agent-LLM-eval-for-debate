#!/usr/bin/env python3
"""
Test script for enhanced transcript functionality including:
- Organized folder structure by model/experiment
- Moderator initial opinions
- Final judgments
- Agent belief evolution tracking
- Round summaries
"""

import os
import json
import tempfile
import shutil
from datetime import datetime

from transcript_saver import (
    save_transcript,
    format_transcript_for_json
)

def test_organized_folder_structure():
    """Test that transcripts are saved in organized model/experiment folders"""
    print("=== Testing Organized Folder Structure ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock debate data
        topic = "Test AI regulation topic"
        rounds = [
            {
                "agent": "A1",
                "topic": topic,
                "round": 0,
                "stance": "Pro regulation",
                "argument": "AI regulation ensures safety and accountability",
                "context": {"persona": "evidence-driven analyst", "incentive": "truth"}
            },
            {
                "agent": "A2",
                "topic": topic,
                "round": 0,
                "stance": "Against regulation",
                "argument": "Regulation stifles innovation and progress",
                "context": {"persona": "contrarian debater", "incentive": "persuasion"}
            }
        ]
        
        # Test with model and experiment folders
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        experiment_name = "prompt_comparison_v1"
        
        filepath = save_transcript(
            topic=topic,
            rounds=rounds,
            moderator_style="neutral",
            model_name=model_name,
            provider="cpp",
            output_dir=temp_dir,
            experiment_name=experiment_name
        )
        
        # Check directory structure
        expected_model_dir = os.path.join(temp_dir, "meta_llama_Llama_3.2_3B_Instruct")
        expected_exp_dir = os.path.join(expected_model_dir, "prompt_comparison_v1")
        
        assert os.path.exists(expected_model_dir), f"Model directory not created: {expected_model_dir}"
        assert os.path.exists(expected_exp_dir), f"Experiment directory not created: {expected_exp_dir}"
        assert os.path.exists(filepath), f"Transcript file not created: {filepath}"
        
        # Verify file is in correct location
        assert expected_exp_dir in filepath, f"File not in expected directory: {filepath}"
        
        print(f"✅ Folder structure created correctly:")
        print(f"   Model dir: {expected_model_dir}")
        print(f"   Exp dir: {expected_exp_dir}")
        print(f"   File: {filepath.split('/')[-1]}")

def test_enhanced_transcript_content():
    """Test that transcripts include all new information"""
    print("\n=== Testing Enhanced Transcript Content ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock comprehensive debate data
        topic = "Universal Basic Income effectiveness"
        rounds = [
            {
                "agent": "A1",
                "topic": topic,
                "round": 0,
                "stance": "UBI reduces poverty effectively",
                "argument": "UBI provides direct financial support to those in need",
                "context": {"persona": "values-focused ethicist", "incentive": "truth"}
            },
            {
                "agent": "A2",
                "topic": topic,
                "round": 0,
                "stance": "UBI creates dependency",
                "argument": "UBI discourages work and creates long-term dependency",
                "context": {"persona": "contrarian debater", "incentive": "persuasion"}
            },
            {
                "agent": "A1",
                "topic": topic,
                "round": 1,
                "stance": "UBI reduces poverty with safeguards",
                "argument": "With proper implementation, UBI can reduce poverty without dependency",
                "context": {"persona": "values-focused ethicist", "incentive": "truth"}
            },
            {
                "agent": "A2",
                "topic": topic,
                "round": 1,
                "stance": "Evidence shows negative effects",
                "argument": "Historical welfare programs show dependency issues persist",
                "context": {"persona": "contrarian debater", "incentive": "persuasion"}
            }
        ]
        
        moderator_opinion = "c) Are Neutral or Undecided\n\nUBI has both potential benefits and risks that need careful evaluation."
        final_judgment = "Agent A1 presented more compelling arguments with better evidence."
        round_summaries = [
            "Round 1: A1 argued for UBI benefits, A2 raised dependency concerns",
            "Round 2: A1 addressed concerns with safeguards, A2 cited historical evidence"
        ]
        agent_belief_evolution = {
            "A1": ["Initial: UBI reduces poverty", "Round 1: UBI reduces poverty with safeguards"],
            "A2": ["Initial: UBI creates dependency", "Round 2: Evidence shows negative effects"]
        }
        
        filepath = save_transcript(
            topic=topic,
            rounds=rounds,
            moderator_style="analytical",
            model_name="test-model",
            provider="test-provider",
            output_dir=temp_dir,
            moderator_initial_opinion=moderator_opinion,
            final_judgment=final_judgment,
            round_summaries=round_summaries,
            agent_belief_evolution=agent_belief_evolution,
            experiment_name="content_test"
        )
        
        # Load and verify transcript content
        with open(filepath, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
        
        # Check metadata
        metadata = transcript["metadata"]
        assert metadata["topic"] == topic
        assert metadata["num_rounds"] == 2
        assert metadata["num_agents"] == 2
        assert metadata["agent_names"] == ["A1", "A2"]
        assert metadata["debate_format"] == "multi-agent-belief-updating"
        
        # Check moderator initial opinion
        assert transcript["moderator_initial_opinion"] == moderator_opinion
        
        # Check transcript entries
        entries = transcript["transcript"]
        assert len(entries) > 6  # Opening + 2 rounds (header + 2 agents + summary each) + final judgment
        
        # Find specific entry types
        opening_entries = [e for e in entries if e.get("action") == "debate_opening"]
        round_headers = [e for e in entries if e.get("action") == "round_header"]
        arguments = [e for e in entries if e.get("action") == "argument"]
        summaries = [e for e in entries if e.get("action") == "round_summary"]
        verdict = [e for e in entries if e.get("action") == "final_judgment"]
        
        assert len(opening_entries) == 1, "Should have one opening entry"
        assert len(round_headers) == 2, "Should have two round headers"
        assert len(arguments) == 4, "Should have four argument entries"
        assert len(summaries) == 2, "Should have two round summaries"
        assert len(verdict) == 1, "Should have one final judgment"
        
        # Check agent belief evolution
        assert "agent_belief_evolution" in transcript
        assert transcript["agent_belief_evolution"]["A1"][0] == "Initial: UBI reduces poverty"
        
        print("✅ Enhanced transcript content verified:")
        print(f"   Metadata: {len(metadata)} fields")
        print(f"   Transcript entries: {len(entries)}")
        print(f"   Belief evolution tracked for {len(transcript['agent_belief_evolution'])} agents")
        print(f"   Final judgment: {final_judgment[:50]}...")

def test_backward_compatibility():
    """Test that the system works without optional parameters"""
    print("\n=== Testing Backward Compatibility ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Basic debate data without enhanced features
        topic = "Simple debate topic"
        rounds = [
            {
                "agent": "A1",
                "topic": topic,
                "round": 0,
                "stance": "Simple stance",
                "argument": "Simple argument",
                "context": {"persona": "analytical", "incentive": "truth"}
            }
        ]
        
        # Save with minimal parameters
        filepath = save_transcript(
            topic=topic,
            rounds=rounds,
            output_dir=temp_dir
        )
        
        # Load and verify basic structure exists
        with open(filepath, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
        
        assert "metadata" in transcript
        assert "moderator_initial_opinion" in transcript
        assert "transcript" in transcript
        assert len(transcript["transcript"]) >= 3  # Opening + argument + closing
        
        print("✅ Backward compatibility maintained")
        print(f"   File created: {filepath.split('/')[-1]}")

def test_multiple_experiments():
    """Test running multiple experiments with different models"""
    print("\n=== Testing Multiple Experiments Organization ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Simulate multiple experiments
        experiments = [
            {"model": "llama-3.2-3b", "experiment": "baseline", "provider": "cpp"},
            {"model": "llama-3.2-3b", "experiment": "enhanced_prompts", "provider": "cpp"},
            {"model": "deepseek-v3", "experiment": "baseline", "provider": "inference_client"},
        ]
        
        created_files = []
        
        for exp in experiments:
            filepath = save_transcript(
                topic="Test topic for organization",
                rounds=[{
                    "agent": "A1",
                    "topic": "Test topic",
                    "round": 0,
                    "stance": "Test stance",
                    "argument": "Test argument",
                    "context": {"persona": "test", "incentive": "test"}
                }],
                model_name=exp["model"],
                provider=exp["provider"],
                experiment_name=exp["experiment"],
                output_dir=temp_dir
            )
            created_files.append(filepath)
        
        # Verify directory structure
        expected_dirs = [
            os.path.join(temp_dir, "llama_3.2_3b", "baseline"),
            os.path.join(temp_dir, "llama_3.2_3b", "enhanced_prompts"),
            os.path.join(temp_dir, "deepseek_v3", "baseline")
        ]
        
        for expected_dir in expected_dirs:
            assert os.path.exists(expected_dir), f"Expected directory not found: {expected_dir}"
        
        print("✅ Multiple experiments organized correctly:")
        for i, exp in enumerate(experiments):
            print(f"   {exp['model']}/{exp['experiment']}: {created_files[i].split('/')[-1]}")

if __name__ == "__main__":
    print("Testing Enhanced Transcript Functionality")
    print("=" * 50)
    
    try:
        test_organized_folder_structure()
        test_enhanced_transcript_content()
        test_backward_compatibility()
        test_multiple_experiments()
        
        print("\n" + "=" * 50)
        print("🎉 All enhanced transcript tests passed!")
        print("✨ The system now supports:")
        print("   • Organized model/experiment folder structure")
        print("   • Moderator initial opinions")
        print("   • Final judgments and verdicts")
        print("   • Agent belief evolution tracking")
        print("   • Round-by-round summaries")
        print("   • Enhanced metadata and action tagging")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise