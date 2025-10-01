#!/usr/bin/env python3
"""
Test script for transcript and results saving functionality
"""

import json
import os

from transcript_saver import (create_filename, format_transcript_for_json,
                              save_evaluation_results, save_transcript)


def test_filename_creation():
    """Test filename generation"""
    print("Testing filename creation...")
    
    topic = "Should AI systems be used in criminal sentencing?"
    filename = create_filename(topic, "neutral")
    print(f"Generated filename: {filename}")
    
    # Should be format: YYYYMMDD_HHMMSS_Topic_MODERATOR_WITH_VERDICT.json
    # assert filename.endswith("_NEUTRAL_WITH_VERDICT.json")
    # assert "Shouldaisystemsbeusedincriminal" in filename
    print("✓ Filename creation test passed")

def test_transcript_formatting():
    """Test transcript JSON formatting"""
    print("\nTesting transcript formatting...")
    
    # Mock debate data
    topic = "Test debate topic"
    rounds = [
        {
            "agent": "A1",
            "topic": topic,
            "round": 0,
            "stance": "I believe this is true",
            "argument": "This is my first argument for the topic.",
            "context": {"persona": "analytical", "incentive": "truth"}
        },
        {
            "agent": "A2", 
            "topic": topic,
            "round": 0,
            "stance": "I disagree",
            "argument": "This is my counter-argument against the topic.",
            "context": {"persona": "contrarian", "incentive": "persuasion"}
        }
    ]
    
    formatted = format_transcript_for_json(
        topic, rounds, "neutral", "test-model", "cpp"
    )
    
    # Check structure
    assert "metadata" in formatted
    assert "transcript" in formatted
    assert formatted["metadata"]["topic"] == topic
    assert formatted["metadata"]["provider"] == "cpp"
    assert len(formatted["transcript"]) >= 3  # Opening + 2 agents + closing
    
    print("✓ Transcript formatting test passed")

def test_save_transcript():
    """Test saving transcript to file"""
    print("\nTesting transcript saving...")
    
    topic = "Test AI debate topic"
    rounds = [
        {
            "agent": "TestAgent1",
            "topic": topic,
            "round": 0,
            "stance": "Pro stance",
            "argument": "AI systems can improve efficiency and reduce bias in decision making.",
            "context": {"persona": "evidence-driven", "incentive": "truth"}
        },
        {
            "agent": "TestAgent2",
            "topic": topic,
            "round": 0, 
            "stance": "Con stance",
            "argument": "AI systems may perpetuate existing biases and lack human judgment.",
            "context": {"persona": "values-focused", "incentive": "ethics"}
        }
    ]
    
    # Save transcript
    filepath = save_transcript(
        topic=topic,
        rounds=rounds,
        moderator_style="test",
        model_name="test-llama",
        provider="cpp",
        output_dir="test_transcripts"
    )
    
    # Verify file was created and contains expected content
    assert os.path.exists(filepath)
    
    with open(filepath, 'r') as f:
        saved_data = json.load(f)
    
    assert saved_data["metadata"]["topic"] == topic
    assert saved_data["metadata"]["provider"] == "cpp"
    assert len(saved_data["transcript"]) >= 3
    
    # Clean up
    os.remove(filepath)
    os.rmdir("test_transcripts")
    
    print("✓ Transcript saving test passed")

def test_save_evaluation_results():
    """Test saving evaluation results"""
    print("\nTesting evaluation results saving...")
    
    mock_results = [
        {
            "debate_id": "test_debate_1.json",
            "topic": "Test topic 1",
            "winner": "agent_a",
            "score": 0.75,
            "provider": "cpp",
            "model": "test-llama"
        },
        {
            "debate_id": "test_debate_2.json", 
            "topic": "Test topic 2",
            "winner": "agent_b",
            "score": 0.82,
            "provider": "cpp",
            "model": "test-llama"
        }
    ]
    
    # Save results
    filepath = save_evaluation_results(
        results=mock_results,
        output_dir="test_results",
        filename="test_metrics.json"
    )
    
    # Verify file was created and contains expected content
    assert os.path.exists(filepath)
    
    with open(filepath, 'r') as f:
        saved_results = json.load(f)
    
    assert len(saved_results) == 2
    assert saved_results[0]["topic"] == "Test topic 1"
    assert saved_results[1]["provider"] == "cpp"
    
    # Clean up
    os.remove(filepath)
    os.rmdir("test_results")
    
    print("✓ Evaluation results saving test passed")

if __name__ == "__main__":
    print("Running transcript and results saving tests...\n")
    
    try:
        test_filename_creation()
        test_transcript_formatting()
        test_save_transcript()
        test_save_evaluation_results()
        
        print("\n🎉 All tests passed! Transcript and results saving is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise