#!/usr/bin/env python3
"""
Test script for the cpp provider in agents.py
Make sure you have llama-cpp-python installed: pip install llama-cpp-python
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents import Agent, Moderator
from config import CONFIG

def test_cpp_agent():
    print("Testing Agent with cpp provider...")
    
    # Create an agent with cpp provider
    agent = Agent(
        name="TestAgent",
        persona="analytical thinker",
        incentive="truth",
        provider="cpp"
    )
    
    # Test a simple prompt
    test_prompt = "What are the key benefits of renewable energy?"
    print(f"Prompt: {test_prompt}")
    
    try:
        response = agent._call_llm(test_prompt)
        print(f"Agent Response: {response}")
        print("✓ Agent test successful!")
        return True
    except Exception as e:
        print(f"✗ Agent test failed: {e}")
        return False

def test_cpp_moderator():
    print("\nTesting Moderator with cpp provider...")
    
    # Create a moderator with cpp provider
    moderator = Moderator(
        style="neutral",
        provider="cpp"
    )
    
    # Test a simple prompt
    test_prompt = "Summarize the key points from this debate about climate change."
    print(f"Prompt: {test_prompt}")
    
    try:
        response = moderator._call_llm(test_prompt)
        print(f"Moderator Response: {response}")
        print("✓ Moderator test successful!")
        return True
    except Exception as e:
        print(f"✗ Moderator test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing cpp provider integration...")
    print(f"Model path from config: {CONFIG.get('cpp_model_path', 'Not specified')}")
    print(f"Provider from config: {CONFIG.get('provider', 'Not specified')}")
    
    agent_success = test_cpp_agent()
    moderator_success = test_cpp_moderator()
    
    if agent_success and moderator_success:
        print("\n🎉 All tests passed! The cpp provider is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")