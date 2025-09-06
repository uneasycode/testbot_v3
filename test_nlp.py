#!/usr/bin/env python3
"""
Test script for the enhanced NLP processor
"""

import json
from bot import NLPProcessor

def test_nlp_matching():
    """Test the enhanced NLP matching with Hinglish examples"""
    
    # Load responses
    with open('responses.json', 'r', encoding='utf-8') as f:
        responses = json.load(f)
    
    # Initialize NLP processor
    nlp = NLPProcessor()
    nlp.initialize_vectorizer(list(responses.keys()))
    
    # Test cases for Hinglish matching
    test_cases = [
        # Basic variations
        ("helo", "hello"),
        ("hllo", "hello"),
        ("namaste", "hello"),
        ("alvida", "bye"),
        ("by", "bye"),
        
        # Hinglish specific
        ("khaana khaya", "khana khaya"),
        ("neend nhi aa rhi", "neend nahi aa rhi hai"),
        ("excercise kro", "excercise kiya karo"),
        ("kya matlab", "what do you mean"),
        ("aap hello bol skte ho", "aap hello bol sakte ho"),
        
        # Typos and variations
        ("kese ho", "kaise ho"),
        ("accha", "acha"),
        ("nhi", "nahi"),
        ("tm kya kr rhe ho", "tum kya kar rahe ho"),
        
        # Mixed cases
        ("Hello ji", "hello"),
        ("BYE BYE", "bye"),
        ("Khana khaya???", "khana khaya"),
    ]
    
    print("Testing Enhanced NLP Matching for Hinglish")
    print("=" * 50)
    
    for test_input, expected_category in test_cases:
        matched_keyword, confidence = nlp.find_best_match(
            test_input, 
            list(responses.keys()), 
            threshold=0.4
        )
        
        print(f"Input: '{test_input}'")
        print(f"Expected category: '{expected_category}'")
        print(f"Matched: '{matched_keyword}' (confidence: {confidence:.3f})")
        
        if matched_keyword:
            print(f"✅ Match found")
        else:
            print(f"❌ No match found")
        print("-" * 30)
    
    # Test preprocessing
    print("\nTesting Text Preprocessing")
    print("=" * 30)
    
    preprocessing_tests = [
        "Hello ji, kaise ho???",
        "mujhe neend nahi aa rahi hai",
        "excercise kiya karo daily",
        "kya matlab hai iska",
        "aap hello bol sakte ho kya"
    ]
    
    for text in preprocessing_tests:
        processed = nlp.preprocess_text(text)
        print(f"Original: '{text}'")
        print(f"Processed: '{processed}'")
        print("-" * 20)

if __name__ == "__main__":
    test_nlp_matching()
