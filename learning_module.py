"""
Learning Module for Telegram Bot
Analyzes conversation data to generate potential responses
"""

import json
import os
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
import re

logger = logging.getLogger(__name__)

class ConversationLearner:
    """Analyzes conversation data to learn and generate potential responses"""

    def __init__(self, conversation_file: str, responses_temp_file: str):
        self.conversation_file = conversation_file
        self.responses_temp_file = responses_temp_file
        self.conversations = {}
        self.learned_responses = {}

    def load_conversations(self) -> Dict[str, Any]:
        """Load conversation data from JSON file"""
        try:
            with open(self.conversation_file, 'r', encoding='utf-8') as f:
                self.conversations = json.load(f)
            logger.info(f"Loaded {len(self.conversations)} conversations")
            return self.conversations
        except FileNotFoundError:
            logger.warning(f"Conversation file {self.conversation_file} not found")
            return {}
        except Exception as e:
            logger.error(f"Error loading conversations: {e}")
            return {}

    def analyze_conversations(self) -> Dict[str, Any]:
        """Analyze conversations to generate potential responses"""
        if not self.conversations:
            self.load_conversations()

        learned_responses = {}
        user_inputs = []
        bot_responses = []

        # Extract user inputs and bot responses
        for user_input, response_data in self.conversations.items():
            if isinstance(response_data, dict) and 'content' in response_data:
                content = response_data['content']
                # Handle both string and list content types
                if isinstance(content, list):
                    has_content = bool(content and any(item.strip() for item in content if isinstance(item, str)))
                else:
                    has_content = bool(content and content.strip())

                if has_content:  # Only if there's actual content
                    user_inputs.append(user_input.lower().strip())
                    bot_responses.append((user_input, content))

        # Group similar inputs and their responses
        input_groups = self._group_similar_inputs(user_inputs)

        # Generate potential responses for each group
        for representative_input, similar_inputs in input_groups.items():
            responses_for_group = []

            # Find all responses for similar inputs
            for similar_input in similar_inputs:
                for orig_input, response in bot_responses:
                    if orig_input.lower().strip() == similar_input:
                        responses_for_group.append(response)

            if responses_for_group:
                # Create a consolidated response entry
                learned_responses[representative_input] = {
                    "type": "text",
                    "content": self._consolidate_responses(responses_for_group),
                    "confidence": len(similar_inputs),
                    "source": "learned",
                    "similar_inputs": similar_inputs[:5]  # Keep top 5 similar inputs
                }

        # Also learn from unanswered conversations (empty responses)
        unanswered_inputs = []
        for user_input, response_data in self.conversations.items():
            if isinstance(response_data, dict):
                content = response_data.get('content', '')
                # Handle both string and list content types
                if isinstance(content, list):
                    has_content = bool(content and any(item.strip() for item in content if isinstance(item, str)))
                else:
                    has_content = bool(content and content.strip())

                if not content or not has_content:
                    unanswered_inputs.append(user_input.lower().strip())

        # Generate suggestions for unanswered inputs
        if unanswered_inputs:
            learned_responses.update(self._generate_suggestions_for_unanswered(unanswered_inputs))

        self.learned_responses = learned_responses
        return learned_responses

    def _group_similar_inputs(self, inputs: List[str]) -> Dict[str, List[str]]:
        """Group similar user inputs together"""
        groups = defaultdict(list)

        for input_text in inputs:
            # Clean the input
            cleaned = self._clean_text(input_text)

            # Find the best representative for this input
            best_match = None
            best_score = 0

            for representative in groups.keys():
                score = self._calculate_similarity(cleaned, self._clean_text(representative))
                if score > best_score and score > 0.6:  # Similarity threshold
                    best_match = representative
                    best_score = score

            if best_match:
                groups[best_match].append(input_text)
            else:
                groups[input_text].append(input_text)

        return dict(groups)

    def _clean_text(self, text: str) -> str:
        """Clean text for better matching"""
        # Remove punctuation and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _consolidate_responses(self, responses: List[Any]) -> List[str]:
        """Consolidate multiple responses into a clean list"""
        # Flatten responses - handle both strings and lists
        flattened_responses = []
        for response in responses:
            if isinstance(response, list):
                flattened_responses.extend([str(item) for item in response if item])
            else:
                flattened_responses.append(str(response))

        # Count frequency of each response
        response_counts = Counter(flattened_responses)

        # Keep responses that appear more than once, or the most common ones
        consolidated = []
        for response, count in response_counts.most_common():
            if count > 1 or len(consolidated) < 3:  # Keep top responses
                consolidated.append(response)

        return consolidated[:5]  # Limit to 5 responses max

    def _generate_suggestions_for_unanswered(self, unanswered_inputs: List[str]) -> Dict[str, Any]:
        """Generate suggestions for unanswered user inputs"""
        suggestions = {}

        # Common response patterns for unanswered questions
        default_responses = [
            "I'm not sure how to respond to that yet.",
            "Let me learn how to answer this better.",
            "This seems like something I should know how to respond to.",
            "I'll work on improving my responses for this type of question."
        ]

        for input_text in unanswered_inputs[:20]:  # Limit to 20 suggestions
            suggestions[f"{input_text} (unanswered)"] = {
                "type": "text",
                "content": [default_responses[len(suggestions) % len(default_responses)]],
                "confidence": 1,
                "source": "unanswered_suggestion",
                "needs_review": True
            }

        return suggestions

    def save_learned_responses(self, file_path: str = None) -> bool:
        """Save learned responses to JSON file"""
        if file_path is None:
            file_path = self.responses_temp_file

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.learned_responses, f, indent=4, ensure_ascii=False)
            logger.info(f"Saved {len(self.learned_responses)} learned responses to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving learned responses: {e}")
            return False

    def learn_from_conversations(self) -> Dict[str, Any]:
        """Main method to learn from conversations and generate responses"""
        logger.info("Starting learning process from conversations...")

        # Load conversations
        conversations = self.load_conversations()
        if not conversations:
            logger.warning("No conversations found to learn from")
            return {}

        # Analyze and generate responses
        learned_responses = self.analyze_conversations()

        # Save to temporary file
        self.save_learned_responses()

        logger.info(f"Learning complete. Generated {len(learned_responses)} potential responses")
        return learned_responses

def learn_from_conversations(conversation_file: str, temp_responses_file: str) -> Dict[str, Any]:
    """Convenience function to learn from conversations"""
    learner = ConversationLearner(conversation_file, temp_responses_file)
    return learner.learn_from_conversations()

if __name__ == "__main__":
    # Example usage
    learner = ConversationLearner("conversation.json", "responses_temp.json")
    learned = learner.learn_from_conversations()
    print(f"Learned {len(learned)} responses")
