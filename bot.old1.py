# bot.py - Telegram Bot with Multiple Image Support

import json
import os
import random
import asyncio
import logging
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Union, Tuple
import time

try:
    from telethon import TelegramClient, events
    from telethon.tl.types import InputMediaPhoto
except ImportError:
    raise ImportError("Telethon is required. Install it with: pip install telethon")

# Optional NLP imports with graceful degradation
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False

from config import (
    API_ID, API_HASH, PHONE, 
    RESPONSES_FILE, CONVERSATION_FILE,
    IMAGES_DIR, AUDIO_DIR
)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class NLPProcessor:
    """Handles Natural Language Processing for better message understanding"""

    def __init__(self):
        self.spell_checker = None
        self.lemmatizer = None
        self.stop_words = set()

        # Initialize NLP components if available
        if SPELLCHECKER_AVAILABLE:
            self.spell_checker = SpellChecker()

        if NLTK_AVAILABLE:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
            except Exception as e:
                logger.warning(f"Failed to initialize NLTK: {e}")

        # Common typo corrections for Hinglish
        self.typo_corrections = {
            'kya': ['kya', 'kya hai', 'kia', 'kyaa'],
            'hai': ['hai', 'he', 'h'],
            'kaise': ['kaise', 'kese', 'kayse'],
            'hello': ['hello', 'helo', 'hllo', 'hullo'],
            'bye': ['bye', 'by', 'bai', 'bbye'],
            'kaam': ['kaam', 'kam', 'kaaam'],
            'acha': ['acha', 'accha', 'achha', 'achchha'],
        }

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better matching"""
        text = text.lower().strip()

        # Fix common typos
        for correct, typos in self.typo_corrections.items():
            for typo in typos:
                text = re.sub(r'\b' + re.escape(typo) + r'\b', correct, text)

        return text

    def similarity_score(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        return SequenceMatcher(None, text1, text2).ratio()

    def find_best_match(self, user_message: str, keywords: List[str]) -> Tuple[str, float]:
        """Find the best matching keyword"""
        processed_message = self.preprocess_text(user_message)
        best_match = None
        best_score = 0.0

        for keyword in keywords:
            processed_keyword = self.preprocess_text(keyword)

            # Direct substring matching (highest priority)
            if processed_keyword in processed_message or processed_message in processed_keyword:
                return keyword, 1.0

            # String similarity
            similarity = self.similarity_score(processed_message, processed_keyword)
            if similarity > best_score:
                best_score = similarity
                best_match = keyword

        return best_match, best_score

class TelegramBot:
    """Main Telegram Bot class with multiple image support"""

    def __init__(self):
        self.client = None
        self.responses = {}
        self.nlp_processor = NLPProcessor()

    def load_responses(self):
        """Load responses from JSON file"""
        try:
            with open(RESPONSES_FILE, 'r', encoding='utf-8') as f:
                self.responses = json.load(f)
            logger.info(f"Loaded {len(self.responses)} responses")
        except FileNotFoundError:
            logger.warning(f"Responses file {RESPONSES_FILE} not found")
            self.responses = {}
        except Exception as e:
            logger.error(f"Error loading responses: {e}")
            self.responses = {}

    async def send_typing(self, chat_id: int, duration: float = None):
        """Simulate typing"""
        if duration is None:
            duration = random.uniform(1, 3)

        try:
            async with self.client.action(chat_id, 'typing'):
                await asyncio.sleep(duration)
        except Exception as e:
            logger.error(f"Error sending typing action: {e}")

    async def send_multiple_images(self, chat_id: int, image_files: List[str], caption: str = ""):
        """Send multiple images as a media group - NEW FEATURE"""
        try:
            media_list = []

            for i, image_file in enumerate(image_files):
                image_path = os.path.join(IMAGES_DIR, image_file)

                if not os.path.exists(image_path):
                    logger.warning(f"Image file not found: {image_path}")
                    continue

                # Add caption only to the first image
                img_caption = caption if i == 0 else ""
                media_list.append(InputMediaPhoto(image_path, caption=img_caption))

            if media_list:
                await self.client.send_file(chat_id, media_list)
                logger.info(f"Sent {len(media_list)} images to {chat_id}")
                return True
            else:
                logger.error("No valid images found to send")
                return False

        except Exception as e:
            logger.error(f"Error sending multiple images: {e}")
            return False

    async def send_response(self, chat_id: int, response_data: Dict):
        """Send response based on type with MULTIPLE IMAGE SUPPORT"""
        try:
            response_type = response_data.get('type', 'text')
            content = response_data.get('content', [])
            caption = response_data.get('caption', '')

            # Handle typing delay
            await self.send_typing(chat_id, random.uniform(1, 3))

            if response_type == 'text':
                # For text responses, choose randomly from list
                if isinstance(content, list):
                    message = random.choice(content) if content else "Sorry, no response available."
                else:
                    message = content if content else "Sorry, no response available."

                await self.client.send_message(chat_id, message)

            elif response_type == 'image':
                # EXISTING: Single image or random selection from list
                if isinstance(content, list):
                    image_file = random.choice(content) if content else None
                else:
                    image_file = content

                if image_file:
                    image_path = os.path.join(IMAGES_DIR, image_file)
                    if os.path.exists(image_path):
                        await self.client.send_file(chat_id, image_path, caption=caption)
                    else:
                        await self.client.send_message(chat_id, f"Image file not found: {image_file}")

            elif response_type == 'multiple-images':
                # NEW: Send all images in the list
                if isinstance(content, list) and len(content) > 1:
                    success = await self.send_multiple_images(chat_id, content, caption)
                    if not success:
                        await self.client.send_message(chat_id, "Sorry, couldn't send the images.")
                elif isinstance(content, list) and len(content) == 1:
                    # Single image in list
                    image_path = os.path.join(IMAGES_DIR, content[0])
                    if os.path.exists(image_path):
                        await self.client.send_file(chat_id, image_path, caption=caption)
                    else:
                        await self.client.send_message(chat_id, f"Image file not found: {content[0]}")
                else:
                    await self.client.send_message(chat_id, "No images configured for multiple-images response.")

            elif response_type == 'audio':
                # For audio responses, choose randomly from list or send single file
                if isinstance(content, list):
                    audio_file = random.choice(content) if content else None
                else:
                    audio_file = content

                if audio_file:
                    audio_path = os.path.join(AUDIO_DIR, audio_file)
                    if os.path.exists(audio_path):
                        await self.client.send_file(chat_id, audio_path)
                    else:
                        await self.client.send_message(chat_id, f"Audio file not found: {audio_file}")
                else:
                    await self.client.send_message(chat_id, "No audio file available.")

        except Exception as e:
            logger.error(f"Error sending response: {e}")
            await self.client.send_message(chat_id, "Sorry, there was an error processing your request.")

    def get_contextual_response(self, message: str) -> Dict:
        """Generate contextual response when no keyword match is found"""
        message_lower = message.lower()

        if any(greeting in message_lower for greeting in ['hello', 'hi', 'hey', 'namaste']):
            return {
                'type': 'text',
                'content': ['Hello!', 'Hi there!', 'Hey!', 'Namaste!'],
            }
        elif any(farewell in message_lower for farewell in ['bye', 'goodbye', 'alvida']):
            return {
                'type': 'text', 
                'content': ['Goodbye!', 'Bye!', 'See you later!', 'Take care!'],
            }
        elif '?' in message:
            return {
                'type': 'text',
                'content': [
                    'That\'s an interesting question!', 
                    'I\'m not sure about that.',
                    'Could you rephrase that?'
                ],
            }
        else:
            return {
                'type': 'text',
                'content': [
                    'I understand.',
                    'Interesting!',
                    'Could you tell me more?'
                ],
            }

    async def process_message(self, event):
        """Process incoming messages"""
        try:
            user_id = event.sender_id
            message = event.message.message.strip()

            if not message:
                return

            logger.info(f"Processing message from {user_id}: {message}")

            # Find best matching response
            keywords = list(self.responses.keys())
            best_match, confidence = self.nlp_processor.find_best_match(message, keywords)

            if best_match and confidence > 0.6:
                response_data = self.responses[best_match]
                await self.send_response(user_id, response_data)
                logger.info(f"Matched '{message}' with '{best_match}' (confidence: {confidence:.2f})")
            else:
                # No good match found, use contextual response
                contextual_response = self.get_contextual_response(message)
                await self.send_response(user_id, contextual_response)
                logger.info(f"No match found for '{message}', sent contextual response")

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            try:
                await self.client.send_message(event.sender_id, 
                    "Sorry, I encountered an error. Please try again.")
            except:
                pass

    async def start(self):
        """Start the Telegram bot"""
        try:
            # Initialize Telegram client
            self.client = TelegramClient('bot_session', API_ID, API_HASH)

            # Load responses
            self.load_responses()

            # Start client
            await self.client.start(phone=PHONE)
            logger.info("Bot started successfully!")

            # Register message handler
            @self.client.on(events.NewMessage)
            async def message_handler(event):
                await self.process_message(event)

            # Keep running
            await self.client.run_until_disconnected()

        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            raise

    def stop(self):
        """Stop the bot"""
        if self.client:
            self.client.disconnect()
            logger.info("Bot stopped")

if __name__ == "__main__":
    bot = TelegramBot()
    asyncio.run(bot.start())
