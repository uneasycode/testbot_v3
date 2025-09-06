# bot_updated.py - Updated bot with multiple-images support

"""
Telegram bot implementation using Telethon with NLP enhancements
Now includes support for multiple-images response type
"""

import os
import json
import random
import asyncio
import logging
import time
import re
from collections import Counter
from difflib import SequenceMatcher
import numpy as np
from telethon import TelegramClient, events
from telethon.tl.types import InputMediaPhoto, InputMediaDocument
from config import API_ID, API_HASH, PHONE, RESPONSES_FILE, IMAGES_DIR, AUDIO_DIR, CONVERSATION_FILE
import fnmatch

##################
# To avoid cert error add SSL bypass
###############

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Optional: Try to import advanced NLP libraries, fallback to basic processing if not available
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    # Download necessary NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class NLPProcessor:
    """Enhanced Natural Language Processing helper class for Hinglish"""

    def __init__(self):
        self.initialize_nlp_tools()

    def initialize_nlp_tools(self):
        """Initialize enhanced NLP tools for Hinglish"""
        # Load NLP configuration from file
        try:
            with open('nlp_config.json', 'r', encoding='utf-8') as f:
                nlp_config = json.load(f)
        except FileNotFoundError:
            logger.warning("nlp_config.json not found, using default settings")
            nlp_config = {
                "nlp_enabled": True,
                "common_typos": {},
                "stop_words": [],
                "phonetic_mappings": {}
            }

        # Load NLP enabled setting
        self.nlp_enabled = nlp_config.get('nlp_enabled', True)

        # Load common typos
        self.common_typos = nlp_config.get('common_typos', {})

        # Load stopwords
        self.stop_words = set(nlp_config.get('stop_words', []))

        # Load phonetic mappings
        self.phonetic_mappings = nlp_config.get('phonetic_mappings', {})

        # No lemmatizer for Hinglish
        if SPELLCHECKER_AVAILABLE:
            self.spell = SpellChecker(language=None)  # Use generic, not English

        if SKLEARN_AVAILABLE:
            self.vectorizer = None

    def preprocess_text(self, text):
        """
        Enhanced preprocessing for Hinglish text:
        - Lowercase
        - Remove punctuation
        - Apply phonetic normalization
        - Fix common typos and variations
        - Remove stopwords
        """
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove extra whitespace and punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Apply phonetic mappings for better matching
        for phonetic, replacement in self.phonetic_mappings.items():
            text = text.replace(phonetic, replacement)
        
        # Split into words and apply typo corrections
        words = text.split()
        normalized_words = []
        
        for word in words:
            # Apply typo corrections
            corrected_word = self.common_typos.get(word, word)
            
            # Skip if it's a stopword
            if corrected_word not in self.stop_words and len(corrected_word) > 1:
                normalized_words.append(corrected_word)
        
        processed_text = ' '.join(normalized_words)
        
        # Additional normalization for common Hinglish patterns
        processed_text = self.normalize_hinglish_patterns(processed_text)
        
        return processed_text
    
    def normalize_hinglish_patterns(self, text):
        """Apply additional Hinglish-specific normalizations"""
        # Handle repeated characters (e.g., "hellooo" -> "hello")
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Normalize common Hinglish word endings
        text = re.sub(r'\bkr\b', 'kar', text)  # kr -> kar
        text = re.sub(r'\bho\b', 'hai', text)  # ho -> hai
        text = re.sub(r'\bna\b', 'nahi', text)  # na -> nahi
        
        # Handle common contractions
        text = re.sub(r'\bkya\s+kar\b', 'kya kar', text)
        text = re.sub(r'\bkaise\s+ho\b', 'kaise hai', text)
        
        return text

    def find_best_match(self, input_text, keywords, threshold=0.4):
        """
        Enhanced matching for Hinglish text with multiple similarity methods
        Returns the matched keyword and the confidence score
        """
        # Preprocess input text
        processed_input = self.preprocess_text(input_text)
        original_input = input_text.lower().strip()
        
        best_match = None
        best_score = 0
        
        for keyword in keywords:
            processed_keyword = self.preprocess_text(keyword)
            original_keyword = keyword.lower().strip()
            
            # Method 1: Direct containment (highest priority)
            if processed_keyword in processed_input or processed_input in processed_keyword:
                return keyword, 1.0
            
            # Method 2: Word overlap similarity
            word_overlap_score = self.calculate_word_overlap(processed_input, processed_keyword)
            
            # Method 3: String similarity on processed text
            processed_similarity = self.calculate_similarity(processed_input, processed_keyword)
            
            # Method 4: String similarity on original text
            original_similarity = self.calculate_similarity(original_input, original_keyword)
            
            # Method 5: Phonetic similarity for Hinglish
            phonetic_score = self.calculate_phonetic_similarity(processed_input, processed_keyword)
            
            # Combine scores with weights
            combined_score = max(
                word_overlap_score * 0.4,
                processed_similarity * 0.3,
                original_similarity * 0.2,
                phonetic_score * 0.1
            )
            
            if combined_score > best_score:
                best_score = combined_score
                best_match = keyword
        
        # Use TF-IDF if available and no good match found yet
        if SKLEARN_AVAILABLE and self.vectorizer is not None and best_score < 0.7:
            try:
                input_vector = self.vectorizer.transform([processed_input])
                
                for keyword in keywords:
                    processed_keyword = self.preprocess_text(keyword)
                    keyword_vector = self.vectorizer.transform([processed_keyword])
                    similarity = cosine_similarity(input_vector, keyword_vector)[0][0]
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_match = keyword
            except Exception as e:
                logger.debug(f"TF-IDF matching failed: {e}")
        
        # Return match if score exceeds threshold
        if best_score >= threshold:
            return best_match, best_score
        
        return None, 0
    
    def calculate_similarity(self, text1, text2):
        """Calculate similarity between two strings using sequence matcher"""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def calculate_word_overlap(self, text1, text2):
        """Calculate word overlap similarity"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def calculate_phonetic_similarity(self, text1, text2):
        """Calculate phonetic similarity for Hinglish words"""
        # Apply phonetic transformations
        phonetic1 = self.apply_phonetic_transform(text1)
        phonetic2 = self.apply_phonetic_transform(text2)
        
        return SequenceMatcher(None, phonetic1, phonetic2).ratio()
    
    def apply_phonetic_transform(self, text):
        """Apply phonetic transformations for better Hinglish matching"""
        # Convert common phonetic variations
        transformations = {
            'ph': 'f', 'gh': 'g', 'kh': 'k', 'th': 't', 'dh': 'd', 'bh': 'b',
            'ch': 'c', 'sh': 's', 'zh': 'z', 'oo': 'u', 'aa': 'a', 'ee': 'i',
            'v': 'w', 'w': 'v', 'z': 's', 's': 'z', 'x': 'ks'
        }
        
        result = text
        for original, replacement in transformations.items():
            result = result.replace(original, replacement)
        
        return result
    
    def initialize_vectorizer(self, keywords):
        """Initialize TF-IDF vectorizer with keywords"""
        if SKLEARN_AVAILABLE:
            # Preprocess keywords
            processed_keywords = [self.preprocess_text(k) for k in keywords]
            # Fit vectorizer
            self.vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
            self.vectorizer.fit(processed_keywords)
            logger.info("TF-IDF vectorizer initialized with %d keywords", len(keywords))
    
    def extract_entities(self, text):
        """Extract potential entities from text (names, locations, etc.)"""
        # Basic entity extraction based on capitalization
        entities = re.findall(r'\b[A-Z][a-z]+\b', text)
        return entities
    
    def extract_context(self, text):
        """Extract context clues from text"""
        context = {}
        
        # Time context
        time_words = ['today', 'tomorrow', 'yesterday', 'morning', 'afternoon', 'evening', 'night']
        context['time_indicators'] = [word for word in time_words if word in text.lower()]
        
        # Question detection
        context['is_question'] = '?' in text or any(q in text.lower() for q in ['what', 'who', 'where', 'when', 'why', 'how'])
        
        # Sentiment analysis (very basic)
        positive_words = ['good', 'great', 'excellent', 'amazing', 'happy', 'love', 'like', 'thanks']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'upset']
        
        positive_count = sum(1 for word in positive_words if word in text.lower())
        negative_count = sum(1 for word in negative_words if word in text.lower())
        
        if positive_count > negative_count:
            context['sentiment'] = 'positive'
        elif negative_count > positive_count:
            context['sentiment'] = 'negative'
        else:
            context['sentiment'] = 'neutral'
        
        return context

class TelegramBot:
    def __init__(self):
        self.client = TelegramClient('bot_session', API_ID, API_HASH)
        self.responses = {}
        self.nlp = NLPProcessor()
        self.user_context = {}  # Store conversation context for each user
        self.load_responses()
        self.setup_handlers()
        
    def load_responses(self):
        """Load responses from JSON file"""
        try:
            with open(RESPONSES_FILE, 'r', encoding='utf-8') as f:
                self.responses = json.load(f)
            logger.info("Responses loaded successfully")
            
            # Initialize NLP vectorizer with keywords
            if self.responses:
                self.nlp.initialize_vectorizer(list(self.responses.keys()))
            logger.info("Load response last functions")
        except Exception as e:
            logger.error(f"Error loading responses: {e}")
            # Create default responses if file doesn't exist
            self.responses = {
                "Hi": {
                    "type": "text",
                    "content": ["Hi"]
                },
                "(.*)": {
                    "type": "text",
                    "content": ["default!", "araam se", "eat food"]
                }
            }
            self.save_responses()
            
            # Initialize NLP vectorizer with default keywords
            self.nlp.initialize_vectorizer(list(self.responses.keys()))

    def save_conversation(self, message_text, selected_content, user_id):
        """Save conversation to JSON file"""
        try:
            # Load current responses
            with open(CONVERSATION_FILE, 'r', encoding='utf-8') as f:
                responses = json.load(f)
            
            # Update response
            response_data = {
                "type": 'text',
                "content": selected_content,
            }
            responses[message_text] = response_data
            
            # Save updated responses
            with open(CONVERSATION_FILE, 'w', encoding='utf-8') as f:
                json.dump(responses, f, indent=4)

            logger.info("Conversation saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving responses: {e}")
            return False

    def save_responses(self):
        """Save responses to JSON file"""
        try:
            with open(RESPONSES_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.responses, f, indent=4)
            logger.info("Responses saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving responses: {e}")
            return False
    
    def setup_handlers(self):
        """Set up message handlers"""
        logger.info("Setup handler starts")

        @self.client.on(events.NewMessage(incoming=True))
        async def handler(event):
            # Ignore messages from channels or non-private chats
            logger.info("Setup handler 2")
            if not event.is_private:
                return
            
            # Process the message
            logger.info("Setup handler 3")
            await self.process_message(event)
    
    def get_user_context(self, user_id):
        """Get or create user context"""
        logger.info("Get user context")
        if user_id not in self.user_context:
            self.user_context[user_id] = {
                'last_messages': [],  # Store last few messages for context
                'last_interaction': time.time(),
                'session_start': time.time(),
                'message_count': 0
            }
        return self.user_context[user_id]
    
    def update_user_context(self, user_id, message_text):
        """Update user context with new message"""
        context = self.get_user_context(user_id)
        
        # Update last messages (keep last 5)
        context['last_messages'].append(message_text)
        if len(context['last_messages']) > 5:
            context['last_messages'].pop(0)
        
        # Update interaction time and message count
        context['last_interaction'] = time.time()
        context['message_count'] += 1
        
        # Extract NLP context
        nlp_context = self.nlp.extract_context(message_text)
        context.update(nlp_context)
        
        return context

    def get_response_for_message(self, message_text):
        """
        Returns the best response for the given message_text:
        1. Exact match in responses
        2. Wildcard match in responses
        3. NLP fuzzy match (only if NLP is enabled)
        """
        # 1. Exact match (case-insensitive)
        for key in self.responses:
            if message_text.strip().lower() == key.strip().lower():
                return self.responses[key], 'exact'

        # 2. Wildcard match (e.g., "hello*", "*bye")
        for key in self.responses:
            if '*' in key:
                if fnmatch.fnmatch(message_text.strip().lower(), key.strip().lower()):
                    return self.responses[key], 'wildcard'

        # 3. NLP fuzzy match (only if NLP is enabled)
        if hasattr(self, 'nlp') and self.nlp and self.nlp.nlp_enabled:
            matched_keyword, confidence = self.nlp.find_best_match(
                message_text,
                list(self.responses.keys()),
                threshold=0.4
            )
            if matched_keyword:
                return self.responses[matched_keyword], 'nlp'

        # No match found
        return None, None

    async def process_message(self, event):
        """Process incoming message and send response"""
        logger.info("Start process message")
        message_text = event.message.text
        user_id = event.sender_id

        await self.client.send_read_acknowledge(event.sender_id)

        # Update user context
        user_context = self.update_user_context(user_id, message_text)
        print(f'user_context {user_context}\n')

        # Use new matching logic
        response_data, match_type = self.get_response_for_message(message_text)
        response_sent = False

        if response_data:
            logger.info(f'Match type: {match_type}, response_data: {response_data}\n')
            typing_delay = response_data.get("typing_delay", [1, 3])
            delay = random.uniform(typing_delay[0], typing_delay[1])

            # Add slight randomness to typing time based on message length
            char_typing_time = len(message_text) * 0.05
            delay = min(delay + random.uniform(0, char_typing_time), 6.0)

            async with self.client.action(event.chat_id, 'typing'):
                await asyncio.sleep(delay)

            response_type = response_data.get("type", "text")
            content = response_data.get("content", ["No response available"])

            # Select random response if multiple options are available
            if isinstance(content, list) and response_type == "text":
                selected_content = random.choice(content)
            else:
                selected_content = content
            
            # Send appropriate response based on type
            if response_type == "text":
                await event.respond(selected_content)
                self.save_conversation(message_text, selected_content, user_id)
                
            elif response_type == "image":
                image_path = os.path.join(IMAGES_DIR, selected_content)
                if os.path.exists(image_path):
                    await self.client.send_file(
                        event.chat_id,
                        image_path,
                        caption=response_data.get("caption", "")
                    )
                    self.save_conversation(message_text, f"Image: {selected_content}", user_id)
                else:
                    await event.respond(f"Image not found: {selected_content}")
                    
            elif response_type == "multiple-images":
                # NEW: Handle multiple images response type
                image_list = content  # content should be a list of image filenames
                caption = response_data.get("caption", "")
                
                if isinstance(image_list, list) and image_list:
                    sent_images = []
                    
                    for i, image_filename in enumerate(image_list):
                        image_path = os.path.join(IMAGES_DIR, image_filename)
                        
                        if os.path.exists(image_path):
                            try:
                                # For the first image, include the caption
                                # For subsequent images, send without caption to avoid repetition
                                image_caption = caption if i == 0 and caption else ""
                                
                                await self.client.send_file(
                                    event.chat_id,
                                    image_path,
                                    caption=image_caption
                                )
                                sent_images.append(image_filename)
                                
                                # Small delay between images to avoid rate limiting
                                if i < len(image_list) - 1:  # Don't delay after last image
                                    await asyncio.sleep(0.5)
                                    
                            except Exception as e:
                                logger.error(f"Failed to send image {image_filename}: {e}")
                                await event.respond(f"Failed to send image: {image_filename}")
                        else:
                            logger.error(f"Image not found: {image_path}")
                            await event.respond(f"Image not found: {image_filename}")
                    
                    if sent_images:
                        self.save_conversation(message_text, f"Multiple images: {', '.join(sent_images)}", user_id)
                    else:
                        await event.respond("No images could be sent")
                else:
                    await event.respond("No images configured for this response")
                    
            elif response_type == "audio":
                audio_path = os.path.join(AUDIO_DIR, selected_content)
                if os.path.exists(audio_path):
                    await self.client.send_file(
                        event.chat_id,
                        audio_path,
                        voice_note=True
                    )
                    self.save_conversation(message_text, f"Audio: {selected_content}", user_id)
                else:
                    await event.respond(f"Audio not found: {selected_content}")

            response_sent = True

        if not response_sent:
            # save unknown messages
            selected_content = ''
            self.save_conversation(message_text, selected_content, user_id)

    async def start(self):
        """Start the bot"""
        try:
            await self.client.start(phone=PHONE)
            logger.info("Bot started successfully")
            
            # Print some account info
            me = await self.client.get_me()
            logger.info(f"Bot running as: {me.first_name} (@{me.username if me.username else 'No username'})")
            
            self.is_running = True
            
            # Run the client until disconnected
            await self.client.run_until_disconnected()
            
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            self.is_running = False
            raise
    
    def run(self):
        """Run the bot in a new event loop"""
        try:
            # Create a new event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # Run the bot
            self.loop.run_until_complete(self.start())
            
        except Exception as e:
            logger.error(f"Error running bot: {e}")
            self.is_running = False
        finally:
            if self.loop and not self.loop.is_closed():
                self.loop.close()
    
    def stop(self):
        """Stop the bot"""
        if hasattr(self, 'is_running') and self.is_running and self.client.is_connected():
            logger.info("Stopping bot...")
            if hasattr(self, 'loop') and self.loop:
                self.loop.call_soon_threadsafe(self.client.disconnect)
            self.is_running = False
