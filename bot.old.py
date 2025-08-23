# bot.py - FIXED VERSION with Memory Management

"""
Telegram bot implementation using Telethon with NLP enhancements
"""

import os
import json
import random
import asyncio
import logging
import time
import re
from collections import Counter, OrderedDict
from difflib import SequenceMatcher
import numpy as np
from telethon import TelegramClient, events
from telethon.tl.types import InputMediaPhoto, InputMediaDocument
from config import API_ID, API_HASH, PHONE, RESPONSES_FILE, IMAGES_DIR, AUDIO_DIR, CONVERSATION_FILE
import fnmatch

# SSL bypass for certificate issues
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Optional NLP libraries with fallback
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
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
    """Natural Language Processing helper class for Hinglish"""

    def __init__(self):
        self.initialize_nlp_tools()

    def initialize_nlp_tools(self):
        """Initialize NLP tools for Hinglish"""
        # Common Hinglish typos and slang
        self.common_typos = {
            'kya': 'kya', 'kyu': 'kyun', 'plz': 'please', 'pls': 'please',
            'thx': 'thanks', 'sry': 'sorry', 'frnd': 'friend', 'bcoz': 'because',
            'gud': 'good', 'hw': 'how', 'u': 'you', 'ur': 'your', 'r': 'are',
            'h': 'hai', 'nahi': 'nahi', 'haan': 'haan', 'okie': 'ok', 'tm': 'tum',
            'mujhe': 'mujhe', 'acha': 'acha', 'kaise': 'kaise', 'kese': 'kaise'
        }

        # Hinglish stopwords
        self.stop_words = set([
            'hai', 'ka', 'ki', 'ke', 'ko', 'se', 'me', 'mein', 'par', 'aur', 'ya', 'to', 'bhi', 'wo', 'woh', 'ye', 'yeh',
            'the', 'is', 'are', 'was', 'were', 'a', 'an', 'and', 'or', 'but', 'if', 'so', 'of', 'on', 'in', 'for', 'with'
        ])

        # Initialize spell checker if available
        if SPELLCHECKER_AVAILABLE:
            try:
                self.spell = SpellChecker(language=None)
            except Exception as e:
                logger.warning(f"Could not initialize spell checker: {e}")
                self.spell = None
        else:
            self.spell = None

        if SKLEARN_AVAILABLE:
            self.vectorizer = None

    def preprocess_text(self, text):
        """Preprocess Hinglish text"""
        if not text:
            return ""
            
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        # Fix common typos
        words = [self.common_typos.get(word, word) for word in words]
        
        # Remove stopwords
        tokens = [word for word in words if word not in self.stop_words]
        
        processed_text = ' '.join(tokens)
        
        # Spelling correction (if available and words are long enough)
        if self.spell and len(processed_text) > 3:
            try:
                corrected_words = []
                for word in processed_text.split():
                    if len(word) > 3 and word in self.spell.unknown([word]):
                        correction = self.spell.correction(word)
                        corrected_words.append(correction if correction else word)
                    else:
                        corrected_words.append(word)
                processed_text = ' '.join(corrected_words)
            except Exception as e:
                logger.warning(f"Spell check failed: {e}")
        
        return processed_text

    def find_best_match(self, input_text, keywords, threshold=0.6):
        """Find the best matching keyword for the input text"""
        if not input_text or not keywords:
            return None, 0
            
        # Preprocess input text
        processed_input = self.preprocess_text(input_text)
        
        # Use TF-IDF if available
        if SKLEARN_AVAILABLE and self.vectorizer is not None:
            try:
                input_vector = self.vectorizer.transform([processed_input])
                
                best_match = None
                best_score = 0
                for keyword in keywords:
                    keyword_vector = self.vectorizer.transform([keyword])
                    similarity = cosine_similarity(input_vector, keyword_vector)[0][0]
                    if similarity > best_score:
                        best_score = similarity
                        best_match = keyword
                
                if best_score >= threshold:
                    return best_match, best_score
            except Exception as e:
                logger.warning(f"TF-IDF matching failed: {e}")
        
        # Fallback to simpler matching
        best_match = None
        best_score = 0
        
        for keyword in keywords:
            # Check for direct containment
            if keyword.lower() in processed_input:
                return keyword, 1.0
            
            # Calculate string similarity
            similarity = self.calculate_similarity(processed_input, keyword.lower())
            if similarity > best_score:
                best_score = similarity
                best_match = keyword
        
        if best_score >= threshold:
            return best_match, best_score
        
        return None, 0

    def calculate_similarity(self, text1, text2):
        """Calculate similarity between two strings"""
        if not text1 or not text2:
            return 0
        return SequenceMatcher(None, text1, text2).ratio()

    def initialize_vectorizer(self, keywords):
        """Initialize TF-IDF vectorizer with keywords"""
        if SKLEARN_AVAILABLE and keywords:
            try:
                processed_keywords = [self.preprocess_text(k) for k in keywords]
                processed_keywords = [k for k in processed_keywords if k]  # Remove empty strings
                
                if processed_keywords:
                    self.vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
                    self.vectorizer.fit(processed_keywords)
                    logger.info(f"TF-IDF vectorizer initialized with {len(keywords)} keywords")
            except Exception as e:
                logger.error(f"Failed to initialize vectorizer: {e}")
                self.vectorizer = None

class TelegramBot:
    # FIXED: Added memory management and better error handling
    MAX_USER_CONTEXT_SIZE = 1000  # Maximum number of user contexts to keep
    CONTEXT_CLEANUP_INTERVAL = 3600  # Cleanup old contexts every hour
    
    def __init__(self):
        self.client = TelegramClient('bot_session', API_ID, API_HASH)
        self.responses = {}
        self.nlp = NLPProcessor()
        self.user_context = OrderedDict()  # Use OrderedDict for LRU-like behavior
        self.last_cleanup_time = time.time()
        self.is_running = False
        self.load_responses()
        self.setup_handlers()

    def cleanup_user_contexts(self):
        """Clean up old user contexts to prevent memory leaks"""
        current_time = time.time()
        
        # Remove contexts older than 24 hours
        cutoff_time = current_time - 24 * 3600
        users_to_remove = []
        
        for user_id, context in self.user_context.items():
            if context.get('last_interaction', 0) < cutoff_time:
                users_to_remove.append(user_id)
        
        for user_id in users_to_remove:
            del self.user_context[user_id]
        
        # Keep only the most recent MAX_USER_CONTEXT_SIZE contexts
        if len(self.user_context) > self.MAX_USER_CONTEXT_SIZE:
            # Remove oldest entries
            while len(self.user_context) > self.MAX_USER_CONTEXT_SIZE:
                self.user_context.popitem(last=False)
        
        self.last_cleanup_time = current_time
        logger.info(f"Cleaned up {len(users_to_remove)} old user contexts")

    def safe_json_operation(self, file_path, operation, data=None, default=None):
        """Safely perform JSON operations with error handling"""
        try:
            if operation == 'load':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif operation == 'save' and data is not None:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                return True
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}")
            return default if default is not None else {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            return default if default is not None else {}
        except Exception as e:
            logger.error(f"JSON operation failed for {file_path}: {e}")
            return default if default is not None else ({} if operation == 'load' else False)

    def load_responses(self):
        """Load responses from JSON file with error handling"""
        try:
            self.responses = self.safe_json_operation(RESPONSES_FILE, 'load', default={
                "Hi": {
                    "type": "text",
                    "content": ["Hi"]
                },
                "(.*)": {
                    "type": "text",
                    "content": ["default!", "araam se", "eat food"]
                }
            })
            
            logger.info(f"Loaded {len(self.responses)} responses")
            
            # Initialize NLP vectorizer
            if self.responses:
                self.nlp.initialize_vectorizer(list(self.responses.keys()))
                
        except Exception as e:
            logger.error(f"Error loading responses: {e}")
            # Create minimal default responses
            self.responses = {
                "hi": {"type": "text", "content": ["Hello!"]},
                "(.*)": {"type": "text", "content": ["I don't understand"]}
            }

    def save_conversation(self, message_text, selected_content, user_id):
        """Save conversation to JSON file with error handling"""
        try:
            # Load existing conversations
            conversations = self.safe_json_operation(CONVERSATION_FILE, 'load', default={})
            
            # Prepare conversation data
            response_data = {
                "type": 'text',
                "content": selected_content,
                # Could add timestamp and user info later
            }
            
            conversations[message_text] = response_data
            
            # Save with error handling
            return self.safe_json_operation(CONVERSATION_FILE, 'save', conversations)
            
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return False

    def setup_handlers(self):
        """Set up message handlers with better error handling"""
        logger.info("Setting up message handlers")

        @self.client.on(events.NewMessage(incoming=True))
        async def handler(event):
            try:
                # Ignore non-private messages
                if not event.is_private:
                    return

                # Process the message
                await self.process_message(event)
                
            except Exception as e:
                logger.error(f"Error in message handler: {e}")
                try:
                    await event.respond("Sorry, I encountered an error processing your message.")
                except Exception as inner_e:
                    logger.error(f"Error sending error message: {inner_e}")

    def get_user_context(self, user_id):
        """Get or create user context with memory management"""
        # Periodic cleanup
        if time.time() - self.last_cleanup_time > self.CONTEXT_CLEANUP_INTERVAL:
            self.cleanup_user_contexts()
        
        if user_id not in self.user_context:
            self.user_context[user_id] = {
                'last_messages': [],
                'last_interaction': time.time(),
                'session_start': time.time(),
                'message_count': 0
            }
        else:
            # Move to end (most recently used)
            self.user_context.move_to_end(user_id)
        
        return self.user_context[user_id]

    def update_user_context(self, user_id, message_text):
        """Update user context with new message"""
        try:
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
            
        except Exception as e:
            logger.error(f"Error updating user context: {e}")
            return {}

    def get_response_for_message(self, message_text):
        """Get the best response for the given message with improved matching"""
        if not message_text:
            return None, None
            
        try:
            # 1. Exact match (case-insensitive)
            for key in self.responses:
                if message_text.strip().lower() == key.strip().lower():
                    return self.responses[key], 'exact'

            # 2. Wildcard match
            for key in self.responses:
                if '*' in key:
                    try:
                        if fnmatch.fnmatch(message_text.strip().lower(), key.strip().lower()):
                            return self.responses[key], 'wildcard'
                    except Exception as e:
                        logger.warning(f"Wildcard matching failed for {key}: {e}")

            # 3. NLP fuzzy match
            if hasattr(self, 'nlp') and self.nlp:
                try:
                    matched_keyword, confidence = self.nlp.find_best_match(
                        message_text,
                        list(self.responses.keys()),
                        threshold=0.6
                    )
                    if matched_keyword:
                        return self.responses[matched_keyword], 'nlp'
                except Exception as e:
                    logger.error(f"NLP matching failed: {e}")

            # 4. Regex/catch-all match
            for key in self.responses:
                if key == "(.*)" or key.startswith("(.*)"):
                    return self.responses[key], 'catchall'
                    
        except Exception as e:
            logger.error(f"Error getting response for message: {e}")
        
        return None, None

    async def process_message(self, event):
        """Process incoming message with comprehensive error handling"""
        try:
            message_text = event.message.text
            if not message_text:
                return
                
            user_id = event.sender_id
            logger.info(f"Processing message from user {user_id}: {message_text[:50]}...")

            # Mark message as read
            try:
                await self.client.send_read_acknowledge(event.sender_id)
            except Exception as e:
                logger.warning(f"Failed to send read acknowledge: {e}")

            # Update user context
            user_context = self.update_user_context(user_id, message_text)

            # Get response
            response_data, match_type = self.get_response_for_message(message_text)
            response_sent = False

            if response_data:
                logger.info(f'Match type: {match_type} for message: {message_text[:30]}...')
                
                # Handle typing delay
                typing_delay = response_data.get("typing_delay", [1, 3])
                try:
                    delay = random.uniform(typing_delay[0], typing_delay[1])
                    # Add slight randomness based on message length
                    char_typing_time = min(len(message_text) * 0.05, 2.0)
                    delay = min(delay + random.uniform(0, char_typing_time), 6.0)

                    async with self.client.action(event.chat_id, 'typing'):
                        await asyncio.sleep(delay)
                except Exception as e:
                    logger.warning(f"Typing simulation failed: {e}")

                # Prepare response content
                response_type = response_data.get("type", "text")
                content = response_data.get("content", ["No response available"])

                # Select random response if multiple options
                if isinstance(content, list) and content:
                    selected_content = random.choice(content)
                else:
                    selected_content = content if content else "No response available"

                # Send appropriate response based on type
                try:
                    if response_type == "text":
                        if selected_content:  # Only send non-empty responses
                            await event.respond(str(selected_content))
                            response_sent = True

                    elif response_type == "image":
                        image_path = os.path.join(IMAGES_DIR, selected_content)
                        if os.path.exists(image_path):
                            await self.client.send_file(
                                event.chat_id,
                                image_path,
                                caption=response_data.get("caption", "")
                            )
                            response_sent = True
                        else:
                            await event.respond(f"Image not found: {selected_content}")
                            response_sent = True

                    elif response_type == "audio":
                        audio_path = os.path.join(AUDIO_DIR, selected_content)
                        if os.path.exists(audio_path):
                            await self.client.send_file(
                                event.chat_id,
                                audio_path,
                                voice_note=True
                            )
                            response_sent = True
                        else:
                            await event.respond(f"Audio not found: {selected_content}")
                            response_sent = True


                    elif response_type == "multiple_images":
                        images = [
                            open('image1.jpg', 'rb'),
                            open('image2.jpg', 'rb'),
                            open('image3.jpg', 'rb')
                        ]
                        media_group = [InputMediaPhoto(img) for img in images]
                        await self.client.bot.send_media_group(chat_id=update.effective_chat.id, media=media_group)
                        for img in images:
                            img.close()
                        response_sent = True
                                
                    # Save conversation
                    if response_sent:
                        self.save_conversation(message_text, selected_content, user_id)

                except Exception as e:
                    logger.error(f"Error sending response: {e}")
                    try:
                        await event.respond("Sorry, I had trouble responding to your message.")
                    except Exception as inner_e:
                        logger.error(f"Failed to send error message: {inner_e}")

            if not response_sent:
                # Save unknown messages
                self.save_conversation(message_text, '', user_id)
                logger.info(f"No response found for message: {message_text}")

        except Exception as e:
            logger.error(f"Critical error in process_message: {e}")

    async def start(self):
        """Start the bot with comprehensive error handling"""
        try:
            # Validate required credentials
            if not all([API_ID, API_HASH, PHONE]):
                raise ValueError("Missing required credentials: API_ID, API_HASH, PHONE")

            await self.client.start(phone=PHONE)
            logger.info("Bot started successfully")

            # Get bot information
            try:
                me = await self.client.get_me()
                logger.info(f"Bot running as: {me.first_name} (@{me.username if me.username else 'No username'})")
            except Exception as e:
                logger.warning(f"Could not get bot info: {e}")

            self.is_running = True

            # Run until disconnected
            await self.client.run_until_disconnected()

        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            self.is_running = False
            raise

    def stop(self):
        """Stop the bot gracefully"""
        if self.is_running and self.client.is_connected():
            logger.info("Stopping bot...")
            try:
                asyncio.create_task(self.client.disconnect())
                self.is_running = False
            except Exception as e:
                logger.error(f"Error stopping bot: {e}")

    def extract_context(self, text):
        """Extract context clues from text (moved from NLPProcessor for consistency)"""
        context = {}
        
        # Time context
        time_words = ['today', 'tomorrow', 'yesterday', 'morning', 'afternoon', 'evening', 'night', 
                     'aaj', 'kal', 'subah', 'shaam', 'raat']
        context['time_indicators'] = [word for word in time_words if word in text.lower()]
        
        # Question detection
        context['is_question'] = ('?' in text or 
                                any(q in text.lower() for q in ['what', 'who', 'where', 'when', 'why', 'how',
                                                               'kya', 'kaun', 'kahan', 'kab', 'kyu', 'kaise']))
        
        # Basic sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'amazing', 'happy', 'love', 'like', 'thanks', 
                         'acha', 'badhiya', 'shukriya', 'dhanyawad']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'upset',
                         'bura', 'ganda', 'nafrat', 'dukh']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            context['sentiment'] = 'positive'
        elif negative_count > positive_count:
            context['sentiment'] = 'negative'
        else:
            context['sentiment'] = 'neutral'
        
        return context