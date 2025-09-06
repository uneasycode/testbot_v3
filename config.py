
# config.py
"""
Configuration settings for the Telegram bot and Flask application
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Telegram API credentials
API_ID = os.getenv('API_ID')
API_HASH = os.getenv('API_HASH')
PHONE = os.getenv('PHONE')

# Flask settings
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key')
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'

# Application paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESPONSES_FILE = os.path.join(BASE_DIR, 'responses.json')
CONVERSATION_FILE = os.path.join(BASE_DIR, 'conversation.json')
NLP_CONFIG_FILE = os.path.join(BASE_DIR, 'nlp_config.json')
MEDIA_DIR = os.path.join(BASE_DIR, 'media')
IMAGES_DIR = os.path.join(MEDIA_DIR, 'images')
AUDIO_DIR = os.path.join(MEDIA_DIR, 'audio')

# Create directories if they don't exist
for directory in [MEDIA_DIR, IMAGES_DIR, AUDIO_DIR]:
    os.makedirs(directory, exist_ok=True)
