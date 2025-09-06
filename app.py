# app_updated.py - Flask app with multiple-images support

"""
Flask application for managing the Telegram bot
Now includes support for multiple-images response type
"""

import json
import os
import threading
import logging
import asyncio
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from bot import TelegramBot
from config import (
    SECRET_KEY, DEBUG, RESPONSES_FILE,
    IMAGES_DIR, AUDIO_DIR, NLP_CONFIG_FILE
)
from learning_module import ConversationLearner

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['DEBUG'] = DEBUG
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# Create bot instance
bot = TelegramBot()
bot_thread = None
bot_running = False

def start_bot_thread():
    """Start the bot in a separate thread with proper asyncio event loop"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(bot.start())

@app.route('/')
def index():
    logger.info("We are in app.route /")
    """Render the main page"""
    try:
        with open(RESPONSES_FILE, 'r', encoding='utf-8') as f:
            responses = json.load(f)
    except FileNotFoundError:
        responses = {}
    
    # Get list of available media files
    try:
        images = [f for f in os.listdir(IMAGES_DIR) if os.path.isfile(os.path.join(IMAGES_DIR, f))]
        audio = [f for f in os.listdir(AUDIO_DIR) if os.path.isfile(os.path.join(AUDIO_DIR, f))]
    except FileNotFoundError:
        images = []
        audio = []
    
    return render_template('index.html', responses=responses, images=images, audio=audio)

@app.route('/api/responses', methods=['GET'])
def get_responses():
    """API endpoint to get all responses"""
    logger.info("We are in app.route /api/responses")
    try:
        with open(RESPONSES_FILE, 'r', encoding='utf-8') as f:
            responses = json.load(f)
        return jsonify(responses)
    except FileNotFoundError:
        return jsonify({})

@app.route('/api/responses/<keyword>', methods=['GET'])
def get_response(keyword):
    logger.info(f"We are in app.route /api/responses/{keyword}")
    """API endpoint to get a specific response"""
    try:
        with open(RESPONSES_FILE, 'r', encoding='utf-8') as f:
            responses = json.load(f)
        
        if keyword in responses:
            return jsonify(responses[keyword])
        else:
            return jsonify({"error": "Keyword not found"}), 404
    except FileNotFoundError:
        return jsonify({"error": "Responses file not found"}), 404

@app.route('/api/responses', methods=['POST'])
def add_response():
    """API endpoint to add a new response"""
    logger.info("We are in app.route /api/responses POST")
    data = request.json
    
    if not data or 'keyword' not in data or 'response_data' not in data:
        return jsonify({"error": "Invalid data"}), 400
    
    keyword = data['keyword']
    response_data = data['response_data']
    
    # Load current responses
    try:
        with open(RESPONSES_FILE, 'r', encoding='utf-8') as f:
            responses = json.load(f)
    except FileNotFoundError:
        responses = {}
    
    # Add new response
    responses[keyword] = response_data
    
    # Save updated responses
    try:
        with open(RESPONSES_FILE, 'w', encoding='utf-8') as f:
            json.dump(responses, f, indent=4)
        
        # Update bot's responses
        bot.load_responses()
        
        return jsonify({"success": True, "message": "Response added successfully"})
    except Exception as e:
        logger.error(f"Error saving response: {e}")
        return jsonify({"error": "Failed to save response"}), 500

@app.route('/api/responses/<keyword>', methods=['PUT'])
def update_response(keyword):
    """API endpoint to update an existing response"""
    logger.info(f"We are in app.route /api/responses/{keyword} PUT")
    data = request.json
    
    if not data or 'response_data' not in data:
        return jsonify({"error": "Invalid data"}), 400
    
    response_data = data['response_data']
    
    # Load current responses
    try:
        with open(RESPONSES_FILE, 'r', encoding='utf-8') as f:
            responses = json.load(f)
    except FileNotFoundError:
        return jsonify({"error": "Responses file not found"}), 404
    
    if keyword not in responses:
        return jsonify({"error": "Keyword not found"}), 404
    
    # Update response
    responses[keyword] = response_data
    
    # Save updated responses
    try:
        with open(RESPONSES_FILE, 'w', encoding='utf-8') as f:
            json.dump(responses, f, indent=4)
        
        # Update bot's responses
        bot.load_responses()
        
        return jsonify({"success": True, "message": "Response updated successfully"})
    except Exception as e:
        logger.error(f"Error updating response: {e}")
        return jsonify({"error": "Failed to update response"}), 500

@app.route('/api/responses/<keyword>', methods=['DELETE'])
def delete_response(keyword):
    """API endpoint to delete a response"""
    logger.info(f"We are in app.route /api/responses/{keyword} DELETE")

    # Load current responses
    try:
        with open(RESPONSES_FILE, 'r', encoding='utf-8') as f:
            responses = json.load(f)
    except FileNotFoundError:
        return jsonify({"error": "Responses file not found"}), 404
    
    if keyword not in responses:
        return jsonify({"error": "Keyword not found"}), 404
    
    # Delete response
    del responses[keyword]
    
    # Save updated responses
    try:
        with open(RESPONSES_FILE, 'w', encoding='utf-8') as f:
            json.dump(responses, f, indent=4)
        
        # Update bot's responses
        bot.load_responses()
        
        return jsonify({"success": True, "message": "Response deleted successfully"})
    except Exception as e:
        logger.error(f"Error deleting response: {e}")
        return jsonify({"error": "Failed to delete response"}), 500

@app.route('/upload/media', methods=['POST'])
def upload_media():
    """Upload media files (images or audio)"""
    logger.info("We are in app.route /upload/media")

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    media_type = request.form.get('type')
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if media_type not in ['image', 'audio']:
        return jsonify({"error": "Invalid media type"}), 400
    
    filename = secure_filename(file.filename)
    
    try:
        if media_type == 'image':
            file.save(os.path.join(IMAGES_DIR, filename))
        else:  # audio
            file.save(os.path.join(AUDIO_DIR, filename))
        
        return jsonify({
            "success": True,
            "message": f"{media_type.capitalize()} uploaded successfully",
            "filename": filename
        })
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return jsonify({"error": f"Failed to upload file: {str(e)}"}), 500

@app.route('/edit/<keyword>', methods=['GET'])
def edit_response(keyword):
    """Render the edit response page"""
    logger.info(f"We are in app.route /edit/{keyword} GET")

    try:
        with open(RESPONSES_FILE, 'r', encoding='utf-8') as f:
            responses = json.load(f)
    except FileNotFoundError:
        flash("Responses file not found", "error")
        return redirect(url_for('index'))
    
    if keyword not in responses:
        flash(f"Keyword '{keyword}' not found", "error")
        return redirect(url_for('index'))
    
    # Get list of available media files
    try:
        images = [f for f in os.listdir(IMAGES_DIR) if os.path.isfile(os.path.join(IMAGES_DIR, f))]
        audio = [f for f in os.listdir(AUDIO_DIR) if os.path.isfile(os.path.join(AUDIO_DIR, f))]
    except FileNotFoundError:
        images = []
        audio = []
    
    return render_template(
        'edit_response.html',
        keyword=keyword,
        response_data=responses[keyword],
        images=images,
        audio=audio
    )

@app.route('/edit/<keyword>', methods=['POST'])
def submit_edit(keyword):
    """Process edit response form submission"""
    logger.info(f"We are in app.route /edit/{keyword} POST")

    response_type = request.form.get('type')
    
    # UPDATED: Handle multiple-images type
    if response_type == 'text':
        content = request.form.get('content', '').split('\n')
        content = [line.strip() for line in content if line.strip()]
        if not content:
            flash("Please enter at least one text response", "error")
            return redirect(url_for('edit_response', keyword=keyword))
            
    elif response_type == 'image':
        # Single image response
        content = request.form.get('image_file', '')
        if not content:
            flash("Please select an image file", "error")
            return redirect(url_for('edit_response', keyword=keyword))
            
    elif response_type == 'multiple-images':
        # NEW: Multiple images response
        content = request.form.getlist('multiple_image_files')  # Get list of selected images
        if not content:
            flash("Please select at least one image file", "error")
            return redirect(url_for('edit_response', keyword=keyword))
            
    elif response_type == 'audio':
        # Audio response
        content = request.form.get('audio_file', '')
        if not content:
            flash("Please select an audio file", "error")
            return redirect(url_for('edit_response', keyword=keyword))
    else:
        flash("Invalid response type", "error")
        return redirect(url_for('edit_response', keyword=keyword))
    
    response_data = {
        "type": response_type,
        "content": content,
    }
    
    if response_type in ['image', 'multiple-images']:
        caption = request.form.get('caption', '')
        response_data["caption"] = caption
    
    # Load current responses
    try:
        with open(RESPONSES_FILE, 'r', encoding='utf-8') as f:
            responses = json.load(f)
    except FileNotFoundError:
        responses = {}
    
    # Update response
    responses[keyword] = response_data
    
    # Save updated responses
    try:
        with open(RESPONSES_FILE, 'w', encoding='utf-8') as f:
            json.dump(responses, f, indent=4)
        
        # Update bot's responses
        bot.load_responses()
        
        flash(f"Response for '{keyword}' updated successfully", "success")
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error updating response: {e}")
        flash("Failed to update response", "error")
        return redirect(url_for('edit_response', keyword=keyword))

@app.route('/add', methods=['GET'])
def add_response_form():
    """Render the add response page"""
    logger.info("We are in app.route /add GET")

    # Get list of available media files
    try:
        images = [f for f in os.listdir(IMAGES_DIR) if os.path.isfile(os.path.join(IMAGES_DIR, f))]
        audio = [f for f in os.listdir(AUDIO_DIR) if os.path.isfile(os.path.join(AUDIO_DIR, f))]
    except FileNotFoundError:
        images = []
        audio = []
    
    return render_template('edit_response.html', images=images, audio=audio, is_new=True)

@app.route('/add', methods=['POST'])
def submit_add():
    """Process add response form submission"""
    logger.info("We are in app.route /add POST")

    keyword = request.form.get('keyword', '').strip()
    response_type = request.form.get('type')
    
    if not keyword:
        flash("Keyword cannot be empty", "error")
        return redirect(url_for('add_response_form'))
    
    # Load current responses
    try:
        with open(RESPONSES_FILE, 'r', encoding='utf-8') as f:
            responses = json.load(f)
    except FileNotFoundError:
        responses = {}
    
    if keyword in responses:
        flash(f"Keyword '{keyword}' already exists", "error")
        return redirect(url_for('add_response_form'))
    
    # UPDATED: Handle multiple-images type
    if response_type == 'text':
        content = request.form.get('content', '').split('\n')
        content = [line.strip() for line in content if line.strip()]
        if not content:
            flash("Please enter at least one text response", "error")
            return redirect(url_for('add_response_form'))
            
    elif response_type == 'image':
        # Single image response
        content = request.form.get('image_file', '')
        if not content:
            flash("Please select an image file", "error")
            return redirect(url_for('add_response_form'))
            
    elif response_type == 'multiple-images':
        # NEW: Multiple images response
        content = request.form.getlist('multiple_image_files')  # Get list of selected images
        if not content:
            flash("Please select at least one image file", "error")
            return redirect(url_for('add_response_form'))
            
    elif response_type == 'audio':
        # Audio response
        content = request.form.get('audio_file', '')
        if not content:
            flash("Please select an audio file", "error")
            return redirect(url_for('add_response_form'))
    else:
        flash("Please select a response type", "error")
        return redirect(url_for('add_response_form'))
    
    response_data = {
        "type": response_type,
        "content": content,
    }
    
    if response_type in ['image', 'multiple-images']:
        caption = request.form.get('caption', '')
        response_data["caption"] = caption
    
    # Add new response
    responses[keyword] = response_data
    
    # Save updated responses
    try:
        with open(RESPONSES_FILE, 'w', encoding='utf-8') as f:
            json.dump(responses, f, indent=4)
        
        # Update bot's responses
        bot.load_responses()
        
        flash(f"Response for '{keyword}' added successfully", "success")
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error adding response: {e}")
        flash("Failed to add response", "error")
        return redirect(url_for('add_response_form'))

@app.route('/delete/<keyword>', methods=['POST'])
def delete_response_route(keyword):
    """Delete a response"""
    logger.info(f"We are in app.route /delete/{keyword} POST")

    # Load current responses
    try:
        with open(RESPONSES_FILE, 'r', encoding='utf-8') as f:
            responses = json.load(f)
    except FileNotFoundError:
        flash("Responses file not found", "error")
        return redirect(url_for('index'))

    if keyword not in responses:
        flash(f"Keyword '{keyword}' not found", "error")
        return redirect(url_for('index'))

    # Delete response
    del responses[keyword]

    # Save updated responses
    try:
        with open(RESPONSES_FILE, 'w', encoding='utf-8') as f:
            json.dump(responses, f, indent=4)

        # Update bot's responses
        bot.load_responses()

        flash(f"Response for '{keyword}' deleted successfully", "success")
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error deleting response: {e}")
        flash("Failed to delete response", "error")
        return redirect(url_for('index'))

@app.route('/nlp-config')
def nlp_config_page():
    """Render the NLP configuration page"""
    logger.info("We are in app.route /nlp-config")
    return render_template('nlp_config.html')

@app.route('/learn-responses')
def learn_responses_page():
    """Render the learn responses page"""
    logger.info("We are in app.route /learn-responses")
    return render_template('learn_responses.html')

@app.route('/review-responses')
def review_responses():
    """Render the review responses page"""
    logger.info("We are in app.route /review-responses")
    return render_template('review_responses.html')

@app.route('/api/media', methods=['GET'])
def get_media():
    """API endpoint to get all available media files"""
    logger.info("We are in app.route /api/media GET")

    try:
        images = [f for f in os.listdir(IMAGES_DIR) if os.path.isfile(os.path.join(IMAGES_DIR, f))]
        audio = [f for f in os.listdir(AUDIO_DIR) if os.path.isfile(os.path.join(AUDIO_DIR, f))]
        
        return jsonify({
            "images": images,
            "audio": audio
        })
    except FileNotFoundError:
        return jsonify({
            "images": [],
            "audio": []
        })

@app.route('/api/media/<media_type>/<filename>', methods=['DELETE'])
def delete_media(media_type, filename):
    """API endpoint to delete a media file"""
    logger.info(f"We are in app.route /api/media/{media_type}/{filename} DELETE")

    if media_type not in ['images', 'audio']:
        return jsonify({"error": "Invalid media type"}), 400
    
    directory = IMAGES_DIR if media_type == 'images' else AUDIO_DIR
    file_path = os.path.join(directory, filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": f"File not found: {filename}"}), 404
    
    try:
        os.remove(file_path)
        return jsonify({"success": True, "message": f"{filename} deleted successfully"})
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        return jsonify({"error": f"Failed to delete file: {str(e)}"}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """API endpoint to get bot status"""
    logger.info("We are in app.route /api/status GET")

    global bot_running
    return jsonify({
        "status": "running" if bot_running else "stopped"
    })

@app.route('/api/bot/start', methods=['POST'])
def start_bot_api():
    """API endpoint to start the bot"""
    global bot_thread, bot_running
    logger.info("We are in app.route /api/bot/start POST")

    # Check if bot_thread exists before checking if it's alive
    if bot_thread and bot_thread.is_alive():
        return jsonify({"message": "Bot is already running"}), 400

    if bot_running:
        return jsonify({"message": "Bot is already running"}), 400

    try:
        bot_thread = threading.Thread(target=start_bot_thread)
        bot_thread.daemon = True
        bot_thread.start()
        bot_running = True
        return jsonify({"success": True, "message": "Bot started successfully"})
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        return jsonify({"error": f"Failed to start bot: {str(e)}"}), 500

@app.route('/api/nlp/config', methods=['GET'])
def get_nlp_config():
    """API endpoint to get NLP configuration"""
    logger.info("We are in app.route /api/nlp/config GET")
    try:
        with open(NLP_CONFIG_FILE, 'r', encoding='utf-8') as f:
            nlp_config = json.load(f)
        return jsonify(nlp_config)
    except FileNotFoundError:
        return jsonify({"error": "NLP config file not found"}), 404

@app.route('/api/nlp/config', methods=['POST'])
def update_nlp_config():
    """API endpoint to update NLP configuration"""
    logger.info("We are in app.route /api/nlp/config POST")
    data = request.json

    if not data:
        return jsonify({"error": "Invalid data"}), 400

    try:
        with open(NLP_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

        # Reload NLP processor
        bot.nlp.initialize_nlp_tools()

        return jsonify({"success": True, "message": "NLP config updated successfully"})
    except Exception as e:
        logger.error(f"Error updating NLP config: {e}")
        return jsonify({"error": "Failed to update NLP config"}), 500

@app.route('/api/nlp/typos', methods=['POST'])
def update_typos():
    """API endpoint to update common typos"""
    logger.info("We are in app.route /api/nlp/typos POST")
    data = request.json

    if not data or 'typos' not in data:
        return jsonify({"error": "Invalid data"}), 400

    try:
        with open(NLP_CONFIG_FILE, 'r', encoding='utf-8') as f:
            nlp_config = json.load(f)
    except FileNotFoundError:
        nlp_config = {"common_typos": {}, "stop_words": [], "phonetic_mappings": {}}

    nlp_config['common_typos'] = data['typos']

    try:
        with open(NLP_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(nlp_config, f, indent=4)

        # Reload NLP processor
        bot.nlp.initialize_nlp_tools()

        return jsonify({"success": True, "message": "Typos updated successfully"})
    except Exception as e:
        logger.error(f"Error updating typos: {e}")
        return jsonify({"error": "Failed to update typos"}), 500

@app.route('/api/nlp/stopwords', methods=['POST'])
def update_stopwords():
    """API endpoint to update stopwords"""
    logger.info("We are in app.route /api/nlp/stopwords POST")
    data = request.json

    if not data or 'stop_words' not in data:
        return jsonify({"error": "Invalid data"}), 400

    try:
        with open(NLP_CONFIG_FILE, 'r', encoding='utf-8') as f:
            nlp_config = json.load(f)
    except FileNotFoundError:
        nlp_config = {"common_typos": {}, "stop_words": [], "phonetic_mappings": {}}

    nlp_config['stop_words'] = data['stop_words']

    try:
        with open(NLP_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(nlp_config, f, indent=4)

        # Reload NLP processor
        bot.nlp.initialize_nlp_tools()

        return jsonify({"success": True, "message": "Stopwords updated successfully"})
    except Exception as e:
        logger.error(f"Error updating stopwords: {e}")
        return jsonify({"error": "Failed to update stopwords"}), 500

@app.route('/api/nlp/phonetics', methods=['POST'])
def update_phonetics():
    """API endpoint to update phonetic mappings"""
    logger.info("We are in app.route /api/nlp/phonetics POST")
    data = request.json

    if not data or 'phonetic_mappings' not in data:
        return jsonify({"error": "Invalid data"}), 400

    try:
        with open(NLP_CONFIG_FILE, 'r', encoding='utf-8') as f:
            nlp_config = json.load(f)
    except FileNotFoundError:
        nlp_config = {"common_typos": {}, "stop_words": [], "phonetic_mappings": {}}

    nlp_config['phonetic_mappings'] = data['phonetic_mappings']

    try:
        with open(NLP_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(nlp_config, f, indent=4)

        # Reload NLP processor
        bot.nlp.initialize_nlp_tools()

        return jsonify({"success": True, "message": "Phonetic mappings updated successfully"})
    except Exception as e:
        logger.error(f"Error updating phonetic mappings: {e}")
        return jsonify({"error": "Failed to update phonetic mappings"}), 500

@app.route('/api/learning/stats', methods=['GET'])
def get_learning_stats():
    """API endpoint to get learning statistics"""
    logger.info("We are in app.route /api/learning/stats GET")

    try:
        with open('conversation.json', 'r', encoding='utf-8') as f:
            conversations = json.load(f)
    except FileNotFoundError:
        return jsonify({"total": 0, "answered": 0, "unanswered": 0, "learned": 0})

    total = len(conversations)
    answered = 0
    unanswered = 0

    for user_input, response_data in conversations.items():
        if isinstance(response_data, dict) and response_data.get('content'):
            content = response_data['content']
            # Handle both string and list content types
            if isinstance(content, list):
                has_content = bool(content and any(item.strip() for item in content if isinstance(item, str)))
            else:
                has_content = bool(content and content.strip())

            if has_content:
                answered += 1
            else:
                unanswered += 1
        else:
            unanswered += 1

    # Check if responses_temp.json exists and count learned responses
    learned = 0
    try:
        with open('responses_temp.json', 'r', encoding='utf-8') as f:
            temp_responses = json.load(f)
            learned = len(temp_responses)
    except FileNotFoundError:
        learned = 0

    return jsonify({
        "total": total,
        "answered": answered,
        "unanswered": unanswered,
        "learned": learned
    })

@app.route('/api/conversations/preview', methods=['GET'])
def get_conversations_preview():
    """API endpoint to get conversations preview"""
    logger.info("We are in app.route /api/conversations/preview GET")

    try:
        with open('conversation.json', 'r', encoding='utf-8') as f:
            conversations = json.load(f)
    except FileNotFoundError:
        return jsonify({"conversations": []})

    preview = []
    for user_input, response_data in list(conversations.items())[:10]:  # Limit to 10
        if isinstance(response_data, dict):
            content = response_data.get('content', '')
            # Handle both string and list content types
            if isinstance(content, list):
                has_response = bool(content and any(item.strip() for item in content if isinstance(item, str)))
            else:
                has_response = bool(content and content.strip())
            preview.append({
                "input": user_input,
                "response": response_data if has_response else {"content": ""},
                "has_response": has_response
            })

    return jsonify({"conversations": preview})

@app.route('/api/learning/start', methods=['POST'])
def start_learning():
    """API endpoint to start the learning process"""
    logger.info("We are in app.route /api/learning/start POST")

    try:
        learner = ConversationLearner('conversation.json', 'responses_temp.json')
        learned_responses = learner.learn_from_conversations()

        return jsonify({
            "success": True,
            "learned_count": len(learned_responses),
            "total_conversations": len(learner.conversations),
            "unanswered_count": sum(1 for resp in learned_responses.values()
                                  if resp.get('source') == 'unanswered_suggestion')
        })
    except Exception as e:
        logger.error(f"Error during learning: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/responses/learned', methods=['GET'])
def get_learned_responses():
    """API endpoint to get learned responses"""
    logger.info("We are in app.route /api/responses/learned GET")

    try:
        with open('responses_temp.json', 'r', encoding='utf-8') as f:
            learned_responses = json.load(f)
        return jsonify({"responses": learned_responses})
    except FileNotFoundError:
        return jsonify({"responses": {}})
    except Exception as e:
        logger.error(f"Error loading learned responses: {e}")
        return jsonify({"error": "Failed to load learned responses"}), 500

@app.route('/api/responses/add-selected', methods=['POST'])
def add_selected_responses():
    """API endpoint to add selected responses to final responses"""
    logger.info("We are in app.route /api/responses/add-selected POST")
    data = request.json

    if not data or 'keywords' not in data:
        return jsonify({"error": "Invalid data"}), 400

    keywords = data['keywords']
    if not isinstance(keywords, list):
        return jsonify({"error": "Keywords must be a list"}), 400

    try:
        # Load learned responses
        with open('responses_temp.json', 'r', encoding='utf-8') as f:
            learned_responses = json.load(f)
    except FileNotFoundError:
        return jsonify({"error": "No learned responses found"}), 404

    # Load current final responses
    try:
        with open(RESPONSES_FILE, 'r', encoding='utf-8') as f:
            final_responses = json.load(f)
    except FileNotFoundError:
        final_responses = {}

    added_count = 0
    for keyword in keywords:
        if keyword in learned_responses:
            # Clean the response data for final storage
            response_data = learned_responses[keyword].copy()
            # Remove learning-specific fields
            response_data.pop('confidence', None)
            response_data.pop('source', None)
            response_data.pop('similar_inputs', None)
            response_data.pop('needs_review', None)

            final_responses[keyword] = response_data
            added_count += 1

    if added_count > 0:
        # Save updated final responses
        try:
            with open(RESPONSES_FILE, 'w', encoding='utf-8') as f:
                json.dump(final_responses, f, indent=4)

            # Update bot's responses
            bot.load_responses()

            return jsonify({
                "success": True,
                "added_count": added_count,
                "message": f"Successfully added {added_count} responses to bot"
            })
        except Exception as e:
            logger.error(f"Error saving final responses: {e}")
            return jsonify({"error": "Failed to save responses"}), 500
    else:
        return jsonify({"success": True, "added_count": 0, "message": "No responses were added"})

if __name__ == '__main__':
    # Start the bot in a separate thread
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not app.config['DEBUG']:
        try:
            bot_thread = threading.Thread(target=start_bot_thread)
            bot_thread.daemon = True
            bot_thread.start()
            bot_running = True
            logger.info("Bot thread started")
        except Exception as e:
            logger.error(f"Failed to start bot thread: {e}")
        
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)
