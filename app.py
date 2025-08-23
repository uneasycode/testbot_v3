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
from bot_updated import TelegramBot
from config import (
    SECRET_KEY, DEBUG, RESPONSES_FILE,
    IMAGES_DIR, AUDIO_DIR
)

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