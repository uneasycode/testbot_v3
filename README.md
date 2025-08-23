# testbot
A Test Chat bot which uses Hindlish input and outputs to a chat app

# **Telegram Human-like Chatbot with NLP Capabilities**
This Python application provides a Telegram chatbot that responds to messages based on a customizable JSON dictionary of responses, while using Natural Language Processing (NLP) to handle spelling mistakes, grammar issues, and understand context.
## **Features**
### **Core Functionality**
- **Natural Language Processing**: Uses multiple NLP techniques to understand user messages despite spelling/grammar errors
- **Human-like Responses**: Simulates typing delays and contextual responses to appear more human-like
- **Media Support**: Can send text, images, and audio responses based on keyword triggers
- **Easy Management**: Web-based Flask interface to manage responses and media files without touching code
- **Real-time Updates**: Changes to responses are applied immediately without restarting the bot
### **NLP Capabilities**
- **Spelling Correction**: Automatically corrects common spelling mistakes
- **Fuzzy Matching**: Uses string similarity to match keywords even with typos
- **Context Understanding**: Analyzes message context for better responses
- **Sentiment Analysis**: Detects positive/negative sentiment for contextual responses
- **TF-IDF Vectorization**: Advanced text matching using TF-IDF and cosine similarity
## **Project Structure**
telegram\_chatbot/

├── app.py              # Main Flask application

├── bot.py              # Telegram bot using Telethon and NLP

├── config.py           # Configuration settings

├── static/             # Static files for Flask web interface

│   ├── css/

│   │   └── style.css

│   └── js/

│       └── script.js

├── templates/          # HTML templates for web interface

│   ├── index.html

│   └── edit\_response.html

├── responses.json      # JSON dictionary for bot responses

├── media/              # Folder for images and audio files

│   ├── images/

│   └── audio/

└── requirements.txt    # Project dependencies
## **Setup Instructions**

**Install dependencies**:

pip install -r requirements.txt


**Create a** .env **file** with your Telegram API credentials:

API\_ID=your\_api\_id

API\_HASH=your\_api\_hash

PHONE=your\_phone\_number

SECRET\_KEY=your\_flask\_secret\_key

DEBUG=True


**Run the application**:

python app.py


**Access the web interface** at http://localhost:5000

## **NLP Implementation Details**
The NLP processing is handled by the NLPProcessor class in bot.py, which:

**Preprocesses text**:

   - Converts to lowercase
   - Removes punctuation
   - Fixes common typos using a predefined dictionary
   - Performs spelling correction using pyspellchecker (if available)
   - Tokenizes, removes stopwords, and lemmatizes using NLTK (if available)

**Finds the best matching keyword** using:

   - TF-IDF vectorization and cosine similarity (if scikit-learn is available)
   - Direct substring matching
   - String similarity using SequenceMatcher

**Extracts context** from messages:

   - Detects questions vs. statements
   - Identifies time indicators
   - Performs basic sentiment analysis
   - Tracks conversation history per user

**Generates contextual responses** when no direct keyword match is found

## **Response Configuration**
Responses are stored in responses.json in the following format:

{

`    `"keyword": {

`        `"type": "text|image|audio",

`        `"content": ["Response 1", "Response 2"] or "filename.jpg",

`        `"typing\_delay": [min\_seconds, max\_seconds],

`        `"caption": "Optional caption for images"

`    `}

}
## **Web Interface**
The Flask web interface allows you to:

- View all configured responses
- Add new responses
- Edit existing responses
- Delete responses
- Upload image and audio files
- Monitor bot status
## **Handling Ambiguity and Errors**
The bot handles various forms of user input errors:

- Misspelled keywords are matched using fuzzy string matching
- Common typos are automatically corrected
- When no match is found, contextual default responses are provided
- Different response options provide variety to make the bot seem more human
## **Dependencies**
- telethon: Telegram client library
- flask: Web framework for the management interface
- nltk: Natural Language Toolkit for text processing
- pyspellchecker: Spelling correction library
- scikit-learn: For TF-IDF vectorization and similarity calculations
- python-dotenv: Environment variable management
- numpy: Required for numerical operations

The application will gracefully degrade if optional NLP libraries are not available, falling back to simpler text matching methods.

