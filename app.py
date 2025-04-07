from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging
from helplines import get_state_helplines

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__)
CORS(app)

# Define system prompt
system_prompt = '''You are a highly knowledgeable, empathetic, and trustworthy **AI Legal Assistant**. 

CRITICAL INSTRUCTION - LANGUAGE:
Always respond in {language} only, following these strict rules:

For Hindi:
- Use pure Devanagari script only
- Do not add English translations in brackets
- Use formal tone (आप, कृपया)
- Keep legal terms in Hindi
Example: "धारा ३५४ के अनुसार..."

For Hinglish:
- Mix Hindi and English naturally
- Do not add translations in brackets
- Keep legal terms in English
Example: "IPC Section 354 ke hisaab se..."

For English:
- Use formal English only
Example: "According to Section 354..."

## Response Structure:
<div class="legal-response">
    <div class="response-header">
        <h2 class="header-title">[Pure {language} title without translations]</h2>
    </div>
    <div class="empathy-section">
        <p class="empathy-message">[Pure {language} message without translations]</p>
    </div>
    <div class="inquiry-section">
        <h3 class="section-title">[Pure {language} title]</h3>
        <div class="inquiry-content">
            <p class="inquiry-intro">[Pure {language} introduction]</p>
            <ul class="inquiry-list">
                [Pure {language} list items without translations]
            </ul>
        </div>
    </div>
</div>
'''

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Debug incoming request
        logger.debug(f"Request Headers: {request.headers}")
        logger.debug(f"Request Data: {request.data}")
        
        data = request.get_json()
        logger.debug(f"Processed Data: {data}")

        if not data:
            logger.error("No data received in request")
            return jsonify({"error": "No data provided"}), 400

        # Extract all fields from request
        current_message = data.get('message')
        chat_history = data.get('history', [])
        location = data.get('location', 'Unknown')
        preferred_language = data.get('preferred_language', 'English')

        logger.info(f"""
            Message: {current_message}
            Location: {location}
            Language: {preferred_language}
        """)

        if not current_message:
            logger.error("No message field in data")
            return jsonify({"error": "No message provided"}), 400

        # Format the system prompt with the preferred language
        formatted_prompt = system_prompt.format(language=preferred_language)

        # Update context to enforce no translations
        context = f"""
        STRICT LANGUAGE RULES:
        - Respond ONLY in: {preferred_language}
        - DO NOT add translations in brackets
        - Keep response pure in the specified language
        - Use natural language style for {preferred_language}
        
        User Location: {location}
        """

        # Create complete prompt with context
        complete_prompt = formatted_prompt + "\n\nContext:\n" + context
        if chat_history:
            formatted_history = "\n".join([
                f"{'Assistant' if msg.get('type') == 'bot' else 'Human'}: {msg.get('content')}"
                for msg in chat_history
            ])
            complete_prompt += f"\n\nPrevious conversation:\n{formatted_history}"
        complete_prompt += f"\n\nHuman: {current_message}"

        # Get response using Gemini model
        model = genai.GenerativeModel("gemini-2.0-flash")
        chat = model.start_chat(history=[])
        response = chat.send_message(
            complete_prompt,
            generation_config={
                "temperature": 0.6,
                "top_k": 40,
                "top_p": 0.8,
            }
        )

        # Clean and format the response
        formatted_response = response.text.replace('```html', '').replace('```', '')

        # Get location-specific helplines
        state = location.split(',')[-1].strip().lower() if location else None
        helplines = get_state_helplines(state) if state else {}

        # Add location-specific helplines to response
        if helplines:
            helpline_html = f'''
            <div class="local-helplines">
                <h3 class="section-title">Local Helplines for {location}</h3>
                <div class="contact-grid">
                    {''.join([f"""
                        <div class="contact-item">
                            <div class="contact-label">{k.replace('_', ' ').title()}:</div>
                            <div class="contact-value">{v}</div>
                        </div>
                    """ for k, v in helplines.items()])}
                </div>
            </div>
            '''
            formatted_response = formatted_response.replace('</div></div>', f'{helpline_html}</div></div>')

        return jsonify({
            'response': formatted_response,
            'status': 'success',
            'format': 'html',
            'location': location,
            'language': preferred_language
        })

    except Exception as e:
        logger.exception("Error in chat endpoint")
        return jsonify({
            "error": "An error occurred while processing your request",
            "details": str(e),
            "format": "html",
            "response": '''
                <div class="error-container">
                    <div class="error-message">
                        <h3>Error Processing Request</h3>
                        <p>I apologize, but I encountered an error. Please try again.</p>
                    </div>
                </div>
            '''
        }), 500

if __name__ == '__main__':
    app.debug = True
    app.run(host='127.0.0.1', port=5000)
