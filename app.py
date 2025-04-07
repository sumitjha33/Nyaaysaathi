from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Pinecone
pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docsearch = Pinecone.from_existing_index(
    index_name="nyaaysaathi",
    embedding=embeddings,
    namespace=""
)

app = Flask(__name__)
# Update CORS configuration
CORS(app, 
     resources={r"/*": {
         "origins": ["*"],  # Allow all origins for now
         "methods": ["GET", "POST", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization"],
         "expose_headers": ["Content-Type"]
     }})

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    if 'Origin' in request.headers:
        origin = request.headers['Origin']
        response.headers['Access-Control-Allow-Origin'] = origin
    else:
        response.headers['Access-Control-Allow-Origin'] = '*'
        
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, X-Requested-With, Content-Type, Accept, Authorization'
    return response

# Update system prompt
system_prompt = '''You are a highly knowledgeable, empathetic, and trustworthy **AI Legal Assistant**. 

CONVERSATION FLOW:
1. First Response:
   - Show empathy
   - Ask essential questions
   - DO NOT provide legal sections yet

2. Counter Questions Phase:
   - Ask follow-up questions based on user's responses
   - Gather specific details about the incident
   - Continue until you have clear understanding

3. Legal Analysis Phase (ONLY after gathering sufficient information):
   - Use the provided IPC sections from context
   - Explain applicable sections and punishments
   - Provide clear next steps

RESPONSE STRUCTURE:
<div class="legal-response">
    <!-- For Initial/Counter Questions Phase -->
    <div class="response-header">
        <h2 class="header-title">[Title in {language}]</h2>
    </div>
    <div class="empathy-section">
        <p class="empathy-message">[Empathy in {language}]</p>
    </div>
    <div class="questions-section">
        <h3>[Important Questions]</h3>
        <ul class="question-list">
            [Specific questions needed]
        </ul>
    </div>

    <!-- For Legal Analysis Phase (only after gathering all info) -->
    <div class="legal-analysis">
        <div class="applicable-sections">
            <h3>Applicable IPC Sections</h3>
            [Insert sections from Pinecone here]
        </div>
        <div class="next-steps">
            <h3>Recommended Actions</h3>
            [Action steps]
        </div>
    </div>
</div>
'''

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        current_message = data.get('message')
        chat_history = data.get('history', [])
        location = data.get('location', 'Unknown')
        preferred_language = data.get('preferred_language', 'English')

        if not current_message:
            return jsonify({"error": "No message provided"}), 400

        formatted_prompt = system_prompt.format(language=preferred_language)
        
        # Get relevant IPC sections
        relevant_docs = docsearch.similarity_search(current_message, k=4)
        legal_context = "\n".join([doc.page_content for doc in relevant_docs])

        # Create context
        context = f"""
        STRICT LANGUAGE RULES:
        - Respond ONLY in: {preferred_language}
        - DO NOT add translations in brackets
        
        LEGAL CONTEXT:
        {legal_context}
        
        CONVERSATION STATE:
        - If initial message or unclear: Ask essential questions first
        - If have partial info: Ask specific follow-up questions
        - Only provide legal sections after gathering complete information
        
        User Location: {location}
        """

        # Create complete prompt
        complete_prompt = formatted_prompt + "\n\nContext:\n" + context
        if chat_history:
            formatted_history = "\n".join([
                f"{'Assistant' if msg.get('type') == 'bot' else 'Human'}: {msg.get('content')}"
                for msg in chat_history
            ])
            complete_prompt += f"\n\nPrevious conversation:\n{formatted_history}"
        complete_prompt += f"\n\nHuman: {current_message}"

        # Get response
        model = genai.GenerativeModel("gemini-2.0-flash")
        chat = model.start_chat(history=[])
        response = chat.send_message(
            complete_prompt,
            generation_config={"temperature": 0.6, "top_k": 40, "top_p": 0.8}
        )

        return jsonify({
            'response': response.text.replace('```html', '').replace('```', ''),
            'status': 'success',
            'format': 'html',
            'language': preferred_language
        })

    except Exception as e:
        return jsonify({
            "error": "An error occurred",
            "status": "error"
        }), 500

if __name__ == '__main__':
    app.debug = True
    app.run()
