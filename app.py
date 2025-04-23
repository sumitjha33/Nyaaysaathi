from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from typing import Any, List, Mapping, Optional, Type
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient
import logging
import traceback
import json
import re
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Pinecone (using the same index you've already set up)
pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docsearch = Pinecone.from_existing_index(
    index_name="nyaaysaathi",
    embedding=embeddings,
    namespace=""
)

# Create a custom Gemini model wrapper for Langchain
class GeminiChatModel(BaseChatModel):
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.5
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> dict:
        # Convert LangChain messages to text for Gemini
        prompt = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                prompt += f"Human: {message.content}\n"
            elif isinstance(message, AIMessage):
                prompt += f"AI: {message.content}\n"
            else:
                prompt += f"{message.type}: {message.content}\n"
        
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt, **kwargs)
        
        return {"generation": AIMessage(content=response.text), "raw": response.text}
    
    @property
    def _llm_type(self) -> str:
        return "gemini-chat"
        
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get identifying parameters."""
        return {"model_name": self.model_name}

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

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

# Location-specific emergency helpline numbers
helpline_data = {
    "Delhi": {
        "Police": "112",
        "Women's Helpline": "1091",
        "Legal Aid": "1516",
        "District Legal Services Authority": "011-23071764",
        "Delhi Commission for Women": "181"
    },
    "Mumbai": {
        "Police": "112",
        "Women's Helpline": "1091",
        "Legal Aid": "1516",
        "District Legal Services Authority": "022-26591131",
        "Maharashtra State Commission for Women": "022-26592707"
    },
    "Bangalore": {
        "Police": "112",
        "Women's Helpline": "1091",
        "Legal Aid": "1516",
        "Karnataka State Legal Services Authority": "080-22111714",
        "Vanitha Sahayavani": "080-22943225"
    },
    "Chennai": {
        "Police": "112",
        "Women's Helpline": "1091",
        "Legal Aid": "1516",
        "Tamil Nadu State Legal Services Authority": "044-25340052",
        "Chennai Women's Helpline": "044-28592750"
    },
    # Default fallback for any other location
    "Default": {
        "Police": "112",
        "Women's Helpline": "1091",
        "Legal Aid": "1516",
        "National Commission for Women": "011-26942369",
        "National Legal Services Authority": "011-24646542"
    }
}

# System prompt template for the legal assistant
system_prompt = '''You are a highly knowledgeable, empathetic, and trustworthy **AI Legal Assistant**, designed to assist users with their legal concerns in a **human-like, conversational, and solution-oriented** manner. Your goal is to provide **accurate legal guidance** while ensuring users feel **understood, reassured, and empowered** to take the right actions.  

## **Guidelines for Responses:**  
1️⃣ **Acknowledge & Show Empathy** – Recognize the user's concern and reassure them that legal remedies exist.  
2️⃣ **Ask Clarifying Questions** – Gather context by counter-questioning until you fully understand the situation.  
3️⃣ **Provide IPC Sections & Punishments** – Clearly outline the relevant Indian laws (IPC, CrPC, Domestic Violence Act, etc.) with associated punishments.  
4️⃣ **Suggest Next Steps** – Guide the user on what they should do, such as filing an FIR, collecting evidence, or contacting a legal aid service.  
5️⃣ **Ensure User Safety** – If the situation is urgent (e.g., domestic violence, harassment), prioritize immediate safety and emergency services.  
6️⃣ **Multilingual Understanding** – Understand and respond in simple, user-friendly language while keeping legal accuracy.  
7️⃣ **Encourage Legal Consultation** – Always recommend seeking legal help from a qualified lawyer for complex matters.  

## **Response Format:**  
### **1️⃣ Acknowledge & Show Empathy**  
"I'm sorry you're going through this. The law is here to protect you, and I'll help you understand what steps you can take."  

### **2️⃣ Counter-Question for Context**  
- "Can you tell me when and how often this happens?"  
- "Do you have any evidence, such as messages, witnesses, or medical reports?"  
- "Have you reported this to the police before?"  
- "Are you currently in a safe environment?"  

### **3️⃣ Provide IPC Sections & Punishments**  
- "Based on your situation, this falls under **Section 498A of the IPC**, which deals with cruelty by a husband or his relatives. The punishment can be **up to 3 years of imprisonment and a fine**."  
- "If there was physical harm, **Section 323 (causing hurt)** or **Section 325 (grievous hurt)** may also apply."  
- "For extreme cases, **Section 307 (attempt to murder)** could be considered, with punishment up to **life imprisonment**."  

### **4️⃣ Recommend Next Legal Steps**  
- "You should file an **FIR at the nearest police station** under these sections."  
- "It would be helpful to collect evidence, such as medical reports, photos of injuries, or messages."  
- "If you need immediate protection, you can apply for a **protection order under the Domestic Violence Act, 2005**."  
- "Legal aid services and NGOs can also assist you free of cost. Would you like me to suggest some?"

### **5️⃣ Ensure User Safety & Provide Counseling-Like Support**  
- "Are you currently safe? If you're in danger, please consider calling 112 (emergency police) or seeking help from a trusted person."  
- "I understand this is difficult, but remember, you have rights, and the law is in your favor."  
- "Would you like guidance on how to explain this to the police or a lawyer?"  

### **6️⃣ Encourage Legal Consultation**  
- "For proper legal action, consulting a lawyer will be the best step. I can suggest resources where you can get **free legal aid**."  

## **Example Conversation:**  
💬 **User:** My husband beats me daily. What should I do?  
🤖 **Legal Assistant:**  
"I'm really sorry you're facing this. You have legal rights, and I'll guide you step by step to get the right help. Let's start by understanding your situation better. Can you tell me if you have suffered any physical injuries or threats?"  

💬 **User:** Yes, he hits me and threatens to kill me if I tell anyone.  
🤖 **Legal Assistant:**  
"This is very serious. Your case falls under **IPC Section 498A (cruelty), Section 323 (causing hurt), and possibly Section 506 (criminal intimidation).** If his threats are severe, **Section 307 (attempt to murder)** might apply, which has punishment up to **life imprisonment**."  

💬 **User:** What should I do now?  
🤖 **Legal Assistant:**  
"You should immediately:  
1️⃣ **File an FIR at your nearest police station** under these sections.  
2️⃣ **Seek a protection order** under the **Domestic Violence Act, 2005**, to prevent further harm.  
3️⃣ Collect **any evidence** (medical reports, messages, witnesses).  
4️⃣ If needed, contact a **women's helpline (181) or free legal aid services.**  Are you currently safe? If you're in immediate danger, please **call 112 (police emergency) right away.**"  

### **Special Features:**  
✔️ **Conversational & Human-Like** – The AI ensures the user feels heard and supported.  
✔️ **Context-Based Legal Assistance** – Asks counter-questions before giving legal advice.  
✔️ **Accurate IPC Sections & Laws** – Provides exact punishments and next steps.  
✔️ **Multilingual Support** – Can assist users in multiple languages.  
✔️ **Ensures Safety First** – Prioritizes user well-being in serious cases.  

User is located in: {location}
Preferred language: {language}
Emergency helplines: {emergency_numbers}

IMPORTANT: 
1. First, acknowledge the user's issue with empathy.
2. If this is the user's first message or if we need more context, ask 3-5 relevant counter-questions to gather more information.
3. If we've already gathered sufficient information through 8 or more exchanges, provide a comprehensive analysis including:
   - Relevant IPC sections and punishments
   - Clear step-by-step guidance on what actions to take
   - Emergency contacts specific to their location
4. ALWAYS respond in the user's {language} language.
5. Don't explicitly follow the format above - make your response feel natural and conversational.
6. Focus on Indian laws and legal procedures.

Conversation history:
{history}

Current user message: {input}

Remember to maintain a supportive, professional tone while being conversational and empathetic.
'''

# Function to get emergency numbers for a specific location
def get_emergency_numbers(location):
    """Get emergency numbers for a specific location"""
    # Clean up the location string
    clean_location = location.split(',')[0].strip() if ',' in location else location.strip()
    
    # Try to match with our predefined locations
    for city in helpline_data:
        if city.lower() in clean_location.lower():
            return helpline_data[city]
    
    # If no match, return default helplines
    return helpline_data["Default"]

# Function to format emergency numbers as text
def format_emergency_numbers(numbers_dict):
    """Format emergency numbers as a string"""
    formatted = []
    for service, number in numbers_dict.items():
        formatted.append(f"{service}: {number}")
    return ", ".join(formatted)

# Function to extract relevant IPC sections based on the conversation
def get_relevant_ipc_sections(query, conversation_memory):
    """Get relevant IPC sections from Pinecone based on conversation context"""
    try:
        # Combine conversation memory to create a rich context
        full_context = query + "\n" + conversation_memory
        
        # Search for relevant legal sections in Pinecone
        results = docsearch.similarity_search(
            full_context,
            k=5  # Get top 5 most relevant sections
        )
        
        # Format the results
        formatted_sections = []
        for doc in results:
            content = doc.page_content.strip()
            
            # Only include relevant IPC sections
            if "section" in content.lower() or "ipc" in content.lower():
                # Extract section number and description
                section_match = re.search(r'Section\s+(\d+[A-Za-z]*)', content)
                section_number = section_match.group(1) if section_match else "Unknown"
                
                # Extract punishment information if available
                punishment = ""
                if "punish" in content.lower():
                    punishment_match = re.search(r'(punish[^\.\n]+)', content, re.IGNORECASE)
                    punishment = punishment_match.group(1) if punishment_match else ""
                
                # Create formatted section entry
                formatted_sections.append({
                    "section": f"Section {section_number}",
                    "description": content.split('.')[0] if '.' in content else content,
                    "punishment": punishment
                })
        
        return formatted_sections
    except Exception as e:
        logger.error(f"Error fetching IPC sections: {str(e)}")
        logger.error(traceback.format_exc())
        return []

# Direct call to Gemini API for generating responses
def generate_response_with_gemini(message, conversation_memory, ipc_sections, location, language):
    """Generate response using Gemini API directly"""
    try:
        # Format IPC sections if available
        ipc_context = ""
        if ipc_sections:
            ipc_context = "Relevant IPC sections for this case:\n"
            for section in ipc_sections:
                ipc_context += f"- {section['section']}: {section['description']}"
                if section['punishment']:
                    ipc_context += f" Punishment: {section['punishment']}"
                ipc_context += "\n"
        
        # Get emergency numbers
        emergency_numbers = get_emergency_numbers(location)
        formatted_emergency = format_emergency_numbers(emergency_numbers)
        
        # Replace template variables in system prompt
        prompt = system_prompt.format(
            history=conversation_memory,
            input=message,
            location=location,
            language=language,
            emergency_numbers=formatted_emergency
        )
        
        if ipc_context:
            prompt += f"\n\nRelevant IPC sections:\n{ipc_context}"
        
        # Call Gemini API
        model = genai.GenerativeModel("gemini-2.0-flash", generation_config={"temperature": 0.3})
        response = model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        logger.error(f"Error generating response with Gemini: {str(e)}")
        logger.error(traceback.format_exc())
        return "I apologize, but I'm experiencing technical difficulties. If you need immediate legal assistance, please contact emergency services."

# Translation function for different languages
def translate_to_language(text, language):
    """Use Google Gemini to translate text to the specified language"""
    if language.lower() == "english":
        return text
    
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"Translate the following text to {language}. Keep any formatting and technical terms intact:\n\n{text}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return text  # Return original text if translation fails

# Initialize LangChain memory
memory = ConversationBufferWindowMemory(k=8)  # Store only the last 8 conversation exchanges

# Chat endpoint
@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json(force=True)
        
        # Validate required fields
        if not data or 'message' not in data:
            return jsonify({"error": "Missing message"}), 400
        
        message = data['message']
        location = data.get('location', 'Delhi, India')
        preferred_language = data.get('preferred_language', 'English')
        
        # Add user message to memory
        memory.save_context({"input": message}, {"output": ""})
        
        # Get conversation history from memory
        memory_variables = memory.load_memory_variables({})
        conversation_buffer = memory_variables.get("history", "")
        
        # Get conversation turns count (messages in memory / 2)
        conversation_turns = len(memory.buffer) // 2
        
        # Get relevant IPC sections if we have enough context
        ipc_sections = []
        if conversation_turns >= 4:  # After 4 turns, we should have enough context
            ipc_sections = get_relevant_ipc_sections(message, conversation_buffer)
        
        # Generate response
        response_text = generate_response_with_gemini(
            message,
            conversation_buffer,
            ipc_sections,
            location,
            preferred_language
        )
        
        # Save assistant response to memory
        memory.save_context({"input": ""}, {"output": response_text})
        
        # If language is not English, ensure response is in the correct language
        if preferred_language.lower() != "english":
            response_text = translate_to_language(response_text, preferred_language)
        
        return jsonify({
            'response': response_text,
            'status': 'success',
            'format': 'text'
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Fallback response
        fallback_response = "I apologize, but I'm experiencing technical difficulties right now. If you need immediate legal assistance, please contact your local emergency services or legal aid hotline."
        
        # Translate fallback if needed
        if 'preferred_language' in locals() and preferred_language.lower() != "english":
            try:
                fallback_response = translate_to_language(fallback_response, preferred_language)
            except:
                pass
                
        return jsonify({
            "error": str(e),
            "status": "error",
            "response": fallback_response
        }), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)