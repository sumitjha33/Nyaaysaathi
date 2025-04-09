from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient
import logging
import traceback
from bs4 import BeautifulSoup
import json
import uuid
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

# In-memory session storage - in production, use Redis or a database
conversation_sessions = {}

app = Flask(__name__)
# Update CORS configuration to match frontend expectations
CORS(app, 
     resources={r"/*": {
         "origins": ["*"],  # Allow all origins for development (restrict in production)
         "methods": ["GET", "POST", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization"],
         "expose_headers": ["Content-Type"]
     }})

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    # Allow origin from request if available
    if 'Origin' in request.headers:
        origin = request.headers['Origin']
        response.headers['Access-Control-Allow-Origin'] = origin
    else:
        response.headers['Access-Control-Allow-Origin'] = '*'
        
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, X-Requested-With, Content-Type, Accept, Authorization'
    return response

def determine_conversation_stage(message, history, session_data=None):
    """Determine the conversation stage based on message and history"""
    
    # Check if this is a greeting
    greeting_phrases = ["hello", "hi", "hey", "greetings", "namaste", "good morning", 
                       "good afternoon", "good evening", "hola", "vanakkam"]
    
    if not history and any(greeting.lower() in message.lower() for greeting in greeting_phrases):
        return "GREETING"
    
    # Check session data for asked questions count
    questions_asked = 0
    if session_data and 'questions_asked' in session_data:
        questions_asked = session_data['questions_asked']
    
    # If we haven't asked enough questions yet, stay in information gathering
    if questions_asked < 8:
        # If history is empty or just started, return initial stage
        if not history or len(history) <= 2:
            return "INITIAL"
        else:
            return "INFORMATION_GATHERING"
    
    # Legal keywords that suggest we should provide legal analysis
    legal_keywords = ["what should i do", "help me", "legal", "police", "complaint", 
                      "action", "rights", "options", "next steps", "law", "section", "fir"]
    
    # Move to analysis if enough questions asked or legal keywords present
    if questions_asked >= 8 or any(word in message.lower() for word in legal_keywords):
        return "LEGAL_ANALYSIS"
    
    # Otherwise, continue gathering information
    return "INFORMATION_GATHERING"

def extract_case_details(message, history):
    """Extract key details from conversation history for better legal analysis"""
    
    # Combine all user messages to form context
    user_messages = [msg["content"] for msg in history if msg.get("type") == "user"]
    user_context = " ".join(user_messages) + " " + message
    
    # Use Gemini to extract key details
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
        Extract key legal details from this conversation:
        
        {user_context}
        
        Focus on:
        1. Nature of the legal issue (categorize it)
        2. Key parties involved (roles and relationships)
        3. Timeline of events (when incidents occurred)
        4. Location specifics (where incidents occurred)
        5. Any evidence mentioned
        6. Physical or emotional harm described
        7. Prior legal actions taken, if any
        
        Format as simple text summaries for each relevant category.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error extracting case details: {str(e)}")
        return user_context

def get_relevant_ipc_sections(query, history):
    """Get relevant IPC sections from Pinecone based on conversation context"""
    try:
        # Extract key details from conversation
        case_details = extract_case_details(query, history)
        
        # Search Pinecone with enhanced context
        results = docsearch.similarity_search(
            case_details,
            k=5  # Get top 5 most relevant sections
        )
        
        # Format the IPC sections with punishments
        formatted_sections = []
        for doc in results:
            content = doc.page_content.strip()
            if any(keyword in content.lower() for keyword in ["section", "ipc", "punishment"]):
                section_line = content.split("\n")[0] if "\n" in content else "IPC Section"
                
                # Extract punishment details if available
                punishment = ""
                if "punishment" in content.lower():
                    for line in content.split("\n"):
                        if "punishment" in line.lower():
                            punishment = line.strip()
                            break
                
                formatted_sections.append({
                    "section": section_line,
                    "details": content,
                    "punishment": punishment
                })
        
        return formatted_sections
    except Exception as e:
        logger.error(f"Pinecone search error: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def generate_greeting():
    """Generate a friendly greeting message"""
    return """Hello! I'm your legal assistant from NyaaySaathi. I'm here to help you understand your legal rights and options.

Feel free to share your situation, and I'll guide you through the appropriate legal steps. How can I assist you today?"""

def generate_question_bank(message, history):
    """Generate a comprehensive bank of questions for information gathering"""
    try:
        # Create context from history
        context = "\n".join([f"{msg['type']}: {msg['content']}" for msg in history])
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
        Based on this legal conversation:
        
        {context}
        
        Latest user message: {message}
        
        Generate 10 important follow-up questions that would help gather critical legal details.
        Questions should cover:
        
        1. INCIDENT DETAILS: When, where, how the incident occurred
        2. PARTIES INVOLVED: Who the perpetrators and victims are, relationships
        3. NATURE OF OFFENSE: Specific actions that constitute legal violations
        4. EVIDENCE: What documentation, witnesses, or proof exists
        5. TIMELINE: Pattern of behavior, frequency, duration
        6. SAFETY: Current safety situation and immediate concerns
        7. PRIOR ACTION: Previous complaints, police reports, or legal steps taken
        8. IMPACT: Physical, emotional, financial consequences
        9. WITNESSES: Who saw or knows about the incidents
        10. JURISDICTION: Location details relevant to legal jurisdiction
        
        Return each question on a new line. Make questions conversational and empathetic.
        """
        
        response = model.generate_content(prompt)
        questions = response.text.strip().split("\n")
        
        # Clean up questions
        clean_questions = []
        for q in questions:
            # Remove numbering and extra formatting
            q = q.strip()
            if q:
                # Remove any leading numbers, dashes or dots
                while q and (q[0].isdigit() or q[0] in "-.*"):
                    q = q[1:].strip()
                clean_questions.append(q)
        
        return clean_questions
    except Exception as e:
        logger.error(f"Error generating question bank: {str(e)}")
        # Fallback questions
        return [
            "Can you tell me when this incident first occurred?",
            "Where exactly did this happen? This helps determine jurisdiction.",
            "Who was involved in the incident? What is your relationship with them?",
            "Has there been any physical harm or threats to your safety?",
            "Do you have any evidence like messages, photos, or witnesses?",
            "Have you filed any police complaints or taken legal action previously?",
            "Are there any ongoing threats or safety concerns right now?",
            "How has this affected you physically, emotionally, or financially?",
            "Can you describe the most recent incident in detail?",
            "Has anyone else witnessed these incidents?"
        ]

def select_next_questions(question_bank, questions_asked, message, history, max_questions=3):
    """Select the next set of questions to ask based on context and priority"""
    try:
        if not question_bank or len(question_bank) == 0:
            # Fallback if no questions in bank
            return ["Could you please provide more details about your situation?"]
        
        # Create context from history
        context = "\n".join([f"{msg['type']}: {msg['content']}" for msg in history])
        
        # If we have many unasked questions, prioritize them
        unasked_questions = question_bank[questions_asked:]
        
        if not unasked_questions:
            return []  # All questions asked
            
        # Select questions to ask in this round
        if len(unasked_questions) <= max_questions:
            return unasked_questions
            
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
        Based on this legal conversation:
        
        {context}
        
        Latest user message: {message}
        
        From these potential follow-up questions:
        {json.dumps(unasked_questions)}
        
        Select the {max_questions} MOST IMPORTANT questions to ask next based on:
        1. What information is still missing
        2. What would be most relevant to determining legal options
        3. Logical flow of conversation
        4. Safety and urgency concerns
        
        Return ONLY the selected questions, one per line, with no additional text.
        """
        
        response = model.generate_content(prompt)
        selected = response.text.strip().split("\n")
        selected = [q.strip() for q in selected if q.strip()]
        
        # Ensure we have the right number of questions
        if len(selected) > max_questions:
            selected = selected[:max_questions]
        elif len(selected) < max_questions and len(unasked_questions) > len(selected):
            # Add questions from unasked until we reach max_questions
            for q in unasked_questions:
                if q not in selected and len(selected) < max_questions:
                    selected.append(q)
                    
        return selected
        
    except Exception as e:
        logger.error(f"Error selecting next questions: {str(e)}")
        # Return the next max_questions from the bank as fallback
        start = questions_asked
        end = min(start + max_questions, len(question_bank))
        return question_bank[start:end]

def format_questions_as_conversation(questions):
    """Format multiple questions into a conversational flow"""
    if not questions:
        return ""
        
    # Conversation bridges
    bridges = [
        "I'd like to understand more about", 
        "Could you also tell me", 
        "It would help to know", 
        "I'm wondering",
        "Can you share with me",
        "I'd appreciate knowing",
        "It's important to understand"
    ]
    
    formatted_questions = []
    for i, question in enumerate(questions):
        # Remove any leading question marks or spaces
        clean_q = question.lstrip("?- ").strip()
        
        # Don't add a bridge phrase if the question already starts with a question word
        if any(clean_q.lower().startswith(w) for w in ["when", "where", "what", "how", "who", "why", "can", "could", "did", "have", "has"]):
            formatted_questions.append(clean_q)
        else:
            formatted_questions.append(f"{bridges[i % len(bridges)]} {clean_q.lower()}")
    
    # Start with first question
    result = formatted_questions[0]
    
    # Add remaining questions with connectors
    for i in range(1, len(formatted_questions)):
        if i == len(formatted_questions) - 1 and len(formatted_questions) > 1:
            result += f". And finally, {formatted_questions[i].lower()}?"
        else:
            result += f". {formatted_questions[i]}?"
            
    return result

def generate_empathy_message(message, history):
    """Generate an empathetic response based on the user's situation"""
    try:
        # Create context from history
        context = "\n".join([f"{msg['type']}: {msg['content']}" for msg in history]) 
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
        Based on this legal conversation:
        
        {context}
        
        Latest user message: {message}
        
        Generate a brief, empathetic response (2-3 sentences) acknowledging the situation.
        The tone should be warm and compassionate but also professional.
        Do not provide legal advice in this response, just acknowledge their situation.
        If the user is clearly distressed, validate their emotions.
        If they mention harm or danger, express appropriate concern.
        
        Remember to be authentic and genuine in the response.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error generating empathy message: {str(e)}")
        # Fallback empathy message if API fails
        return "I understand this situation must be difficult for you. Thank you for sharing these details with me. You have legal rights, and I'm here to help you navigate through this challenging time."

def generate_legal_assessment(message, history, ipc_sections):
    """Generate a legal assessment based on the conversation and IPC sections"""
    try:
        # Create context from history
        context = "\n".join([f"{msg['type']}: {msg['content']}" for msg in history])
        
        # Format IPC sections for the model
        ipc_context = "\n".join([f"Section: {section['section']}\nDetails: {section['details']}" 
                               for section in ipc_sections]) if ipc_sections else "No specific IPC sections identified yet."
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
        Based on this legal conversation:
        
        {context}
        
        Latest user message: {message}
        
        And these relevant IPC sections:
        
        {ipc_context}
        
        Provide a CLEAR, DIRECT legal assessment (2-3 sentences).
        Specifically identify IPC sections that apply using this format: 
        "Based on what you've shared, your case falls under [IPC sections]"
        
        Then add 1-2 sentences explaining why these sections apply in plain language.
        Be extremely concise and direct in your assessment.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error generating legal assessment: {str(e)}")
        # Fallback assessment if API fails
        return "Based on what you've shared, your case falls under IPC Section 498A (cruelty), Section 323 (causing hurt), and Section 506 (criminal intimidation). These sections address the physical harm and threatening behavior you've experienced, which constitute criminal offenses under Indian law."

def generate_action_steps(message, history, location):
    """Generate action steps based on the legal situation"""
    try:
        # Create context from history
        context = "\n".join([f"{msg['type']}: {msg['content']}" for msg in history])
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
        Based on this legal conversation:
        
        {context}
        
        Latest user message: {message}
        User location: {location}
        
        Generate 6-7 clear action steps the person should take next.
        Each step should be specific, practical, and immediately actionable.
        Format as a numbered list.
        Include:
        1. Immediate safety measures if relevant
        2. Documentation and evidence collection steps
        3. Filing appropriate legal complaints (specify which type and where)
        4. Protection orders or restraining orders if applicable (with specific process)
        5. Support services to contact (with specific names/numbers for {location})
        6. Legal aid options in {location} (be specific)
        7. Follow-up measures and timeline
        
        Each step should have 15-25 words of explanation.
        Be extremely direct and practical.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error generating action steps: {str(e)}")
        # Fallback action steps if API fails
        return """
        1. File an FIR at your nearest police station immediately. Bring ID proof and write down key incident details beforehand.
        
        2. Collect and preserve all evidence including photos, messages, medical reports, and witness statements in a secure location.
        
        3. Seek a protection order under the Domestic Violence Act, 2005 through your local magistrate court with help from a protection officer.
        
        4. Contact the Women's Helpline (1091) for immediate assistance, guidance, and connections to support services in your area.
        
        5. Consult with a lawyer from your District Legal Services Authority for free legal advice specific to your situation.
        
        6. Keep a detailed journal documenting all incidents with dates, times, descriptions, and impact on your wellbeing.
        
        7. Connect with local women's NGOs like Shakti Shalini or Jagori for counseling and ongoing support through the legal process.
        """

def get_helpline_numbers(location):
    """Get location-specific helpline numbers"""
    # Dictionary of helpline numbers for different locations
    helplines = {
        "delhi": {
            "Police": "112",
            "Women's Helpline": "1091",
            "Legal Aid": "1516",
            "District Legal Services Authority": "011-23071764",
            "Delhi Commission for Women": "181",
            "Shakti Shalini": "011-24373737"
        },
        "mumbai": {
            "Police": "112",
            "Women's Helpline": "1091",
            "Legal Aid": "1516",
            "District Legal Services Authority": "022-26591131",
            "Maharashtra State Commission for Women": "022-26592707",
            "Special Cell for Women & Children": "022-22186901"
        },
        "bangalore": {
            "Police": "112",
            "Women's Helpline": "1091",
            "Legal Aid": "1516",
            "District Legal Services Authority": "080-25501803",
            "Vanitha Sahayavani": "080-22943225",
            "Parihar Family Counselling Centre": "080-22942150"
        },
        "jaipur": {
            "Police": "112",
            "Women's Helpline": "1091",
            "Legal Aid": "1516",
            "District Legal Services Authority": "0141-2227481",
            "Rajasthan Commission for Women": "0141-2779001",
            "Mahila Suraksha Evam Salah Kendra": "0141-2744000"
        }
    }
    
    # Get helplines for specific location or use default
    return helplines.get(location.lower(), helplines["delhi"])

def format_ipc_sections_text(ipc_sections):
    """Format IPC sections as a human-readable string for the response"""
    if not ipc_sections or len(ipc_sections) == 0:
        return ""
    
    sections_text = "**"
    for i, section in enumerate(ipc_sections):
        section_name = section['section']
        
        # Try to extract section name and description
        if "(" in section_name and ")" in section_name:
            parts = section_name.split("(")
            section_num = parts[0].strip()
            description = parts[1].split(")")[0].strip()
            formatted_section = f"{section_num} ({description})"
        else:
            formatted_section = section_name.strip()
        
        if i > 0:
            if i == len(ipc_sections) - 1:
                sections_text += " and "
            else:
                sections_text += ", "
        
        sections_text += formatted_section
    
    sections_text += "**"
    
    # Add punishment for the most severe section if available
    severe_punishment = ""
    for section in ipc_sections:
        if section.get('punishment'):
            if "life" in section['punishment'].lower() or "death" in section['punishment'].lower():
                severe_punishment = section['punishment']
                break
    
    if severe_punishment:
        punishment_info = severe_punishment.lower()
        if "punishment" in punishment_info and "up to" in punishment_info:
            # Try to extract just the punishment term
            parts = punishment_info.split("up to")
            if len(parts) > 1:
                punishment_term = parts[1].strip()
                sections_text += f", which has punishment up to **{punishment_term}**"
    
    return sections_text

def generate_structured_response(stage, message, history, location, session_data, ipc_sections=None):
    """Generate conversational response based on conversation stage"""
    
    if stage == "GREETING":
        # Return a friendly greeting
        return generate_greeting()
    
    if stage == "INITIAL":
        # First response: Empathy + Initial Questions
        
        # Generate question bank and store in session
        question_bank = generate_question_bank(message, history)
        session_data['question_bank'] = question_bank
        session_data['questions_asked'] = 0
        
        # Select first set of questions
        questions_to_ask = select_next_questions(
            question_bank, 
            session_data['questions_asked'], 
            message, 
            history, 
            max_questions=3
        )
        
        # Update questions asked count
        session_data['questions_asked'] += len(questions_to_ask)
        
        # Generate empathy message
        empathy_message = generate_empathy_message(message, history)
        
        # Format questions conversationally
        questions_text = format_questions_as_conversation(questions_to_ask)
        
        response = f"""{empathy_message}

Let's start by understanding your situation better. {questions_text}"""
        return response
    
    elif stage == "INFORMATION_GATHERING":
        # Generate empathy message
        empathy_message = generate_empathy_message(message, history)
        
        # Get question bank from session or generate new one
        question_bank = session_data.get('question_bank', [])
        if not question_bank:
            question_bank = generate_question_bank(message, history)
            session_data['question_bank'] = question_bank
            
        # Get count of questions already asked
        questions_asked = session_data.get('questions_asked', 0)
        
        # Select next questions to ask
        questions_to_ask = select_next_questions(
            question_bank, 
            questions_asked, 
            message, 
            history, 
            max_questions=3
        )
        
        # Update questions asked count
        session_data['questions_asked'] = questions_asked + len(questions_to_ask)
        
        # Format questions conversationally
        questions_text = format_questions_as_conversation(questions_to_ask)
        
        response = f"""Thank you for sharing that information. {empathy_message}

{questions_text}"""
        return response
    
    elif stage == "LEGAL_ANALYSIS":
        # Provide legal assessment, IPC sections, and action steps
        legal_assessment = generate_legal_assessment(message, history, ipc_sections)
        
        # If legal assessment doesn't include IPC sections, add them
        if "section" not in legal_assessment.lower() and ipc_sections:
            ipc_text = format_ipc_sections_text(ipc_sections)
            if ipc_text:
                legal_assessment = f"Based on what you've shared, your case falls under {ipc_text}."
        
        # Format action steps as a list with emoji numbers
        action_steps = generate_action_steps(message, history, location).strip().split('\n')
        formatted_steps = ""
        count = 1
        for step in action_steps:
            clean_step = step.strip().replace("- ", "").replace("* ", "").replace(f"{count}. ", "")
            if clean_step:
                formatted_steps += f"{count}️⃣ **{clean_step}**\n\n"
                count += 1
        
        helplines = get_helpline_numbers(location)
        
        response = f"""{legal_assessment}

**Here are the steps you should take:**

{formatted_steps}

Are you currently safe? If you're in immediate danger, please **call {helplines.get('Police', '112')} (police emergency) right away.**

Would you like more specific information about any of these steps?"""
        
        return response
    
    # Default response if stage determination fails
    return get_fallback_response()

def get_fallback_response():
    """Provide a fallback response if API calls fail"""
    return """I apologize, but I'm having trouble processing your request right now. If you're facing a domestic violence situation, please call the Women's Helpline at 1091 or the Police Emergency number 112 immediately. You can also reach out to the Legal Aid Services at 1516."""

@app.route('/create_session', methods=['POST'])
def create_session():
    """Create a new conversation session"""
    try:
        session_id = str(uuid.uuid4())
        conversation_sessions[session_id] = {
            'created_at': datetime.now().isoformat(),
            'question_bank': None,
            'questions_asked': 0,
            'last_activity': datetime.now().isoformat()
        }
        return jsonify({"session_id": session_id, "status": "success"})
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        return jsonify({"error": "Failed to create session", "status": "error"}), 500

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json(force=True)
        if not data or 'message' not in data:
            return jsonify({"error": "Missing message"}), 400

        message = data['message']
        location = data.get('location', 'delhi')
        preferred_language = data.get('preferred_language', 'English')
        history = data.get('history', [])  # Get history from request
        session_id = data.get('session_id', None)
        
        # Get or create session data
        if session_id and session_id in conversation_sessions:
            session_data = conversation_sessions[session_id]
        else:
            # Create new session ID if not provided or invalid
            session_id = str(uuid.uuid4())
            session_data = {
                'question_bank': None,
                'questions_asked': 0
            }
            conversation_sessions[session_id] = session_data
        
        # Determine conversation stage based on message, history and session
        conversation_stage = determine_conversation_stage(message, history, session_data)
        
        # Get relevant IPC sections if in legal analysis stage
        ipc_sections = None
        if conversation_stage == "LEGAL_ANALYSIS":
            ipc_sections = get_relevant_ipc_sections(message, history)
        
        # Generate structured response based on stage
        response_text = generate_structured_response(
            conversation_stage, 
            message, 
            history, 
            location,
            session_data, 
            ipc_sections
        )

        # Update session with last activity timestamp
        if session_id in conversation_sessions:
            conversation_sessions[session_id]['last_activity'] = datetime.now().isoformat()

        return jsonify({
            'response': response_text,
            'status': 'success',
            'format': 'text',  # For markdown formatting
            'stage': conversation_stage,
            'session_id': session_id
        })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e), 
            "status": "error",
            "response": get_fallback_response()
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)