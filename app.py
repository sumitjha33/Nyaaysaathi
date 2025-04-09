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
    """Generate a professional advocate-style greeting message"""
    return """Greetings, I am your legal consultant from NyaaySaathi. As a legal professional, I am here to provide you with information regarding your legal rights and options under Indian law.

Please share the details of your situation, and I will guide you through the appropriate legal procedures and remedies available to you. How may I assist you today?"""

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
        
        Return each question on a new line. Make questions formal and professional while remaining empathetic.
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
            "Can you provide the specific date and time when this incident first occurred?",
            "In which jurisdiction did these events take place? This is essential for determining applicable legal provisions.",
            "Could you identify all parties involved in the incident and specify your relationship with them?",
            "Have you sustained any physical injuries or received threats to your safety or wellbeing?",
            "What documentary evidence (such as communications, photographs, or medical reports) do you possess regarding this matter?",
            "Have you previously initiated any legal proceedings or filed complaints with law enforcement?",
            "Are there any ongoing threats or immediate safety concerns that require urgent attention?",
            "What physical, emotional, or financial consequences have you experienced as a result of these incidents?",
            "Could you describe the most recent incident in comprehensive detail?",
            "Are there witnesses who can corroborate the incidents you have described?"
        ]

def select_next_questions(question_bank, questions_asked, message, history, max_questions=3):
    """Select the next set of questions to ask based on context and priority"""
    try:
        if not question_bank or len(question_bank) == 0:
            # Fallback if no questions in bank
            return ["Could you please provide additional details regarding your legal situation?"]
        
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
    """Format multiple questions into a professional advocate-style conversation"""
    if not questions:
        return ""
        
    # Professional advocate-style question bridges
    bridges = [
        "For proper legal assessment, I need to inquire about", 
        "It would be pertinent to understand", 
        "Please provide details regarding", 
        "For the purposes of legal clarity, could you elaborate on",
        "As your legal consultant, I need to know",
        "It is essential for your case that I understand",
        "From a legal standpoint, I must ask about"
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
            result += f". Furthermore, {formatted_questions[i].lower()}?"
        else:
            result += f". {formatted_questions[i]}?"
            
    return result

def generate_empathy_message(message, history):
    """Generate an empathetic yet professional response based on the user's situation"""
    try:
        # Create context from history
        context = "\n".join([f"{msg['type']}: {msg['content']}" for msg in history]) 
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
        Based on this legal conversation:
        
        {context}
        
        Latest user message: {message}
        
        Generate a brief, professionally empathetic response (2-3 sentences) acknowledging the situation.
        The tone should be that of a dignified, experienced advocate who shows compassion while maintaining professional decorum.
        Do not provide legal advice in this response, just acknowledge their situation with appropriate gravity.
        If the user is clearly distressed, validate their concerns in a measured, respectful manner.
        If they mention harm or danger, express appropriate professional concern.
        
        Use formal legal language while remaining accessible.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error generating empathy message: {str(e)}")
        # Fallback empathy message if API fails
        return "I acknowledge the seriousness of the situation you have described. As your legal consultant, I assure you that the law provides remedies for such circumstances. Let us proceed methodically to address your concerns."

def generate_legal_assessment(message, history, ipc_sections):
    """Generate a professional legal assessment based on the conversation and IPC sections"""
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
        
        Provide a FORMAL, AUTHORITATIVE legal assessment (2-3 sentences) as a professional advocate would.
        Use proper legal terminology and citation format.
        Specifically identify IPC sections that apply using this format: 
        "Based on the facts presented, your matter falls within the ambit of [IPC sections]"
        
        Then add 1-2 sentences explaining the legal implications in professional yet accessible language.
        Maintain a dignified, authoritative tone throughout.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error generating legal assessment: {str(e)}")
        # Fallback assessment if API fails
        return "Based on the facts presented, your matter falls within the ambit of IPC Section 498A (matrimonial cruelty), Section 323 (voluntarily causing hurt), and Section 506 (criminal intimidation). These provisions specifically address the infliction of physical harm and threatening conduct as described, which constitute cognizable offenses under the Indian Penal Code."

def generate_action_steps(message, history, location):
    """Generate professional action steps based on the legal situation"""
    try:
        # Create context from history
        context = "\n".join([f"{msg['type']}: {msg['content']}" for msg in history])
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
        Based on this legal conversation:
        
        {context}
        
        Latest user message: {message}
        User location: {location}
        
        Generate 6-7 precisely formulated legal action steps the person should take, in the style of a professional advocate.
        Each step should be specific, procedurally correct, and immediately actionable.
        Format as a numbered list using formal legal terminology while remaining comprehensible.
        Include:
        1. Immediate safety protocols if pertinent
        2. Documentation and evidence preservation procedures
        3. Filing appropriate legal complaints (specify the exact type and venue)
        4. Protection orders or restraining orders if applicable (with specific procedural requirements)
        5. Support services to contact (with specific institutions/numbers for {location})
        6. Legal aid options in {location} (provide specific institutions)
        7. Follow-up legal procedures and timeline
        
        Each step should have 15-25 words of precise explanation using formal legal language.
        Maintain professional advocate tone throughout.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error generating action steps: {str(e)}")
        # Fallback action steps if API fails
        return """
        1. Lodge a First Information Report (FIR) at the jurisdictional police station with proper identification documentation and a concise written account of the incident.
        
        2. Secure and catalogue all evidentiary materials including photographic documentation, electronic communications, medical certificates, and witness testimonials in chronological order.
        
        3. Petition for a Protection Order under Section 18 of the Protection of Women from Domestic Violence Act, 2005 through the appropriate Magistrate's Court with assistance from a Protection Officer.
        
        4. Establish contact with the Women's Helpline (1091) for immediate intervention, procedural guidance, and referral to appropriate support services in your jurisdiction.
        
        5. Seek legal consultation through your District Legal Services Authority for pro bono legal counsel tailored to the specific circumstances of your case.
        
        6. Maintain a comprehensive chronological record documenting all incidents with precise dates, times, detailed descriptions, and impact assessment for evidentiary purposes.
        
        7. Establish communication with recognized women's rights organizations such as Shakti Shalini or Jagori for psychological support and procedural assistance throughout the legal proceedings.
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
    """Generate professional advocate-style response based on conversation stage"""
    
    if stage == "GREETING":
        # Return a professional greeting
        return generate_greeting()
    
    if stage == "INITIAL":
        # First response: Professional empathy + Initial Questions
        
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

In order to provide you with accurate legal counsel, I need to gather pertinent information about your case. {questions_text}"""
        return response
    
    elif stage == "INFORMATION_GATHERING":
        # Generate professional empathy message
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
        
        response = f"""Thank you for providing this information. {empathy_message}

For further clarity on your legal position, {questions_text}"""
        return response
    
    elif stage == "LEGAL_ANALYSIS":
        # Provide legal assessment, IPC sections, and action steps in professional manner
        legal_assessment = generate_legal_assessment(message, history, ipc_sections)
        
        # If legal assessment doesn't include IPC sections, add them
        if "section" not in legal_assessment.lower() and ipc_sections:
            ipc_text = format_ipc_sections_text(ipc_sections)
            if ipc_text:
                legal_assessment = f"Based on the facts presented, your matter appears to fall within the purview of {ipc_text}."
        
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

**I recommend the following course of action:**

{formatted_steps}

Regarding your immediate safety: If you perceive any imminent threat to your wellbeing, please **immediately contact {helplines.get('Police', '112')} (police emergency).**

Would you require any specific clarification regarding the recommended procedures?"""
        
        return response
    
    # Default response if stage determination fails
    return get_fallback_response()

def get_fallback_response():
    """Provide a professional fallback response if API calls fail"""
    return """I regret to inform you that I am experiencing technical difficulties in processing your query at this moment. If you are facing a situation requiring immediate legal intervention, please contact the Women's Helpline at 1091 or the Police Emergency number 112 without delay. For legal assistance, you may also reach the Legal Aid Services at 1516."""

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
        
        # Add language instruction to model prompt
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
        IMPORTANT: Respond ONLY in {preferred_language} language.
        
        User message: {message}
        Location: {location}
        Context: {history}
        
        Generate a helpful legal response in {preferred_language} language.
        Keep legal terms clear but explain everything in {preferred_language}.
        """
        
        response = model.generate_content(prompt)
        
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