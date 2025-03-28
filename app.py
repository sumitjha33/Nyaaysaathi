import os
from dotenv import load_dotenv
from pinecone import Pinecone  # Update import back
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from datetime import datetime
from typing import Optional, List, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
load_dotenv()

def Hugging_face_embedding():
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embeddings = Hugging_face_embedding()

# Get API keys from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["PINECONE_ENVIRONMENT"] = "us-east-1"  # Change based on your Pinecone region
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize Pinecone Client - Updated initialization
pc = Pinecone(api_key=PINECONE_API_KEY)

# Update existing index initialization
index = pc.Index("nyaaysaathi")
docsearch = PineconeVectorStore.from_existing_index(
    index_name="nyaaysaathi",
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY,
    environment="us-east-1"
)

retriever = docsearch.as_retriever(search_type='similarity', search_kwargs={"k": 4})

retriever_docs = retriever.invoke("My husband beat me everyday. what can i do for it?")

# Initialize the model
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.6)

system_prompt = ('''You are a highly knowledgeable, empathetic, and trustworthy **AI Legal Assistant**, designed to assist users with their legal concerns in a **human-like, conversational, and solution-oriented** manner. Your goal is to provide **accurate legal guidance** while ensuring users feel **understood, reassured, and empowered** to take the right actions.  

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
"I'm really sorry you're facing this. You have legal rights, and I’ll guide you step by step to get the right help. Let’s start by understanding your situation better. Can you tell me if you have suffered any physical injuries or threats?"  

💬 **User:** Yes, he hits me and threatens to kill me if I tell anyone.  
🤖 **Legal Assistant:**  
"This is very serious. Your case falls under **IPC Section 498A (cruelty), Section 323 (causing hurt), and possibly Section 506 (criminal intimidation).** If his threats are severe, **Section 307 (attempt to murder)** might apply, which has punishment up to **life imprisonment**."  

💬 **User:** What should I do now?  
🤖 **Legal Assistant:**  
"You should immediately:  
1️⃣ **File an FIR at your nearest police station** under these sections.  
2️⃣ **Seek a protection order** under the **Domestic Violence Act, 2005**, to prevent further harm.  
3️⃣ Collect **any evidence** (medical reports, messages, witnesses).  
4️⃣ If needed, contact a **women’s helpline (181) or free legal aid services.**  

Are you currently safe? If you're in immediate danger, please **call 112 (police emergency) right away.**"  

---  

### **Special Features:**  
✔️ **Conversational & Human-Like** – The AI ensures the user feels heard and supported.  
✔️ **Context-Based Legal Assistance** – Asks counter-questions before giving legal advice.  
✔️ **Accurate IPC Sections & Laws** – Provides exact punishments and next steps.  
✔️ **Multilingual Support** – Can assist users in multiple languages.  
✔️ **Ensures Safety First** – Prioritizes user well-being in serious cases.  

"{context}"  
''')

# Initialize memory with proper configuration
memory = ConversationBufferWindowMemory(
    k=5,
    memory_key="chat_history",
    return_messages=True,
    input_key="input",
    output_key="output"
)

# Update prompt template to properly include chat history
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("system", "Previous conversation:\n{chat_history}"),
    ("human", "{input}"),
])

# Create the chain without memory parameter
question_answer_chain = create_stuff_documents_chain(
    llm, 
    prompt,
    document_variable_name="context",
)

rag_chain = create_retrieval_chain(
    retriever, 
    question_answer_chain
)

def format_chat_history(chat_history):
    formatted_history = ""
    for message in chat_history:
        if hasattr(message, 'content') and hasattr(message, 'type'):
            role = "Assistant" if message.type == "ai" else "Human"
            formatted_history += f"{role}: {message.content}\n"
    return formatted_history

def get_response(query: str):
    # Get chat history
    history_vars = memory.load_memory_variables({})
    chat_history = format_chat_history(history_vars.get("chat_history", []))
    
    # Get relevant documents from vector store
    docs = retriever.get_relevant_documents(query)
    
    # Combine context
    context = "\n\n".join([
        "Vector Store Results:",
        "\n".join([doc.page_content for doc in docs])
    ])
    
    # Get response
    response = rag_chain.invoke({
        "input": query,
        "chat_history": chat_history,
        "context": context
    })
    
    # Save the interaction to memory
    memory.save_context(
        {"input": query},
        {"output": response['answer']}
    )
    
    return response['answer']

def get_formatted_history() -> List[Dict[str, str]]:
    history_vars = memory.load_memory_variables({})
    formatted_history = []
    
    for message in history_vars.get("chat_history", []):
        if hasattr(message, 'content') and hasattr(message, 'type'):
            formatted_history.append({
                "role": "assistant" if message.type == "ai" else "user",
                "content": message.content
            })
    
    return formatted_history

# Initialize FastAPI
app = FastAPI(
    title="NyaySaathi API",
    description="Legal Assistant API with chat functionality",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatInput(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(chat_input: ChatInput):
    response = get_response(chat_input.message)
    history = get_formatted_history()
    return {
        "response": response,
        "history": history
    }

@app.get("/chat-history")
async def get_chat_history():
    return get_formatted_history()