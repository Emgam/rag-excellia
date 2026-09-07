import chainlit as cl
import requests

API_QUERY_URL = "http://localhost:8000/query"

@cl.on_message
async def main(message: cl.Message):
    """
    Handles user questions. 
    File upload is disabled for demo stability.
    """
    # 1. Bloquer l'upload de fichiers pour protéger la base de données
    uploaded_files = [elem for elem in message.elements if elem.type == "file"]
    
    if uploaded_files:
        msg = cl.Message(content="⚠️ File upload is disabled during the live demonstration to preserve the database. Please ask a question about the existing 546 documents.")
        await msg.send()
        return

    # 2. Gérer la question de l'utilisateur
    if message.content.strip():
        # Message de chargement intelligent pour expliquer l'attente au jury
        msg = cl.Message(content="🧠 Analyzing query complexity (Smart Routing)...\n🔍 Searching Qdrant & BM25 (Hybrid Retrieval)...")
        await msg.send()
        
        payload = {
            "question": message.content,
            "top_k": 3,
            "stream": False
        }
        
        try:
            # Requête vers le backend FastAPI
            response = requests.post(API_QUERY_URL, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
            
            answer = data.get("answer", "Error: No answer received.")
            sources = data.get("sources", [])
            
            # Création des éléments interactifs pour les citations sur le côté
            elements = []
            for src in sources:
                doc_name = src.get("document", "Unknown")
                section = src.get("section", "N/A")
                page = src.get("page", "N/A")
                
                content = f"Document: {doc_name}\nSection: {section}\nPage: {page}"
                
                elements.append(
                    cl.Text(
                        name=f"Source {src['index']}", 
                        content=content, 
                        display="side"
                    )
                )
            
            # Mise à jour du message avec la réponse finale et les sources
            msg.content = answer
            msg.elements = elements
            await msg.update()
            
        except Exception as e:
            msg.content = f"An error occurred while contacting the backend: {e}"
            await msg.update()

@cl.on_chat_start
async def start():
    await cl.Message(content="Bonjour ! Je suis **Excellia RAG**. Posez-moi une question sur la documentation interne.").send()