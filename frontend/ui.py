import gradio as gr
import requests

# ✅ CORRIGEZ CETTE LIGNE : Utilisez /query et NON /query/stream
API_URL = "http://localhost:8000/query"

def chat(question, history):
    payload = {"question": question, "top_k": 3, "stream": False}
    
    try:
        # Timeout de 180s car le modèle 3B sur CPU peut prendre du temps
        response = requests.post(API_URL, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        
        answer = data.get("answer", "Error: No answer received.")
        sources = data.get("sources", [])
        
        # Formater les sources pour qu'elles s'affichent bien dans le chat
        if sources:
            src_text = "\n\n---\n**📚 Sources:**\n"
            for src in sources:
                doc = src.get("document", "Unknown")
                section = src.get("section", "N/A")
                src_text += f"- **[{src['index']}]** {doc} | *{section}*\n"
            return answer + src_text
        else:
            return answer
            
    except Exception as e:
        return f"An error occurred: {e}"

# Création de l'interface de chat
demo = gr.ChatInterface(
    fn=chat,
    title="Excellia RAG Assistant",
    description="Ask me anything about the internal company documents.",
)

if __name__ == "__main__":
    demo.launch(server_port=8001)