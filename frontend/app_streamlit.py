import streamlit as st
import requests

# Configuration de la page
st.set_page_config(page_title="Excellia RAG", page_icon="🤖", layout="wide")

API_URL = "http://localhost:8000/query"

# Initialiser l'historique de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Excellia RAG Assistant")
st.markdown("Posez une question sur la documentation interne bancaire ou technique.")

# Afficher l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie
if prompt := st.chat_input("What is the IFSC code for Kamla Nagar?"):
    # Afficher le message utilisateur
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Appeler l'API
    payload = {"question": prompt, "top_k": 3, "stream": False}
    
    with st.chat_message("assistant"):
        with st.spinner("Searching internal documents and generating answer..."):
            try:
                response = requests.post(API_URL, json=payload, timeout=180)
                response.raise_for_status()
                data = response.json()
                
                answer = data.get("answer", "Error: No answer received.")
                sources = data.get("sources", [])
                
                full_response = answer
                if sources:
                    full_response += "\n\n---\n**📚 Sources:**\n"
                    for src in sources:
                        doc = src.get("document", "Unknown")
                        section = src.get("section", "N/A")
                        full_response += f"- **[{src['index']}]** {doc} | *{section}*\n"
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")