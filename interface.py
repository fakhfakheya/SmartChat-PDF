import streamlit as st
import os
from pdf_utils import pdf_to_text, split_documents
from faiss_index import add_to_faiss, query_faiss
from llm_client import get_mistral_client, generate_answer

# --- Clé API NVIDIA ---
API_KEY = "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"

# --- Initialisation du client Mistral ---
client = get_mistral_client(API_KEY)

# --- Configuration Streamlit ---
st.set_page_config(page_title="RAG Chat", page_icon="💬")
st.title("💬 SmartChat PDF")
st.write("Téléversez un PDF médical, puis discutez librement avec l'assistant.")

# --- Initialisation de session ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []
if "faiss_ready" not in st.session_state:
    st.session_state.faiss_ready = False

# --- Téléversement de PDF ---
uploaded_file = st.file_uploader("📄 Choisir un PDF", type=["pdf"])

if uploaded_file is not None and not st.session_state.faiss_ready:
    pdf_folder = "pdfs"
    os.makedirs(pdf_folder, exist_ok=True)
    pdf_path = os.path.join(pdf_folder, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("🔍 Extraction du texte..."):
        text = pdf_to_text(pdf_path)
        chunks = split_documents(text, chunk_size=300)
        st.session_state.documents.extend(chunks)
        add_to_faiss(st.session_state.documents)
        st.session_state.faiss_ready = True
    st.success("✅ Document indexé avec succès ! Vous pouvez maintenant discuter.")

# --- Affichage du chat existant ---
st.markdown("### 💬 Discussion")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Zone d'entrée utilisateur ---
if st.session_state.faiss_ready:
    if prompt := st.chat_input("Posez votre question médicale ici..."):
        # Ajouter la question de l'utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Récupérer contexte FAISS
        with st.spinner("🧠 Recherche du contexte pertinent..."):
            results = query_faiss(prompt, top_k=3)
            context = "\n".join([r[0] for r in results])

        # Créer un historique du dialogue
        history = ""
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                history += f"Utilisateur : {msg['content']}\n"
            else:
                history += f"Assistant : {msg['content']}\n"

        # Générer la réponse
        with st.spinner("💡 Génération de la réponse..."):
            answer = generate_answer(client, f"{context}\n\nHistorique:\n{history}", prompt)

        # Afficher et sauvegarder la réponse
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("⬆️ Téléversez un PDF pour démarrer la conversation.")
