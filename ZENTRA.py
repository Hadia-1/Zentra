import os
import fitz  # PyMuPDF
import requests
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle
from PREP_PLAN import get_coal_study_planner

# Load model and label encoder once (outside function so it's not reloaded every time)
with open("zentra_intent_model.pkl", "rb") as f:
    intent_model = pickle.load(f)

with open("zentra_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# API Key
OPENROUTER_API_KEY = "insert your api key here"  

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- PDF Parsing ----------------
def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        return "\n".join(page.get_text() for page in doc)

def build_combined_corpus(folder="pdfs"):
    tfidf_corpus = []
    sbert_corpus = []
    titles = []
    sbert_embeddings = []

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            text = extract_text_from_pdf(path)
            titles.append(file)
            tfidf_corpus.append(text)
            sbert_corpus.append(text)
            sbert_embeddings.append(embedding_model.encode(text))

    return titles, tfidf_corpus, sbert_corpus, sbert_embeddings

titles, tfidf_docs, sbert_docs, sbert_embeddings = build_combined_corpus()
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(tfidf_docs)

# ---------------- IR Methods ----------------
def tfidf_search(query, top_k=1):
    query_vec = vectorizer.transform([query])
    sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    idxs = sim.argsort()[-top_k:][::-1]
    return sim[idxs[0]], tfidf_docs[idxs[0]][:1500] if idxs else ""

def sbert_search(query, top_k=1):
    query_vec = embedding_model.encode(query)
    sim = cosine_similarity([query_vec], sbert_embeddings).flatten()
    idxs = sim.argsort()[-top_k:][::-1]
    return sim[idxs[0]], sbert_docs[idxs[0]][:1500] if idxs else ""

def best_context(query):
    tfidf_score, tfidf_result = tfidf_search(query)
    sbert_score, sbert_result = sbert_search(query)
    return sbert_result if sbert_score > tfidf_score else tfidf_result

# ---------------- ZENTRA Response ----------------

def respond(user_input, history=None):
    if history is None:
        history = []

    # Predict intent
    predicted_label_index = intent_model.predict([user_input])[0]
    predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]

    # Route response based on intent
    if predicted_label == "blocked_query":
        return "Kiu nhi ho rhi prhai?🤨"

    if predicted_label == "identity_query":
        return ("I'm ZENTRA – your AI tutor built to help you master x86 Assembly Language, "
            "computer science, and engineering topics. I was created by H.Afsheen and Sehar T. using "
            "large language models and a strict academic framework. I focus only on studies, "
            "reject distractions, and respond using clean, ready-to-run MASM Irvine32 code. "
            "Think of me as your strict but helpful digital teacher.")

    # Default: Fall back to OpenAI + PDF context logic
    context = best_context(user_input)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost:7860",
        "Content-Type": "application/json"
    }

    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are ZENTRA, a strict but helpful AI tutor created by your founder H.Afsheen. "
                    "You specialize in x86 Assembly Language using the Irvine32 library for MASM. "
                    "Your should answer the study related question in theoratical form unless and untill someone asks you to generate code."
                    "Only Whenever the user asks for code generation, Your responsibility is to generate correct, clean, and fully functional assembly code that strictly follows the coding style defined below:\n\n"
                    "STYLE RULES:\n"
                    "- Always start with: INCLUDE Irvine32.inc\n"
                    "- Use `.data` section for all variables and messages\n"
                    "- Use `.code` section for all logic\n"
                    "- Begin code with: main PROC\n"
                    "- End with: main ENDP followed by END main\n"
                    "- Use these Irvine32 procedures correctly:\n"
                    "    - mov edx, offset variable\n"
                    "    - call WriteString\n"
                    "    - call WriteInt\n"
                    "    - call Crlf\n"
                    "    - call WaitMsg\n"
                    "- NEVER include: `.model`, `org 100h`, `int 20h`, `ret`, or any DOS-style instructions\n"
                    "- Add clear indentation and comments in your code\n"
                    "- Do not explain the code unless explicitly asked\n"
                    "- Never return invalid, partial, or incorrect syntax\n\n"
                    "BEHAVIOR RULES:\n"
                    "- Always review your code before replying\n"
                    "- Ensure the code compiles in MASM with Irvine32.inc without errors\n"
                    "- Incorporate context from user-provided PDFs when applicable\n\n"
                    "- Incorporate context from user-provided PDFs when someone asks study related questions\n\n"
                    "- Answer in a little polite way."
                    "- First analyze the language in which the user is speaking and then answer in the same language."
                    "- Your default talking language is English."    
                    "- Always First understand what the user is saying and then reply accordingly."
                    "FINAL OUTPUT:\n"
                    "- Only return working code, whenever the user asks for it\n"
                    "- Format exactly in the defined Irvine32 style\n"
                    "- Act as if the code will be compiled immediately"
                    "- First understand what the user is saying and then reply accordingly."
                )
            }] + history +
            [{
                "role": "user",
                "content": f"Question: {user_input}\n\nContext:\n{context}"
            }
        ]
    }

    try:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ---------------- Gradio UI ----------------
with gr.Blocks(css="""
.gr-block {
    background: linear-gradient(to right, #1a1b41, #2c2c54);
    color: white;
    font-family: 'Inter', sans-serif;
    border-radius: 12px;
    padding: 10px;
}
.gr-markdown h1 {
    font-family: 'Sora', sans-serif;
    font-weight: bold;
    text-align: center;
    color: white;
}
button {
    background-color: #005578 !important;
    color: white !important;
    border-radius: 12px !important;
}
button:hover {
    background-color: #0098d8 !important;
}
textarea, input {
    border-radius: 10px !important;
    background-color: #2c2c54 !important;
    color: white !important;
    border: 1px solid #444 !important;
}
""") as demo:
    gr.Markdown("""<h1>ZENTRA - Study-Only AI Tutor 👨‍🏫</h1><h3 style='color:gray;'>COAL Specialist</h3>""")
    chatbot = gr.Chatbot(label="Chat with ZENTRA", type='messages')
    user_input = gr.Textbox(placeholder="Ask about x86, AI, registers, memory, etc...", label="Your Question")

        # Add state tracking
    planner_visible = gr.State(False)

    with gr.Row():
        submit = gr.Button("Submit")
        clear = gr.Button("Clear Chat")
        toggle_btn = gr.Button("🕯️Don't panic, PLAN")
        planner_ui = get_coal_study_planner()
        planner_ui.visible = False

    def toggle_visibility(current_state):
        return gr.update(visible=not current_state), not current_state

    toggle_btn.click(
        fn=toggle_visibility,
        inputs=planner_visible,
        outputs=[planner_ui, planner_visible]
    )

    def user_submit(message, history):
        reply = respond(message, history)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        return "", history

    submit.click(fn=user_submit, inputs=[user_input, chatbot], outputs=[user_input, chatbot])
    user_input.submit(fn=user_submit, inputs=[user_input, chatbot], outputs=[user_input, chatbot])
    clear.click(lambda: [], None, chatbot)

demo.launch(share=True, inbrowser=True)

