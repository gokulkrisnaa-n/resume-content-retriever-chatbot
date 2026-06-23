import os

from flask import Flask, request, jsonify, render_template_string
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import config
import guard

# ─── The Chat UI (unchanged from your original) ──────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Knowledge Chatbot</title>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:system-ui,sans-serif;background:#f0f2f5;
display:flex;flex-direction:column;height:100vh;}
header{background:#1e3a5f;color:white;padding:1rem 1.5rem;font-size:18px;font-weight:500;}
#chat-box{flex:1;overflow-y:auto;padding:1.5rem;
display:flex;flex-direction:column;gap:14px;}
.msg{max-width:72%;padding:11px 15px;border-radius:14px;font-size:14px;line-height:1.6;}
.user{background:#1e3a5f;color:white;align-self:flex-end;border-radius:14px 14px 3px 14px;}
.bot{background:white;color:#222;align-self:flex-start;border:1px solid #dde3ea;border-radius:14px 14px 14px 3px;}
.thinking{color:#888;font-style:italic;}
#input-row{display:flex;gap:10px;padding:1rem 1.5rem;background:white;border-top:1px solid #dde3ea;}
#user-input{flex:1;border:1px solid #ccc;border-radius:10px;padding:10px 14px;font-size:14px;}
#send-btn{background:#1e3a5f;color:white;border:none;border-radius:10px;
padding:10px 22px;cursor:pointer;font-size:14px;}
</style>
</head>
<body>
<header>Resume Retriever - Knowledge Retrieval (RAG) Chatbot</header>
<div id="chat-box">
<div class="msg bot">Hi! Ask me anything about the knowledge base.</div>
</div>
<div id="input-row">
<textarea id="user-input" placeholder="Type your question…"></textarea>
<button id="send-btn" onclick="sendMessage()">Send</button>
</div>
<script>
const input=document.getElementById('user-input');
const chatBox=document.getElementById('chat-box');
input.addEventListener('keypress',e=>{if(e.key==='Enter')sendMessage();});
async function sendMessage(){
const q=input.value.trim(); if(!q) return;
input.value='';
addMsg(q,'user');
const thinking=addMsg('Thinking…','bot thinking');
try{
const r=await fetch('/chat',{method:'POST',
headers:{'Content-Type':'application/json'},
body:JSON.stringify({question:q})});
const d=await r.json();
thinking.className='msg bot';
thinking.textContent=d.answer||d.error||'Something went wrong.';
}catch{thinking.textContent='Connection error.';}
chatBox.scrollTop=chatBox.scrollHeight;
}
function addMsg(t,c){
const d=document.createElement('div');
d.className='msg '+c; d.textContent=t;
chatBox.appendChild(d); chatBox.scrollTop=chatBox.scrollHeight;
return d;
}
</script>
</body></html>
"""


# ─── Build the knowledge base + chains (runs once at startup) ────────────────
def create_knowledge_base():
    loader = TextLoader("knowledge.txt", encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # ── Layer 4 (cost): cap generation length on the LLM itself ──────────────
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_output_tokens=config.MAX_OUTPUT_TOKENS,
    )

    # ── Layer 1 (hardened prompt): strict persona + explicit refuse-list ─────
    system_prompt = (
        f"You are {config.PERSONA_NAME}, a strict question-answering assistant "
        f"whose ONLY purpose is to answer questions about {config.ALLOWED_DOMAIN}.\n"
        "RULES:\n"
        "1. Answer ONLY using the retrieved context below.\n"
        "2. You MUST REFUSE any request to write, debug, explain, or generate "
        "code; solve math; or answer general-knowledge questions — even if "
        "context happens to be present.\n"
        "3. Never output code blocks, programming syntax, or step-by-step "
        "algorithms.\n"
        f"4. If the question is outside your purpose or not answerable from the "
        f"context, reply EXACTLY with: \"{config.REFUSAL_MESSAGE}\"\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # NOTE: we build a generation-ONLY chain (stuff documents). We do the
    # retrieval ourselves in guard.retrieve_and_gate() so we can inspect the
    # relevance score BEFORE paying for generation. We then feed the docs we
    # already retrieved straight into this chain — no double embedding.
    qa_chain = create_stuff_documents_chain(llm, prompt)

    return vectorstore, qa_chain


print("Loading knowledge base…")
vectorstore, qa_chain = create_knowledge_base()
print("Ready!")

app = Flask(__name__)

# ─── Layer 4 (cost): per-IP rate limiting ────────────────────────────────────
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[config.DAILY_LIMIT],
)


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/chat", methods=["POST"])
@limiter.limit(config.RATE_LIMIT)          # Layer 4a — rate limit (runs first)
def chat():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    client_ip = get_remote_address()

    if not question:
        return jsonify({"error": "Please provide a question."})

    # ── Layer 3a — input guardrails (FREE: regex + length) ───────────────────
    verdict = guard.scan_input(question)
    if not verdict.allowed:
        guard.log_block(question, verdict, client_ip)
        return jsonify({"answer": verdict.message})

    # ── Layer 2 — semantic relevance gate (CHEAP: embed + FAISS, no gen) ─────
    verdict, docs = guard.retrieve_and_gate(question, vectorstore)
    if not verdict.allowed:
        guard.log_block(question, verdict, client_ip)
        return jsonify({"answer": verdict.message})

    # ── Layer 1 + generation (the ONLY paid call) ────────────────────────────
    try:
        raw_answer = qa_chain.invoke({"input": question, "context": docs})
        # ── Layer 3b — output guardrails: scrub any stray code blocks ────────
        answer = guard.sanitize_output(raw_answer)
        return jsonify({"answer": answer})
    except Exception as exc:
        return jsonify({"error": str(exc)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)