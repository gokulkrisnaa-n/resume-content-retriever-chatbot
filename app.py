import os
from flask import Flask, request, jsonify, render_template_string
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ─── The Chat UI (HTML page shown in the browser) ───────────────────────────
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
    <input id="user-input" type="text" placeholder="Type your question…"/>
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

# ─── Build the knowledge base (runs once when the app starts) ────────────────
def create_knowledge_base():
    loader = TextLoader("knowledge.txt", encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # 1. Define a system prompt that guides the LLM
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know."
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 2. Build the LCEL Chains
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return qa_chain

# Build it NOW (at import time — works with gunicorn's multi-worker model)
print("Loading knowledge base…")
qa_chain = create_knowledge_base()
print("Ready!")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Please provide a question."})
    try:
        # 3. The input key is now "input" instead of "query", 
        # and the output key is "answer" instead of "result"
        result = qa_chain.invoke({"input": question})
        return jsonify({"answer": result["answer"]})
    except Exception as exc:
        return jsonify({"error": str(exc)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)