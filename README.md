# 🤖 Knowledge Retrieval AI Chatbot — Deployed on Azure

A RAG (Retrieval-Augmented Generation) chatbot that reads any text document and answers questions about it — built with Python, LangChain, and Gemini, deployed on Microsoft Azure App Service.

> **Week 1 of my cloud learning journey.** No Docker. No prior deployment experience. Just Python and a lot of terminal commands.

---

## 🚀 Live Demo

🔗 **[resume-content-retriever-app.azurewebsites.net](https://resume-content-retriever-app.azurewebsites.net/)**

---

## 🧠 How It Works

```
User Question → Flask App → FAISS Vector Search → GPT-4o-mini → Answer
```

1. **Document Loading** — Reads `knowledge.txt` on startup
2. **Chunking** — Splits text into 500-character overlapping chunks
3. **Embedding** — Converts chunks into vectors using `text-embedding-3-small`
4. **Retrieval** — FAISS finds the 3 most relevant chunks for your question
5. **Generation** — Gemini-2.5-Flash reads those chunks and writes a clear answer

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Web Framework | Flask |
| AI Orchestration | LangChain |
| Vector Store | FAISS (in-memory) |
| LLM | Gemini 2.5 Flash model |
| Embeddings | gemini-embedding-001 |
| Production Server | Gunicorn |
| Cloud Hosting | Microsoft Azure App Service (Free F1 tier) |

---

## 📁 Project Structure

```
chatbot/
├── app.py            # Main Flask application + LangChain RAG pipeline
├── knowledge.txt     # Your knowledge document (customize this!)
├── requirements.txt  # Python dependencies
├── .gitignore        # Files excluded from Git
└── README.md         # You are here
```

---

## ⚙️ Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI API key
```bash
# Windows
set GEMINI_API_KEY=sk-your-key-here

# Mac / Linux
export GEMINI_API_KEY=sk-your-key-here
```

### 5. Customize the knowledge base
Edit `knowledge.txt` with any content you want the chatbot to know about — a resume, product documentation, company FAQs, research notes, anything.

### 6. Run the app
```bash
python app.py
```

Visit **http://localhost:8000** in your browser.

---

## ☁️ Deploy to Azure (Free Tier)

> Full step-by-step tutorial included in this project.

### Prerequisites
- Azure account (free or pay-as-you-go)
- [Azure CLI](https://aka.ms/installazurecliwindows) installed

### Quick deploy commands
```bash
# 1. Login
az login

# 2. Create resource group
az group create --name chatbot-rg --location eastus

# 3. Create free App Service plan
az appservice plan create --name chatbot-plan --resource-group chatbot-rg --sku F1 --is-linux

# 4. Create web app (replace YOUR-UNIQUE-NAME)
az webapp create --name YOUR-UNIQUE-NAME --resource-group chatbot-rg --plan chatbot-plan --runtime "PYTHON:3.11"

# 5. Set your API key securely
az webapp config appsettings set --name YOUR-UNIQUE-NAME --resource-group chatbot-rg --settings GEMINI_API_KEY="sk-your-key-here"

# 6. Zip and deploy
zip deploy.zip app.py requirements.txt knowledge.txt
az webapp deploy --name YOUR-UNIQUE-NAME --resource-group chatbot-rg --src-path deploy.zip --type zip

# 7. Set startup command
az webapp config set --name YOUR-UNIQUE-NAME --resource-group chatbot-rg --startup-file "gunicorn --bind=0.0.0.0:8000 app:app --timeout 120"

# 8. Restart
az webapp restart --name YOUR-UNIQUE-NAME --resource-group chatbot-rg
```

**Cost:** Azure F1 tier = **$0**. OpenAI API for casual testing = **~$0.01**.

---

## 💡 Customization

**Change the knowledge source:** Replace the content in `knowledge.txt` with anything — a product manual, your resume, company policies, research papers.

**Adjust retrieval sensitivity:** In `app.py`, change `k=3` to retrieve more or fewer chunks per question.

---

## 📌 Lessons Learned

- Deployment teaches you things that coding alone never does
- Environment variables are the right way to handle API keys — never hardcode them
- Azure App Service F1 is a perfect zero-cost sandbox for learning cloud deployment
- RAG is surprisingly simple to implement but powerful in practice

---

## 👤 Author

**Gokul Krisnaa Nattarayan**
MS Data Science, University of Houston – Clear Lake

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/your-username)

---

## 📄 License

MIT License — free to use, modify, and distribute.