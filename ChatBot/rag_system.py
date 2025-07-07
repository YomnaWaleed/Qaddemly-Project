# rag_system.py
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from groq import Groq
from dotenv import load_dotenv


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


# === Step 1: JSON Loader ===
def load_json_data(file_path):
    loader = JSONLoader(
        file_path=file_path,
        jq_schema='.[] | "Question: \(.question) Answer: \(.answer)"',
        text_content=False,
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)


# === Step 2: Load Embeddings and FAISS Index ===
def build_vectorstore():
    docs = load_json_data("data/QA.json")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    vectorstore.save_local("QA_faiss_index")
    return vectorstore


# === Step 3: Load Saved Vectorstore ===
from pathlib import Path

def load_vectorstore():
    index_dir = Path("QA_faiss_index")
    index_file = index_dir / "index.faiss"
    
    if not index_file.exists():
        print(f"[WARNING] FAISS index not found at {index_file}, building new one...")
        return build_vectorstore()
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("[INFO] Loading existing FAISS index...")
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True
    )


# === Step 4: Prompt Template ===
PROMPT = PromptTemplate(
    template="""Answer the question based only on the following context.\n{context}\n\nQuestion: {question}\nAnswer in the exact format:\nAnswer: [your answer here]""",
    input_variables=["context", "question"],
)


# === Step 5: Final RAG Answer Function ===
def get_answer(question):
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are Qaddemly's AI assistant."},
            {
                "role": "user",
                "content": PROMPT.format(context=context, question=question),
            },
        ],
        model="llama3-8b-8192",
        temperature=0.7,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()