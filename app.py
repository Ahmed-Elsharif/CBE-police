import streamlit as st
import pandas as pd
import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document

# --- إعدادات الصفحة ---
st.set_page_config(page_title="المساعد الائتماني الذكي", page_icon="🤖", layout="wide")
st.title("🤖 المساعد الائتماني الخبير (DeepSeek RAG)")
st.subheader("تحليل القوانين والمبادرات الائتمانية بدقة عالية")

# --- التحميل الذكي للمحركات ---
@st.cache_resource
def load_embeddings():
    # هذا السطر سيقوم بتحميل الموديل في المجلد المؤقت للموقع تلقائياً
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    return HuggingFaceEmbeddings(model_name=model_name)
    
    # تحميل قاعدة البيانات
    if os.path.exists("./chroma_db_pro"):
        vectorstore = Chroma(persist_directory="./chroma_db_pro", embedding_function=embeddings)
    else:
        # بناء القاعدة إذا لم تكن موجودة
        df = pd.read_csv("my_database.csv", encoding='utf-8-sig', sep=';', on_bad_lines='skip', engine='python')
        df.columns = df.columns.str.strip().str.lower()
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        for _, row in df.iterrows():
            doc = Document(page_content=str(row['content']), metadata={"source": row['source'], "page": row['page']})
            documents.extend(text_splitter.split_documents([doc]))
        vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory="./chroma_db_pro")
    
    # إعداد الموديل (Ollama يعمل محلياً، للنشر السحابي نحتاج API)
    llm = OllamaLLM(model="deepseek-r1:8b", temperature=0)
    return vectorstore, llm

# بدء النظام
vectorstore, llm = load_system()

# --- إدارة الذاكرة (History) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض المحادثة السابقة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- واجهة الدردشة ---
if prompt := st.chat_input("اسألني أي شيء عن القوانين الائتمانية..."):
    # إضافة سؤال المستخدم للذاكرة والعرض
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("جاري مراجعة الملفات القانونية والتحليل..."):
            # 1. صياغة السياق من الذاكرة (آخر رسالتين)
            history_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-3:-1]])
            
            # 2. البحث عن المراجع
            docs = vectorstore.similarity_search(prompt, k=5)
            doc_context = "\n\n".join([f"[{d.metadata['source']} ص{d.metadata['page']}]: {d.page_content}" for d in docs])
            
            # 3. بناء الـ Prompt
            final_prompt = f"""أجب باللغة العربية الفصحى فقط. خذ سياق المحادثة والمراجع في الاعتبار.
            السياق السابق: {history_context}
            المراجع: {doc_context}
            السؤال: {prompt}
            الإجابة:"""
            
            # 4. توليد الإجابة وتنظيفها
            response = llm.invoke(final_prompt)
            full_answer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            
            # عرض الإجابة والمصادر
            st.markdown(full_answer)
            sources = set([f"{d.metadata['source']} (ص{d.metadata['page']})" for d in docs])
            st.info(f"📍 المصادر: " + " | ".join(sources))
            
    # حفظ إجابة البوت في الذاكرة
    st.session_state.messages.append({"role": "assistant", "content": full_answer})