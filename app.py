import streamlit as st
import pandas as pd
import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq  # التغيير للنشر
from langchain_core.documents import Document

# --- إعدادات الصفحة ---
st.set_page_config(page_title="المساعد الائتماني الذكي", page_icon="🤖", layout="wide")
st.title("🤖 المساعد الائتماني الخبير (DeepSeek RAG)")

# --- دالة التحميل الموحدة ---
@st.cache_resource
def load_full_system():
    # 1. تحميل الـ Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # 2. إعداد قاعدة البيانات
    if os.path.exists("./chroma_db_pro"):
        vectorstore = Chroma(persist_directory="./chroma_db_pro", embedding_function=embeddings)
    else:
        # بناء القاعدة إذا لم تكن موجودة (لأول مرة على السيرفر)
        if not os.path.exists("my_database.csv"):
            st.error("❌ ملف my_database.csv غير موجود!")
            st.stop()
            
        df = pd.read_csv("my_database.csv", encoding='utf-8-sig', sep=';', on_bad_lines='skip', engine='python')
        df.columns = df.columns.str.strip().str.lower()
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        for _, row in df.iterrows():
            doc = Document(page_content=str(row['content']), metadata={"source": row['source'], "page": row['page']})
            documents.extend(text_splitter.split_documents([doc]))
        vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory="./chroma_db_pro")
    
    # 3. إعداد الموديل عبر API للنشر (Groq)
    # تأكد من إضافة GROQ_API_KEY في Streamlit Secrets
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        st.warning("⚠️ يرجى ضبط GROQ_API_KEY في الإعدادات")
    
    llm = ChatGroq(
        temperature=0, 
        model_name="deepseek-r1-distill-llama-70b", 
        groq_api_key=api_key
    )
    
    return vectorstore, llm

# تشغيل النظام
try:
    vectorstore, llm = load_full_system()
except Exception as e:
    st.error(f"حدث خطأ في تحميل النظام: {e}")
    st.stop()

# --- إدارة الذاكرة وعرض الدردشة (نفس منطقك السابق) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("اسألني أي شيء عن القوانين الائتمانية..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("جاري مراجعة الملفات القانونية والتحليل..."):
            history_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-3:-1]])
            docs = vectorstore.similarity_search(prompt, k=5)
            doc_context = "\n\n".join([f"[{d.metadata['source']} ص{d.metadata['page']}]: {d.page_content}" for d in docs])
            
            final_prompt = f"""أجب باللغة العربية الفصحى فقط. خذ سياق المحادثة والمراجع في الاعتبار.
            السياق السابق: {history_context}
            المراجع: {doc_context}
            السؤال: {prompt}
            الإجابة:"""
            
            # استدعاء الموديل (تعديل بسيط ليتناسب مع ChatGroq)
            response = llm.invoke(final_prompt)
            full_answer = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()
            
            st.markdown(full_answer)
            sources = set([f"{d.metadata['source']} (ص{d.metadata['page']})" for d in docs])
            st.info(f"📍 المصادر: " + " | ".join(sources))
            
    st.session_state.messages.append({"role": "assistant", "content": full_answer})
