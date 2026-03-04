import streamlit as st
import pandas as pd
import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="المساعد الائتماني الذكي", page_icon="🏦", layout="wide")
st.title("🤖 المساعد الائتماني الخبير (DeepSeek & Groq)")
st.markdown("---")

# --- 2. دالة تحميل النظام (محمية بذاكرة مؤقتة) ---
@st.cache_resource
def load_full_system():
    # تحميل التضمين (Embeddings) - سيعمل تلقائياً على السيرفر
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # إدارة قاعدة البيانات
    if os.path.exists("./chroma_db_pro"):
        vectorstore = Chroma(persist_directory="./chroma_db_pro", embedding_function=embeddings)
    else:
        if not os.path.exists("my_database.csv"):
            st.error("❌ ملف 'my_database.csv' غير موجود في المجلد الرئيسي!")
            st.stop()
            
        df = pd.read_csv("my_database.csv", encoding='utf-8-sig', sep=';', on_bad_lines='skip', engine='python')
        df.columns = df.columns.str.strip().str.lower()
        
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        for _, row in df.iterrows():
            # التأكد من وجود الأعمدة المطلوبة
            content = str(row.get('content', ''))
            source = str(row.get('source', 'Unknown'))
            page = str(row.get('page', '0'))
            
            doc = Document(page_content=content, metadata={"source": source, "page": page})
            documents.extend(text_splitter.split_documents([doc]))
            
        vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory="./chroma_db_pro")
    
    # إعداد موديل الذكاء الاصطناعي عبر API
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        st.error("❌ مفتاح GROQ_API_KEY غير مضبوط في الـ Secrets!")
        st.stop()
    
    # استخدمنا llama-3.3-70b لأنه الأكثر استقراراً وقوة في الـ RAG حالياً
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.3-70b-versatile", 
        groq_api_key=api_key
    )
    
    return vectorstore, llm

# تشغيل النظام
try:
    vectorstore, llm = load_full_system()
    st.sidebar.success("✅ النظام متصل بالقاعدة والمحرك")
except Exception as e:
    st.error(f"❌ فشل تحميل النظام: {e}")
    st.stop()

# --- 3. إدارة ذاكرة المحادثة ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض الرسائل السابقة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. معالجة سؤال المستخدم ---
if prompt := st.chat_input("اسأل عن القوانين أو المبادرات الائتمانية..."):
    # عرض سؤال المستخدم
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # توليد الإجابة
    with st.chat_message("assistant"):
        with st.spinner("جاري مراجعة المراجع والتحليل..."):
            try:
                # أ. صياغة السياق من الذاكرة
                history_context = ""
                if len(st.session_state.messages) > 1:
                    history_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-3:-1]])

                # ب. البحث عن المراجع في ملفاتك
                docs = vectorstore.similarity_search(prompt, k=5)
                doc_context = "\n\n".join([f"[{d.metadata['source']} ص{d.metadata['page']}]: {d.page_content}" for d in docs])
                
                # ج. بناء الـ Prompt الاحترافي
                final_prompt = f"""أنت مستشار بنكي وقانوني خبير. أجب باللغة العربية الفصحى فقط.
                يجب أن تكون الإجابة دقيقة، مبنية على المراجع، وغير مكررة.

                [سياق المحادثة السابقة]:
                {history_context}

                [المراجع القانونية المتاحة]:
                {doc_context}

                [السؤال الحالي]:
                {prompt}

                الإجابة القانونية المركزّة:"""

                # د. استدعاء الموديل (تنسيق Messages لمنع BadRequestError)
                response = llm.invoke([HumanMessage(content=final_prompt)])
                full_answer = response.content
                
                # تنظيف من أي وسوم تفكير
                full_answer = re.sub(r'<think>.*?</think>', '', full_answer, flags=re.DOTALL).strip()
                
                # عرض النتيجة
                st.markdown(full_answer)
                
                # عرض المصادر بشكل منظم
                sources = set([f"{d.metadata['source']} (ص{d.metadata['page']})" for d in docs])
                st.info("📍 **المصادر المستند إليها:**\n\n" + "\n\n".join([f"- {s}" for s in sources]))
                
                # حفظ الإجابة في الذاكرة
                st.session_state.messages.append({"role": "assistant", "content": full_answer})

            except Exception as e:
                st.error(f"❌ حدث خطأ أثناء معالجة السؤال: {e}")
