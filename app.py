import streamlit as st
import pandas as pd
import os
import re
import gc  # لتنظيف الذاكرة بشكل يدوي
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="المساعد الائتماني الذكي", page_icon="🏦", layout="wide")
st.title("🤖 المساعد الائتماني الذكي")
st.markdown("---")

# --- 2. دالة التحميل المحسنة (Memory Optimized) ---
@st.cache_resource
def load_full_system():
    # استخدام موديل خفيف جداً لتقليل استهلاك الرام
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # مسار قاعدة البيانات
    if os.path.exists("./chroma_db_pro"):
        vectorstore = Chroma(persist_directory="./chroma_db_pro", embedding_function=embeddings)
    else:
        # بناء القاعدة فقط إذا لم تكن موجودة
        if not os.path.exists("my_database.csv"):
            st.error("❌ ملف 'my_database.csv' غير موجود!")
            st.stop()
            
        df = pd.read_csv("my_database.csv", encoding='utf-8-sig', sep=';', on_bad_lines='skip', engine='python')
        df.columns = df.columns.str.strip().str.lower()
        
        documents = []
        # تقليل الـ Chunk size لتقليل استهلاك الذاكرة أثناء المعالجة
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        
        for _, row in df.iterrows():
            content = str(row.get('content', ''))
            source = str(row.get('source', 'Unknown'))
            page = str(row.get('page', '0'))
            doc = Document(page_content=content, metadata={"source": source, "page": page})
            documents.extend(text_splitter.split_documents([doc]))
            
        vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory="./chroma_db_pro")
        
        # تنظيف الذاكرة فوراً بعد البناء
        del df
        del documents
        gc.collect() 
    
    # إعداد Groq (لا يستهلك ذاكرة السيرفر لأنه يعمل سحابياً)
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        st.error("❌ GROQ_API_KEY مفقود في Secrets!")
        st.stop()
        
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.3-70b-versatile", 
        groq_api_key=api_key
    )
    
    return vectorstore, llm

# محاولة تشغيل النظام مع تنظيف مسبق
gc.collect()
try:
    vectorstore, llm = load_full_system()
    st.sidebar.success("✅ النظام متصل (استهلاك الرام مستقر)")
except Exception as e:
    st.error(f"❌ فشل تحميل النظام: {e}")
    st.stop()

# --- 3. إدارة الذاكرة وعرض الدردشة ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. معالجة سؤال المستخدم ---
if prompt := st.chat_input("اسأل عن القوانين أو المبادرات الائتمانية..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("جاري مراجعة المراجع..."):
            try:
                # تقليل عدد النتائج (k=3) للحفاظ على الذاكرة وسرعة الرد
                docs = vectorstore.similarity_search(prompt, k=3)
                
                doc_context = "\n\n".join([f"[{d.metadata['source']} ص{d.metadata['page']}]: {d.page_content}" for d in docs])
                
                final_prompt = f"""
أنت مستشار قانوني تلتزم **فقط** بالمراجع المقدمة إليك. 
                
                قواعد هامة جداً:
                1. أجب باللغة العربية الفصحى فقط وبشكل مهني.
                2. التزم **حصرياً** بالبيانات الموجودة في [المراجع القانونية المتاحة] أدناه.
                3. إذا لم تجد الإجابة في المراجع، قل نصاً: "عذراً، هذه المعلومة غير متوفرة في المراجع القانونية المتاحة حالياً."
4. كن مرن بمعني الكلمة لو لها جمع او مفرد او مثنى الخ تصبح واحده
                5. لا تكرر النقاط.
6. كن مرنا في شكل الحروف حيث حرف ا هو نفسه أ و هو نفسه إ الخ..

                المراجع:
                {doc_context}
                
                السؤال: {prompt}
                الإجابة:"""

                response = llm.invoke([HumanMessage(content=final_prompt)])
                full_answer = response.content
                
                # إزالة أي وسوم تفكير إذا ظهرت
                full_answer = re.sub(r'<think>.*?</think>', '', full_answer, flags=re.DOTALL).strip()
                
                st.markdown(full_answer)
                
                sources = set([f"{d.metadata['source']} (ص{d.metadata['page']})" for d in docs])
                st.info("📍 **المصادر:**\n\n" + " | ".join(sources))
                
                st.session_state.messages.append({"role": "assistant", "content": full_answer})
                
                # تنظيف الذاكرة بعد كل سؤال
                gc.collect()

            except Exception as e:
                st.error(f"❌ حدث خطأ: {e}")
