# app.py
import streamlit as st
from rag_core import RAGSystem
import time

st.set_page_config(
    page_title="Локальная RAG-система",
    page_icon="🤖",
    layout="centered"
)

@st.cache_resource
def init_rag_system():
    rag = RAGSystem()
    rag.load_vector_store()
    return rag

def main():
    st.title("🤖 Локальная RAG-система для внутренних процессов")
    st.markdown("Задайте вопрос о процессах компании (например, 'Как оформить закупку на 30 тысяч?')")

    rag = init_rag_system()
    query = st.text_input("Ваш вопрос:", placeholder="Введите ваш вопрос здесь...")

    if query:
        with st.spinner("Ищу информацию и генерирую ответ..."):
            start_time = time.time()
            answer = rag.generate_answer(query)
            end_time = time.time()

        st.success(f"Ответ получен за {round(end_time - start_time, 2)} секунды")
        st.markdown("### Ответ:")
        st.write(answer)

        with st.expander("Показать детали поиска"):
            relevant_docs = rag.get_relevant_context(query)
            for i, doc in enumerate(relevant_docs):
                st.markdown(f"**Релевантный фрагмент #{i+1} (из {doc.metadata['source']}):**")
                st.info(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

if __name__ == "__main__":
    main()