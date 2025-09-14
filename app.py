# app.py
import streamlit as st
from rag_core import RAGSystem
import time
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Локальная RAG-система",
    page_icon="🤖",
    layout="wide"
)

# Инициализация состояния сессии
if "history" not in st.session_state:
    st.session_state.history = []
if "rag_system" not in st.session_state:
    try:
        logger.info("Инициализация RAG системы")
        rag = RAGSystem()
        rag.load_vector_store()
        st.session_state.rag_system = rag
        logger.info("RAG система успешно инициализирована")
    except Exception as e:
        logger.error(f"Ошибка инициализации RAG системы: {str(e)}")
        st.error(f"Ошибка инициализации системы: {str(e)}")
        st.session_state.rag_system = None


def main():
    st.title("🤖 Локальная RAG-система для внутренних процессов")

    # Основная область для диалога
    st.header("Диалог с помощником")

    # Отображение истории диалога
    for i, message in enumerate(st.session_state.history):
        if message["role"] == "user":
            st.markdown(f"**Вы ({message['time']}):** {message['content']}")
        else:
            st.markdown(f"**Помощник ({message['time']}):**")
            st.markdown(message["content"])
            st.markdown("---")

    # Форма для нового запроса
    with st.form(key="query_form"):
        query = st.text_input("Ваш вопрос:", placeholder="Введите ваш вопрос здесь...", key="query_input")
        submit_button = st.form_submit_button(label="Отправить")

    if submit_button and query:
        if st.session_state.rag_system is None:
            st.error("Система не инициализирована. Пожалуйста, проверьте логи.")
            return

        try:
            # Добавляем вопрос в историю
            st.session_state.history.append({
                "role": "user",
                "content": query,
                "time": time.strftime("%H:%M:%S")
            })

            with st.spinner("Обработка запроса..."):
                start_time = time.time()
                answer = st.session_state.rag_system.process_query(query)
                end_time = time.time()

                # Добавляем ответ в историю
                st.session_state.history.append({
                    "role": "assistant",
                    "content": answer,
                    "time": time.strftime("%H:%M:%S"),
                    "response_time": f"{round(end_time - start_time, 2)} секунды"
                })

            # Обновляем интерфейс
            st.rerun()

        except Exception as e:
            logger.error(f"Ошибка при обработке запроса: {str(e)}")
            st.error(f"Произошла ошибка при обработке вашего запроса: {str(e)}")


if __name__ == "__main__":
    logger.info("Запуск Streamlit приложения")
    main()