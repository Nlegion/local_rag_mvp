# check_vector_db.py
from rag_core import RAGSystem


def check_vector_db_contents():
    rag = RAGSystem()
    rag.load_vector_store()

    # Получаем все документы из коллекции
    collection = rag.vectorstore._collection
    results = collection.get()

    print(f"Всего документов в базе: {len(results['ids'])}")

    for i, (metadata, doc_content) in enumerate(zip(results['metadatas'], results['documents'])):
        print(f"\n--- Документ {i + 1} ---")
        print(f"Источник: {metadata.get('source', 'Unknown')}")
        print(f"Длина контента: {len(doc_content)} символов")
        print(f"Первые 200 символов: {doc_content[:200]}...")


if __name__ == "__main__":
    check_vector_db_contents()