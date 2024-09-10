import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from uuid import uuid4
from langchain_core.documents import Document



class VectorDBHandler:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    # vector_store = FAISS(
    #     embedding_function=embeddings,
    #     index=index,
    #     docstore=InMemoryDocstore(),
    #     index_to_docstore_id={},
    # )
    
    vector_store = FAISS.load_local(
        "vector-db", embeddings, allow_dangerous_deserialization=True
    )
        
    @classmethod
    def mock_data(cls):
        document_1 = Document(
        page_content="Artificial intelligence is transforming the medical field with new diagnostic and treatment options.",
        )

        document_2 = Document(
            page_content="Machine learning is increasingly used for analyzing large data sets in finance.",
        )

        document_3 = Document(
            page_content="Recent advancements in natural language processing include better context understanding.",
        )

        documents = [
            document_1,
            document_2,
            document_3
        ]
        uuids = [str(uuid4()) for _ in range(len(documents))]
        
        cls.vector_store.add_documents(documents=documents, ids=uuids)
        cls.vector_store.save_local(folder_path="vector-db")
    
    @classmethod
    def search_docs(cls, query):
        return cls.vector_store.similarity_search(query, k=1)

