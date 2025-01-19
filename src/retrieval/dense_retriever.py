from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(self, model: str = "text-embedding-3-small",
                 chunk_size: int = 512, chunk_overlap: int = 64):
        self.embeddings = OpenAIEmbeddings(model=model)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.vectorstore: Optional[FAISS] = None

    def index(self, documents: List[Document]) -> int:
        chunks = self.splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        logger.info(f"Indexed {len(chunks)} chunks from {len(documents)} documents")
        return len(chunks)

    def retrieve(self, query: str, k: int = 8) -> List[Tuple[Document, float]]:
        if not self.vectorstore:
            raise ValueError("Index documents first")
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def save(self, path: str):
        if self.vectorstore:
            self.vectorstore.save_local(path)

    def load(self, path: str):
        self.vectorstore = FAISS.load_local(path, self.embeddings,
                                            allow_dangerous_deserialization=True)
