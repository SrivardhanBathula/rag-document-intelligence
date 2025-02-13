"""
Multi-Format Document Loader
Supports PDF, DOCX, TXT, HTML, and web URLs with metadata extraction.
"""

import logging
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class DocumentPage:
    content: str
    metadata: dict = field(default_factory=dict)
    # metadata keys: source, page, section, doc_type, char_count


class DocumentLoader:
    """
    Unified loader for PDF, DOCX, TXT, HTML files and web URLs.
    Extracts clean text with rich metadata for RAG indexing.
    """

    def load(self, source: str) -> list[DocumentPage]:
        """Auto-detect source type and dispatch to correct loader."""
        if source.startswith("http://") or source.startswith("https://"):
            return self._load_url(source)

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        ext = path.suffix.lower()
        dispatch = {
            ".pdf":  self._load_pdf,
            ".docx": self._load_docx,
            ".txt":  self._load_txt,
            ".html": self._load_html,
            ".htm":  self._load_html,
            ".md":   self._load_txt,
        }

        loader_fn = dispatch.get(ext)
        if loader_fn is None:
            raise ValueError(f"Unsupported file type: {ext}")

        logger.info(f"Loading {ext.upper()} document: {path.name}")
        pages = loader_fn(source)
        logger.info(f"  Loaded {len(pages)} page(s) from {path.name}")
        return pages

    def load_directory(self, directory: str, glob: str = "**/*") -> list[DocumentPage]:
        """Recursively load all supported documents from a directory."""
        supported = {".pdf", ".docx", ".txt", ".html", ".htm", ".md"}
        all_pages = []
        for path in Path(directory).glob(glob):
            if path.suffix.lower() in supported:
                try:
                    all_pages.extend(self.load(str(path)))
                except Exception as e:
                    logger.warning(f"Skipping {path.name}: {e}")
        logger.info(f"Loaded {len(all_pages)} pages from directory: {directory}")
        return all_pages

    def _load_pdf(self, path: str) -> list[DocumentPage]:
        import fitz  # PyMuPDF
        pages = []
        doc = fitz.open(path)
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if not text:
                continue
            pages.append(DocumentPage(
                content=text,
                metadata={
                    "source": Path(path).name,
                    "page": i + 1,
                    "total_pages": len(doc),
                    "doc_type": "pdf",
                    "char_count": len(text),
                },
            ))
        doc.close()
        return pages

    def _load_docx(self, path: str) -> list[DocumentPage]:
        from docx import Document
        doc = Document(path)
        pages, current_section, chunks = [], "Introduction", []

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            # Treat Heading styles as section markers
            if para.style.name.startswith("Heading"):
                if chunks:
                    pages.append(DocumentPage(
                        content="\n".join(chunks),
                        metadata={"source": Path(path).name, "section": current_section, "doc_type": "docx"},
                    ))
                    chunks = []
                current_section = text
            else:
                chunks.append(text)

        if chunks:
            pages.append(DocumentPage(
                content="\n".join(chunks),
                metadata={"source": Path(path).name, "section": current_section, "doc_type": "docx"},
            ))
        return pages

    def _load_txt(self, path: str) -> list[DocumentPage]:
        text = Path(path).read_text(encoding="utf-8", errors="replace").strip()
        return [DocumentPage(
            content=text,
            metadata={"source": Path(path).name, "page": 1, "doc_type": "txt", "char_count": len(text)},
        )]

    def _load_html(self, path: str) -> list[DocumentPage]:
        from bs4 import BeautifulSoup
        html = Path(path).read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n").strip()
        title = soup.title.string if soup.title else Path(path).name
        return [DocumentPage(
            content=text,
            metadata={"source": title, "page": 1, "doc_type": "html", "char_count": len(text)},
        )]

    def _load_url(self, url: str) -> list[DocumentPage]:
        import requests
        from bs4 import BeautifulSoup
        resp = requests.get(url, timeout=15, headers={"User-Agent": "RAG-DocumentLoader/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n").strip()
        title = soup.title.string if soup.title else url
        return [DocumentPage(
            content=text,
            metadata={"source": url, "title": title, "page": 1, "doc_type": "web"},
        )]
