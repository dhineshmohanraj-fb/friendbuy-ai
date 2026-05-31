"""Chunk LangChain Documents using language-aware or generic text splitters."""

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from config import LANGUAGE_MAP, settings


def _get_splitter(file_type: str) -> RecursiveCharacterTextSplitter:
    """Return a language-aware splitter when available, else a generic one."""
    lang_key = LANGUAGE_MAP.get(file_type)
    if lang_key:
        try:
            lang = Language(lang_key)
            return RecursiveCharacterTextSplitter.from_language(
                language=lang,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        except (ValueError, KeyError):
            pass

    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        add_start_index=True,
    )


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split a list of Documents into chunks.

    Groups documents by file type so each group uses the optimal splitter.
    Metadata is preserved and a ``chunk_index`` field is added.
    """
    # Group by file type to reuse splitter instances
    by_type: dict[str, list[Document]] = {}
    for doc in documents:
        ft = doc.metadata.get("file_type", "")
        by_type.setdefault(ft, []).append(doc)

    chunks: list[Document] = []
    for file_type, docs in by_type.items():
        splitter = _get_splitter(file_type)
        split = splitter.split_documents(docs)
        # Add chunk index within each source file
        file_counters: dict[str, int] = {}
        for chunk in split:
            fp = chunk.metadata.get("file_path", "")
            idx = file_counters.get(fp, 0)
            chunk.metadata["chunk_index"] = idx
            file_counters[fp] = idx + 1
        chunks.extend(split)

    return chunks
