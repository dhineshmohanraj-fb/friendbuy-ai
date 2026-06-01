"""
Chunk LangChain Documents into smaller pieces for embedding.

CP1 upgrade: for Python, JS, TS, JSX, and TSX files the splitter uses
tree-sitter to find natural AST boundaries (function / class definitions)
instead of blindly cutting at N characters.  Each chunk carries rich
metadata — symbol name, type, start/end line, docstring — which improves
both retrieval relevance and the graph data we store in CP2.

For all other file types (YAML, JSON, Markdown, etc.) the existing
RecursiveCharacterTextSplitter is used unchanged.

If tree-sitter is not installed the module silently falls back to the
character splitter for every file type.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from rich.console import Console

from config import LANGUAGE_MAP, get_settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Extensions that get AST-aware splitting when tree-sitter is available
_AST_EXTENSIONS: frozenset[str] = frozenset({".py", ".js", ".jsx", ".ts", ".tsx"})

# Warn only once if tree-sitter is missing
_TS_WARNED: bool = False


# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------

def _tree_sitter_available() -> bool:
    """Return True if tree-sitter core library is importable."""
    global _TS_WARNED
    try:
        import tree_sitter  # noqa: F401
        return True
    except ImportError:
        if not _TS_WARNED:
            _TS_WARNED = True
            Console().print(
                "[yellow]tree-sitter not installed — using character-based chunking.[/yellow]\n"
                "For AST-aware splitting run:\n"
                "  [bold]pip install tree-sitter tree-sitter-python "
                "tree-sitter-javascript tree-sitter-typescript[/bold]"
            )
        return False


# ---------------------------------------------------------------------------
# Character-based fallback splitter
# ---------------------------------------------------------------------------

def _char_splitter(file_type: str) -> RecursiveCharacterTextSplitter:
    """Return a language-aware or generic character splitter."""
    settings = get_settings()
    lang_key = LANGUAGE_MAP.get(file_type)
    if lang_key:
        try:
            return RecursiveCharacterTextSplitter.from_language(
                language=Language(lang_key),
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


def _char_split_doc(doc: Document, file_type: str) -> list[Document]:
    """Split *doc* using the character splitter, preserving all metadata."""
    splitter = _char_splitter(file_type)
    return splitter.split_documents([doc])


# ---------------------------------------------------------------------------
# AST-aware splitting
# ---------------------------------------------------------------------------

def _ast_split_doc(doc: Document, file_type: str) -> list[Document]:
    """
    Split a source file at AST symbol boundaries.

    Falls back to character splitting if:
    - The grammar package for this language is not installed.
    - The file has no extractable top-level symbols.
    - Any tree-sitter exception is raised.
    """
    from indexer.ast_parser import (
        extract_js_symbols,
        extract_python_symbols,
        extract_ts_symbols,
    )

    settings = get_settings()
    content = doc.page_content

    # Dispatch to the right extractor
    if file_type == ".py":
        symbols = extract_python_symbols(content)
    elif file_type in (".js", ".jsx"):
        symbols = extract_js_symbols(content, tsx=False)
    elif file_type == ".tsx":
        symbols = extract_ts_symbols(content, tsx=True)
    elif file_type in (".ts",):
        symbols = extract_ts_symbols(content, tsx=False)
    else:
        symbols = []

    if not symbols:
        # No top-level symbols found (e.g. file is only imports/constants)
        # Fall back to character splitter so the file is still indexed
        return _char_split_doc(doc, file_type)

    chunks: list[Document] = []
    max_single = settings.chunk_size * 2   # symbols up to this size stay whole

    for sym in symbols:
        sym_text: str = sym["text"]
        sym_meta: dict = {
            **doc.metadata,
            "symbol_name": sym["name"],
            "symbol_type": sym["type"],
            "start_line":  sym["start_line"],
            "end_line":    sym["end_line"],
            "docstring":   sym["docstring"],
        }

        if len(sym_text) <= max_single:
            # Keep the symbol as a single chunk
            chunks.append(Document(page_content=sym_text, metadata=sym_meta))
        else:
            # Symbol is very large — sub-split it, but keep the symbol metadata
            # so retrieval knows which function/class the chunk belongs to.
            sub_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            for sub_idx, sub_text in enumerate(sub_splitter.split_text(sym_text)):
                chunks.append(Document(
                    page_content=sub_text,
                    metadata={**sym_meta, "sub_chunk": True, "_sub_idx": sub_idx},
                ))

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split every Document in *documents* into embedding-ready chunks.

    - Python / JS / TS / JSX / TSX → tree-sitter AST boundaries (with
      character-splitter fallback if tree-sitter is unavailable or the
      file has no top-level symbols).
    - All other file types → language-aware RecursiveCharacterTextSplitter.

    Every output chunk has ``chunk_index`` set (sequential within its
    source file) and inherits all metadata from the parent Document.
    Symbol chunks also carry ``symbol_name``, ``symbol_type``,
    ``start_line``, ``end_line``, and ``docstring``.
    """
    ts_ok = _tree_sitter_available()

    all_chunks: list[Document] = []
    file_counters: dict[str, int] = {}  # file_path → next chunk_index

    for doc in documents:
        ft = doc.metadata.get("file_type", "")

        # Choose splitting strategy
        if ts_ok and ft in _AST_EXTENSIONS:
            raw_chunks = _ast_split_doc(doc, ft)
        else:
            raw_chunks = _char_split_doc(doc, ft)

        # Assign sequential chunk_index within this file
        fp = doc.metadata.get("file_path", "")
        base = file_counters.get(fp, 0)
        for i, chunk in enumerate(raw_chunks):
            chunk.metadata["chunk_index"] = base + i
        file_counters[fp] = base + len(raw_chunks)

        all_chunks.extend(raw_chunks)

    return all_chunks
