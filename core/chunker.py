import re
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


PYTHON_SEPARATORS = [
    "\nclass ",
    "\ndef ",
    "\n\tdef ",
    "\n    def ",
    "\n\n\n",
    "\n\n",
    "\n",
    " ",
    ""
]

JS_TS_SEPARATORS = [
    "\nfunction ",
    "\nconst ",
    "\nclass ",
    "\nexport function ",
    "\nexport const ",
    "\nexport default ",
    "\nasync function ",
    "\n\n\n",
    "\n\n",
    "\n",
    " ",
    ""
]

GENERIC_SEPARATORS = [
    "\n\n\n",
    "\n\n",
    "\n",
    " ",
    ""
]

LANGUAGE_SEPARATORS = {
    "python"        : PYTHON_SEPARATORS,
    "javascript"    : JS_TS_SEPARATORS,
    "typescript"    : JS_TS_SEPARATORS,
    "java"          : ["\npublic ", "\nprivate ", "\nprotected ", "\nclass ", "\n\n", "\n", " ", ""],
    "cpp"           : ["\nvoid ", "\nint ", "\nclass ", "\nstruct ", "\n\n", "\n", " ", ""],
    "go"            : ["\nfunc ", "\ntype ", "\nvar ", "\nconst ", "\n\n", "\n", " ", ""],
    "rust"          : ["\nfn ", "\npub fn ", "\nstruct ", "\nimpl ", "\n\n", "\n", " ", ""],
}


def get_separators(language: str) -> List[str]:
    """Return the best separators for a given language."""
    return LANGUAGE_SEPARATORS.get(language, GENERIC_SEPARATORS)


def extract_function_name(
    content  : str,
    language : str
) -> Optional[str]:
    """
    Try to extract the primary function or class name
    from a code chunk.
    """

    patterns = {
        "python": [
            r'def\s+(\w+)\s*\(',
            r'class\s+(\w+)\s*[:\(]',
        ],
        "javascript": [
            r'function\s+(\w+)\s*\(',
            r'const\s+(\w+)\s*=\s*(?:async\s*)?\(',
            r'class\s+(\w+)\s*[{\(]',
            r'(?:export\s+)?(?:default\s+)?function\s+(\w+)',
        ],
        "typescript": [
            r'function\s+(\w+)\s*[<\(]',
            r'const\s+(\w+)\s*=\s*(?:async\s*)?\(',
            r'class\s+(\w+)\s*[{\(<]',
            r'interface\s+(\w+)\s*[{\(]',
        ],
        "java": [
            r'(?:public|private|protected)\s+\w+\s+(\w+)\s*\(',
            r'class\s+(\w+)\s*[{\(]',
        ],
        "go": [
            r'func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(',
        ],
        "rust": [
            r'fn\s+(\w+)\s*[<\(]',
            r'struct\s+(\w+)\s*[{\(]',
        ]
    }

    lang_patterns = patterns.get(language, [])
    for pattern in lang_patterns:
        match = re.search(pattern, content)
        if match:
            return match.group(1)

    return None


def estimate_start_line(
    full_content : str,
    chunk_content: str
) -> Optional[int]:
    """Estimate the starting line number of a chunk in the full file."""
    try:
        idx = full_content.find(chunk_content[:100])
        if idx == -1:
            return None
        return full_content[:idx].count('\n') + 1
    except Exception:
        return None


def chunk_documents(
    documents   : List[Document],
    chunk_size  : int = 1000,
    chunk_overlap: int = 100
) -> List[Document]:
    """
    Split code documents into chunks using language-aware separators.
    Enriches each chunk with function name and line number metadata.
    """

    all_chunks : List[Document] = []

    for doc in documents:
        language    = doc.metadata.get("language", "text")
        filepath    = doc.metadata.get("filepath", "unknown")
        filename    = doc.metadata.get("filename", "unknown")
        full_content= doc.page_content

        separators = get_separators(language)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size      = chunk_size,
            chunk_overlap   = chunk_overlap,
            separators      = separators,
            length_function = len,
        )

        chunks = splitter.split_documents([doc])

        for i, chunk in enumerate(chunks):
            function_name = extract_function_name(
                chunk.page_content,
                language
            )
            start_line = estimate_start_line(
                full_content,
                chunk.page_content
            )

            chunk.metadata.update({
                "chunk_index"   : i,
                "function_name" : function_name,
                "start_line"    : start_line,
                "chunk_type"    : "code",
                "char_count"    : len(chunk.page_content),
            })

            all_chunks.append(chunk)

    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


def get_chunk_stats(chunks: List[Document]) -> dict:
    """Return statistics about the created chunks."""

    lang_counts = {}
    for chunk in chunks:
        lang = chunk.metadata.get("language", "unknown")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    avg_size = (
        sum(len(c.page_content) for c in chunks) // len(chunks)
        if chunks else 0
    )

    return {
        "total_chunks"      : len(chunks),
        "avg_chunk_size"    : avg_size,
        "chunks_by_language": lang_counts
    }