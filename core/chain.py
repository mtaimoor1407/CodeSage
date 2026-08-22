import os
import json
import re
from typing import List
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_history_aware_retriever
from langchain_core.documents import Document
from core.prompts import CODE_RAG_PROMPT, REPHRASE_PROMPT, ARCHITECTURE_PROMPT
from models.schemas import CodeAnswer, CodeSource, ConfidenceLevel


def build_llm(provider: str = "groq") -> ChatGroq:
    if provider == "groq":
        return ChatGroq(
            model       = "openai/gpt-oss-120b",
            temperature = 0,
            max_tokens  = 2048,
            api_key     = os.getenv("GROQ_API_KEY")
        )
    raise ValueError(f"Unknown provider: {provider}")


def format_docs_for_context(docs: List[Document]) -> str:
    """Format retrieved code chunks into a clean context string."""
    pieces = []
    for i, doc in enumerate(docs):
        filepath      = doc.metadata.get("filepath", "unknown")
        language      = doc.metadata.get("language", "text")
        function_name = doc.metadata.get("function_name")
        start_line    = doc.metadata.get("start_line")

        header = f"[File {i+1}: {filepath}"
        if function_name:
            header += f" | Function: {function_name}"
        if start_line:
            header += f" | Line: {start_line}"
        header += f" | Language: {language}]"

        pieces.append(f"{header}\n{doc.page_content}")

    return "\n\n" + "─" * 60 + "\n\n".join(pieces)


def _extract_sources_from_docs(docs: List[Document]) -> List[CodeSource]:
    """Build source list from retrieved docs as fallback."""
    sources = []
    for doc in docs[:3]:
        source = CodeSource(
            filepath      = doc.metadata.get("filepath", "unknown"),
            filename      = doc.metadata.get("filename", "unknown"),
            language      = doc.metadata.get("language", "text"),
            function_name = doc.metadata.get("function_name"),
            start_line    = doc.metadata.get("start_line"),
            excerpt       = doc.page_content[:300] + "..."
                            if len(doc.page_content) > 300
                            else doc.page_content
        )
        sources.append(source)
    return sources


def parse_code_answer(
    raw_json      : dict,
    retrieved_docs: List[Document]
) -> CodeAnswer:
    """Parse LLM JSON output into a CodeAnswer object."""

    sources = []
    for src in raw_json.get("sources", []):
        source = CodeSource(
            filepath      = src.get("filepath", "unknown"),
            filename      = src.get("filename", "unknown"),
            language      = src.get("language", "text"),
            function_name = src.get("function_name"),
            start_line    = src.get("start_line"),
            excerpt       = src.get("excerpt", "")
        )
        sources.append(source)

    if not sources:
        sources = _extract_sources_from_docs(retrieved_docs)

    try:
        confidence = ConfidenceLevel(raw_json.get("confidence", "low"))
    except ValueError:
        confidence = ConfidenceLevel.LOW

    return CodeAnswer(
        answer       = raw_json.get("answer", "Unable to parse answer."),
        confidence   = confidence,
        sources      = sources,
        has_code     = raw_json.get("has_code", False),
        code_snippet = raw_json.get("code_snippet"),
        language     = raw_json.get("language"),
        warning      = raw_json.get("warning")
    )


class CodeSageChain:
    """Main RAG chain for CodeSage."""

    def __init__(self, retriever, provider: str = "groq"):
        self.retriever    = retriever
        self.llm          = build_llm(provider)
        self.chat_history : List = []

        self.history_retriever = create_history_aware_retriever(
            llm       = self.llm,
            retriever = self.retriever,
            prompt    = REPHRASE_PROMPT
        )

    def ask(
        self,
        question         : str,
        architecture_mode: bool = False
    ) -> CodeAnswer:
        """Ask a question about the codebase."""

        prompt = ARCHITECTURE_PROMPT if architecture_mode else CODE_RAG_PROMPT

        retrieved_docs = self.history_retriever.invoke({
            "input"        : question,
            "chat_history" : self.chat_history
        })

        context = format_docs_for_context(retrieved_docs)

        filled_prompt = prompt.invoke({
            "context"      : context,
            "question"     : question,
            "chat_history" : self.chat_history
        })

        response     = self.llm.invoke(filled_prompt)
        code_answer  = self._parse_response(response.content, retrieved_docs)

        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response.content))

        return code_answer

    def _parse_response(
        self,
        raw_content   : str,
        retrieved_docs: List[Document]
    ) -> CodeAnswer:
        """Multi-layer JSON parsing with fallbacks."""

        cleaned = raw_content.strip()

        # Strip markdown fences
        cleaned = re.sub(r'^```[a-zA-Z]*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```$',           '', cleaned)
        cleaned = cleaned.strip()

        # Layer 1 — direct parse
        try:
            return parse_code_answer(json.loads(cleaned), retrieved_docs)
        except json.JSONDecodeError:
            pass

        # Layer 2 — regex extract JSON
        try:
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                return parse_code_answer(
                    json.loads(match.group()), retrieved_docs
                )
        except Exception:
            pass

        # Layer 3 — fix truncated JSON
        try:
            partial       = cleaned
            open_braces   = partial.count('{') - partial.count('}')
            open_brackets = partial.count('[') - partial.count(']')
            open_quotes   = partial.count('"') % 2
            if open_quotes   : partial += '"'
            if open_brackets : partial += ']' * open_brackets
            if open_braces   : partial += '}' * open_braces
            return parse_code_answer(json.loads(partial), retrieved_docs)
        except Exception:
            pass

        # Layer 4 — raw fallback
        return CodeAnswer(
            answer       = cleaned,
            confidence   = ConfidenceLevel.LOW,
            sources      = _extract_sources_from_docs(retrieved_docs),
            has_code     = False,
            code_snippet = None,
            language     = None,
            warning      = "Parsing failed — showing raw response."
        )

    def clear_history(self):
        self.chat_history = []
        print("History cleared.")

    def get_history_length(self) -> int:
        return len(self.chat_history)