from pygments import highlight
from pygments.lexers import get_lexer_by_name, TextLexer
from pygments.formatters import HtmlFormatter
from models.schemas import CodeAnswer, ConfidenceLevel
from typing import Optional


def get_confidence_badge(confidence: ConfidenceLevel) -> str:
    badges = {
        ConfidenceLevel.HIGH   : "🟢 High",
        ConfidenceLevel.MEDIUM : "🟡 Medium",
        ConfidenceLevel.LOW    : "🔴 Low",
        ConfidenceLevel.NONE   : "⚫ Not Found"
    }
    return badges.get(confidence, "⚫ Unknown")


def highlight_code(code: str, language: str = "text") -> str:
    """Apply syntax highlighting to a code snippet."""
    try:
        lexer     = get_lexer_by_name(language, stripall=True)
    except Exception:
        lexer     = TextLexer()

    formatter  = HtmlFormatter(
        style     = "monokai",
        noclasses = True,
        nowrap    = False
    )
    return highlight(code, lexer, formatter)


def format_file_tree(file_paths: list) -> str:
    """Format a list of file paths into a readable tree."""
    if not file_paths:
        return "No files"

    lines = []
    for path in sorted(file_paths):
        parts  = path.split("/")
        indent = "  " * (len(parts) - 1)
        icon   = get_file_icon(parts[-1])
        lines.append(f"{indent}{icon} {parts[-1]}")

    return "\n".join(lines)


def get_file_icon(filename: str) -> str:
    """Return an emoji icon for a file based on its extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    icons = {
        "py"    : "🐍",
        "js"    : "🟨",
        "ts"    : "🔷",
        "jsx"   : "⚛️",
        "tsx"   : "⚛️",
        "java"  : "☕",
        "go"    : "🐹",
        "rs"    : "🦀",
        "md"    : "📝",
        "json"  : "📋",
        "yaml"  : "⚙️",
        "yml"   : "⚙️",
        "toml"  : "⚙️",
        "html"  : "🌐",
        "css"   : "🎨",
        "sql"   : "🗄️",
        "sh"    : "💻",
    }
    return icons.get(ext, "📄")


def format_answer_for_display(code_answer: CodeAnswer) -> dict:
    """Convert CodeAnswer into display-ready dict for Streamlit."""

    sources_display = []
    for src in code_answer.sources:
        label = f"{get_file_icon(src.filename)} {src.filepath}"
        if src.function_name:
            label += f" → {src.function_name}()"
        if src.start_line:
            label += f" (line {src.start_line})"

        sources_display.append({
            "label"    : label,
            "excerpt"  : src.excerpt,
            "language" : src.language
        })

    warnings = []
    if code_answer.warning:
        warnings.append(f"📌 {code_answer.warning}")

    highlighted_snippet = None
    if code_answer.has_code and code_answer.code_snippet:
        highlighted_snippet = highlight_code(
            code_answer.code_snippet,
            code_answer.language or "text"
        )

    return {
        "answer"             : code_answer.answer,
        "confidence"         : get_confidence_badge(code_answer.confidence),
        "sources"            : sources_display,
        "warnings"           : warnings,
        "has_code"           : code_answer.has_code,
        "highlighted_snippet": highlighted_snippet,
        "language"           : code_answer.language
    }