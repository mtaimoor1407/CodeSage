import os
import zipfile
import shutil
from typing import List, Tuple, Dict
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {
    ".py"   : "python",
    ".js"   : "javascript",
    ".ts"   : "typescript",
    ".jsx"  : "javascript",
    ".tsx"  : "typescript",
    ".java" : "java",
    ".cpp"  : "cpp",
    ".c"    : "c",
    ".cs"   : "csharp",
    ".go"   : "go",
    ".rs"   : "rust",
    ".php"  : "php",
    ".rb"   : "ruby",
    ".swift": "swift",
    ".kt"   : "kotlin",
    ".md"   : "markdown",
    ".txt"  : "text",
    ".json" : "json",
    ".yaml" : "yaml",
    ".yml"  : "yaml",
    ".toml" : "toml",
    ".env"  : "text",
    ".sh"   : "bash",
    ".html" : "html",
    ".css"  : "css",
    ".sql"  : "sql",
}

SKIP_DIRS = {
    "__pycache__", ".git", "node_modules", ".venv", "venv",
    "env", ".env", "dist", "build", ".idea", ".vscode",
    "*.egg-info", ".pytest_cache", "coverage", ".nyc_output"
}

SKIP_FILES = {
    ".DS_Store", "Thumbs.db", "package-lock.json",
    "yarn.lock", "poetry.lock", "Pipfile.lock"
}

MAX_FILE_SIZE_KB = 2048

PRIORITY_DIRS = {
    "src", "app", "lib", "core", "api",
    "routes", "controllers", "models",
    "services", "utils", "helpers"
}

def load_code_files(
    repo_path       : str,
    focus_dirs      : list = None
) -> tuple:
    """
    focus_dirs: if provided, only index files inside these folders.
    Example: ["src", "app", "lib"]
    """
    for root, dirs, files in os.walk(repo_path):

        # Filter to focus directories if specified
        if focus_dirs:
            rel_root = os.path.relpath(root, repo_path)
            top_level = rel_root.split(os.sep)[0]
            if rel_root != "." and top_level not in focus_dirs:
                dirs.clear()
                continue

def extract_zip(
    zip_path    : str,
    extract_dir : str = "extracted_repos"
) -> str:
    """
    Extract a ZIP file and return the path to extracted contents.
    Cleans up any previous extraction first.
    """

    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    entries = os.listdir(extract_dir)
    if len(entries) == 1 and os.path.isdir(
        os.path.join(extract_dir, entries[0])
    ):
        return os.path.join(extract_dir, entries[0])

    return extract_dir


def should_skip(path: str) -> bool:
    """Check if a file or directory should be skipped."""

    parts = path.replace("\\", "/").split("/")
    for part in parts:
        if part in SKIP_DIRS:
            return True

    filename = os.path.basename(path)
    if filename in SKIP_FILES:
        return True

    if os.path.isfile(path):
        size_kb = os.path.getsize(path) / 1024
        if size_kb > MAX_FILE_SIZE_KB:
            return True

    return False


def load_code_files(repo_path: str) -> Tuple[List[Document], Dict]:
    """
    Walk through the repository and load all supported code files.
    Returns (documents, stats).
    """

    documents   : List[Document] = []
    stats       : Dict = {
        "total_files"   : 0,
        "skipped_files" : 0,
        "languages"     : set(),
        "file_tree"     : []
    }

    for root, dirs, files in os.walk(repo_path):

        dirs[:] = [
            d for d in dirs
            if d not in SKIP_DIRS and not d.startswith('.')
        ]

        for filename in sorted(files):
            filepath = os.path.join(root, filename)

            if should_skip(filepath):
                stats["skipped_files"] += 1
                continue

            ext = os.path.splitext(filename)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                stats["skipped_files"] += 1
                continue

            language = SUPPORTED_EXTENSIONS[ext]

            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                if not content.strip():
                    continue

                rel_path = os.path.relpath(filepath, repo_path)
                rel_path = rel_path.replace("\\", "/")

                doc = Document(
                    page_content = content,
                    metadata     = {
                        "filepath"  : rel_path,
                        "filename"  : filename,
                        "language"  : language,
                        "extension" : ext,
                        "file_size" : len(content),
                        "repo_path" : repo_path
                    }
                )

                documents.append(doc)
                stats["total_files"]    += 1
                stats["languages"].add(language)
                stats["file_tree"].append(rel_path)

            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                stats["skipped_files"] += 1
                continue

    stats["languages"] = sorted(list(stats["languages"]))
    stats["file_tree"] = sorted(stats["file_tree"])

    print(f"Files loaded    : {stats['total_files']}")
    print(f"Files skipped   : {stats['skipped_files']}")
    print(f"Languages found : {stats['languages']}")

    return documents, stats


def save_uploaded_zip(uploaded_file, save_dir: str = "uploaded_repos") -> str:
    """Save Streamlit uploaded ZIP file to disk."""

    os.makedirs(save_dir, exist_ok=True)
    zip_path = os.path.join(save_dir, uploaded_file.name)

    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    print(f"Saved ZIP: {zip_path}")
    return zip_path