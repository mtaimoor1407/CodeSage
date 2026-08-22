import warnings
import os
import logging
import gc
import time

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"]           = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]  = "1"

import streamlit as st


def load_secrets():
    try:
        if "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        if "HF_TOKEN" in st.secrets:
            os.environ["HF_TOKEN"]              = st.secrets["HF_TOKEN"]
            os.environ["HUGGINGFACE_HUB_TOKEN"] = st.secrets["HF_TOKEN"]
    except Exception:
        pass
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

load_secrets()

from core.extractor   import save_uploaded_zip, extract_zip, load_code_files
from core.chunker     import chunk_documents, get_chunk_stats
from core.vectorstore import (
    get_embedding_model,
    build_vectorstore,
    close_vectorstore
)
from core.retriever   import build_retriever
from core.chain       import CodeSageChain
from utils.helpers    import format_answer_for_display, format_file_tree
from models.schemas   import RepoInfo

st.set_page_config(
    page_title = "CodeSage",
    page_icon  = "🧠",
    layout     = "wide"
)


# ──────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────

def initialize_session_state():
    defaults = {
        "chain"           : None,
        "repo_info"       : None,
        "all_chunks"      : [],
        "chat_history"    : [],
        "vectorstore"     : None,
        "embeddings"      : None,
        "processing_done" : False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ──────────────────────────────────────────────
# Processing Pipeline
# ──────────────────────────────────────────────

def process_uploaded_zip(uploaded_zip) -> bool:
    try:

        # Step 0 — close existing store to release Windows file locks
        if st.session_state.vectorstore is not None:
            close_vectorstore(st.session_state.vectorstore)
            st.session_state.vectorstore = None
            st.session_state.chain       = None
            gc.collect()
            time.sleep(0.5)

        # Step 1 — save and extract ZIP
        with st.spinner("📦 Extracting repository..."):
            zip_path  = save_uploaded_zip(uploaded_zip)
            repo_path = extract_zip(zip_path)
            repo_name = uploaded_zip.name.replace(".zip", "")
            st.toast(f"✅ Repository extracted", icon="📦")

        # Step 2 — load code files
        with st.spinner("📂 Reading code files..."):
            documents, stats = load_code_files(repo_path)
            if not documents:
                st.error(
                    "No supported code files found in the ZIP. "
                    "Make sure your ZIP contains source code files "
                    "(.py, .js, .ts, .java, etc.)"
                )
                return False
            st.toast(
                f"✅ {stats['total_files']} files loaded "
                f"({stats['skipped_files']} skipped)",
                icon="📂"
            )

        # Step 3 — chunk documents
        with st.spinner(f"✂️ Chunking {len(documents)} files intelligently..."):
            chunks      = chunk_documents(
                documents,
                chunk_size    = 1500,
                chunk_overlap = 150
            )
            chunk_stats = get_chunk_stats(chunks)
            st.session_state.all_chunks = chunks
            st.toast(
                f"✅ {chunk_stats['total_chunks']} chunks created",
                icon="✂️"
            )

        # Step 4 — load embedding model (only once per session)
        if st.session_state.embeddings is None:
            with st.spinner("🧠 Loading embedding model (first time only)..."):
                st.session_state.embeddings = get_embedding_model()
                st.toast("✅ Embedding model ready", icon="🧠")

        # Step 5 — build vector store
        with st.spinner(f"🗄️ Indexing {len(chunks)} chunks into vector store..."):
            vectorstore = build_vectorstore(
                chunks     = chunks,
                embeddings = st.session_state.embeddings,
                reset      = True
            )
            st.session_state.vectorstore = vectorstore
            st.toast("✅ Vector store ready", icon="🗄️")

        # Step 6 — build retriever and chain
        with st.spinner("🔗 Setting up CodeSage intelligence..."):
            retriever = build_retriever(
                vectorstore   = vectorstore,
                all_chunks    = chunks,
                fetch_k       = 20,
                final_k       = 6,
                use_reranking = True
            )
            st.session_state.chain = CodeSageChain(
                retriever = retriever,
                provider  = "groq"
            )
            st.toast("✅ CodeSage ready", icon="🔗")

        # Step 7 — store repo metadata
        st.session_state.repo_info = RepoInfo(
            repo_name    = repo_name,
            total_files  = stats["total_files"],
            total_chunks = chunk_stats["total_chunks"],
            languages    = stats["languages"],
            file_tree    = stats["file_tree"]
        )
        st.session_state.processing_done = True
        return True

    except Exception as e:
        st.error(f"❌ Error processing repository: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False


# ──────────────────────────────────────────────
# UI — Sidebar
# ──────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:

        st.title("🧠 CodeSage")
        st.caption("Chat with any codebase in plain English.")
        st.divider()

        # Upload section
        st.subheader("📦 Upload Repository")
        st.caption(
            "ZIP your project folder and upload it. "
            "Supports up to 500MB."
        )

        uploaded_zip = st.file_uploader(
            label = "Upload a ZIP file",
            type  = ["zip"],
            help  = "Zip your entire project folder and upload here"
        )

        if uploaded_zip:
            file_size_mb = uploaded_zip.size / (1024 * 1024)
            st.caption(f"📁 File size: {file_size_mb:.1f} MB")

            if st.button(
                "🚀 Index Codebase",
                type             = "primary",
                use_container_width = True
            ):
                success = process_uploaded_zip(uploaded_zip)
                if success:
                    repo = st.session_state.repo_info
                    st.success(
                        f"✅ {repo.repo_name} indexed successfully!\n"
                        f"{repo.total_files} files · "
                        f"{repo.total_chunks} chunks"
                    )
                    st.session_state.chat_history = []
                    if st.session_state.chain:
                        st.session_state.chain.clear_history()
                    st.rerun()

        # Repository info
        if st.session_state.repo_info:
            repo = st.session_state.repo_info
            st.divider()
            st.subheader(f"📁 {repo.repo_name}")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Files",  repo.total_files)
            with col2:
                st.metric("Chunks", repo.total_chunks)

            if repo.languages:
                st.markdown("**Languages detected:**")
                lang_icons = {
                    "python"    : "🐍",
                    "javascript": "🟨",
                    "typescript": "🔷",
                    "java"      : "☕",
                    "go"        : "🐹",
                    "rust"      : "🦀",
                    "cpp"       : "⚙️",
                    "csharp"    : "💜",
                    "markdown"  : "📝",
                    "json"      : "📋",
                    "yaml"      : "⚙️",
                    "html"      : "🌐",
                    "css"       : "🎨",
                    "sql"       : "🗄️",
                }
                for lang in repo.languages:
                    icon = lang_icons.get(lang, "📄")
                    st.markdown(f"{icon} {lang.capitalize()}")

            with st.expander("📂 File Tree", expanded=False):
                tree = format_file_tree(repo.file_tree)
                st.code(tree, language=None)

        # Chunk language breakdown
        if st.session_state.all_chunks:
            chunks = st.session_state.all_chunks
            if chunks:
                st.divider()
                with st.expander("📊 Chunk Breakdown", expanded=False):
                    stats = get_chunk_stats(chunks)
                    for lang, count in sorted(
                        stats["chunks_by_language"].items(),
                        key     = lambda x: x[1],
                        reverse = True
                    ):
                        st.markdown(f"**{lang}:** {count} chunks")

        # Clear conversation
        if st.session_state.processing_done:
            st.divider()
            if st.button(
                "🗑️ Clear Conversation",
                use_container_width = True
            ):
                st.session_state.chat_history = []
                if st.session_state.chain:
                    st.session_state.chain.clear_history()
                st.rerun()

            if st.button(
                "🔄 Upload New Repository",
                use_container_width = True
            ):
                st.session_state.processing_done = False
                st.session_state.repo_info       = None
                st.session_state.chat_history    = []
                st.session_state.all_chunks      = []
                if st.session_state.chain:
                    st.session_state.chain.clear_history()
                if st.session_state.vectorstore:
                    close_vectorstore(st.session_state.vectorstore)
                    st.session_state.vectorstore = None
                st.rerun()

        st.divider()
        st.caption(
            "Supported: Python · JavaScript · TypeScript · "
            "Java · Go · Rust · C++ · C# · PHP · Ruby · "
            "Swift · Kotlin · HTML · CSS · SQL · and more."
        )


# ──────────────────────────────────────────────
# UI — Answer Card
# ──────────────────────────────────────────────

def render_answer(display_data: dict):
    """Render a structured code answer card."""

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### 💡 Answer")
    with col2:
        st.markdown(f"**{display_data['confidence']}**")

    st.write(display_data["answer"])

    if display_data["warnings"]:
        for w in display_data["warnings"]:
            st.warning(w)

    if display_data["has_code"] and display_data["highlighted_snippet"]:
        st.markdown("**🔑 Key Code:**")
        st.markdown(
            display_data["highlighted_snippet"],
            unsafe_allow_html=True
        )

    if display_data["sources"]:
        with st.expander("📌 Source Files", expanded=True):
            for i, src in enumerate(display_data["sources"]):
                st.markdown(f"**{src['label']}**")
                if src["excerpt"]:
                    st.code(
                        src["excerpt"],
                        language = src.get("language", "text")
                    )
                if i < len(display_data["sources"]) - 1:
                    st.divider()


# ──────────────────────────────────────────────
# UI — Welcome Screen
# ──────────────────────────────────────────────

def render_welcome():
    """Show onboarding screen before any repo is uploaded."""

    st.markdown("## 🧠 Welcome to CodeSage")
    st.markdown(
        "Upload a ZIP of any codebase and ask questions about it "
        "in plain English. CodeSage finds the exact file and function "
        "that answers your question — with source citations."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🏗️ Architecture")
        st.markdown("""
- *"How is this project structured?"*
- *"What does the main entry point do?"*
- *"How are the modules organized?"*
- *"What design patterns are used here?"*
        """)

    with col2:
        st.markdown("### 🔐 Specific Features")
        st.markdown("""
- *"How does user authentication work?"*
- *"Where is the database connected?"*
- *"How are API routes defined?"*
- *"Where is error handling implemented?"*
        """)

    with col3:
        st.markdown("### 🔍 Code Understanding")
        st.markdown("""
- *"What does the process_payment() function do?"*
- *"How does this class manage state?"*
- *"What does this module export?"*
- *"How are environment variables loaded?"*
        """)

    st.divider()

    st.markdown("### 📦 How to Upload")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\nFind your project folder on your computer")
    with col2:
        st.info("**Step 2**\nRight-click → Send to → Compressed (ZIP) folder")
    with col3:
        st.info("**Step 3**\nUpload the ZIP from the sidebar and click Index")

    st.divider()
    st.info(
        "👈 ZIP your project folder and upload it from the sidebar to get started."
    )


# ──────────────────────────────────────────────
# UI — Main Chat Interface
# ──────────────────────────────────────────────

def render_main():
    """Render the main chat interface."""

    st.title("🧠 CodeSage")
    st.subheader("Ask anything about your codebase — in plain English.")

    if not st.session_state.processing_done:
        render_welcome()
        return

    repo = st.session_state.repo_info

    # Mode toggles
    col1, col2 = st.columns([1, 1])
    with col1:
        architecture_mode = st.toggle(
            "🏗️ Architecture Mode",
            help = "Get a high-level overview of the codebase structure"
        )
    with col2:
        if repo:
            st.caption(
                f"📁 **{repo.repo_name}** · "
                f"{repo.total_files} files · "
                f"{repo.total_chunks} chunks"
            )

    if architecture_mode:
        st.info(
            "🏗️ Architecture mode active — "
            "ask about overall structure, patterns, and design decisions."
        )

    st.divider()

    # Render chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant", avatar="🧠"):
                render_answer(message["content"])

    # Chat input
    question = st.chat_input(
        placeholder = "Ask about the codebase... e.g. 'How does authentication work?'"
    )

    if question:

        # Add to history and show immediately
        st.session_state.chat_history.append({
            "role"    : "user",
            "content" : question
        })
        with st.chat_message("user"):
            st.write(question)

        # Generate answer
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("🔍 Searching the codebase..."):
                try:
                    code_answer  = st.session_state.chain.ask(
                        question          = question,
                        architecture_mode = architecture_mode
                    )
                    display_data = format_answer_for_display(code_answer)

                except Exception as e:
                    display_data = {
                        "answer"             : f"An error occurred: {str(e)}",
                        "confidence"         : "⚫ Error",
                        "sources"            : [],
                        "warnings"           : [
                            "Something went wrong. Please try again."
                        ],
                        "has_code"           : False,
                        "highlighted_snippet": None,
                        "language"           : None
                    }
                    import traceback
                    print(traceback.format_exc())

            render_answer(display_data)

        # Save to history
        st.session_state.chat_history.append({
            "role"    : "assistant",
            "content" : display_data
        })


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

def main():
    initialize_session_state()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()