import streamlit as st
from src.retriever import Retriever
from src.generator import GeminiGenerator, CLASSIC_INSUFFICIENT_MSG
from src.vision import VisionExtractor

st.set_page_config(page_title="NexusAI", layout="wide", initial_sidebar_state="expanded")

# ---------- Custom CSS for Dark Theme ----------
st.markdown("""
<style>
    /* Main background and text colors */
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Header styling */
    h1 {
        color: #fff !important;
        font-weight: 800;
        letter-spacing: -1px;
        text-align: center;
        padding: 1rem 0;
        font-size: 3rem !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f1e 100%);
        border-right: 2px solid #7a5fff;
    }
    
    [data-testid="stSidebar"] h1 {
        color: #00d4ff;
        font-size: 1.8rem !important;
        text-align: left;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: rgba(26, 26, 46, 0.6);
        border: 1px solid rgba(122, 95, 255, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
    }
    
    /* User message */
    [data-testid="stChatMessageContent"] {
        color: #e0e0e0;
    }
    
    /* Input styling */
    .stChatInputContainer {
        background: rgba(26, 26, 46, 0.8);
        border: 2px solid #7a5fff;
        border-radius: 25px;
        padding: 0.5rem;
    }
    
    .stTextInput input {
        background: rgba(15, 15, 30, 0.9);
        border: 1px solid #7a5fff;
        border-radius: 10px;
        color: #fff;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(90deg, #7a5fff 0%, #00d4ff 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(122, 95, 255, 0.4);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(122, 95, 255, 0.6);
    }
    
    /* Slider styling */
    .stSlider {
        padding: 1rem 0;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(26, 26, 46, 0.6);
        border: 2px dashed #7a5fff;
        border-radius: 15px;
        padding: 2rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(122, 95, 255, 0.2);
        border-radius: 10px;
        color: #00d4ff;
        font-weight: 600;
    }
    
    /* Caption text */
    .stCaption {
        color: #00d4ff;
        font-weight: 500;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #7a5fff !important;
    }
    
    /* Success message */
    .stSuccess {
        background: rgba(0, 212, 255, 0.1);
        border-left: 4px solid #00d4ff;
        color: #00d4ff;
    }
    
    /* Markdown in chat */
    .stMarkdown {
        color: #e0e0e0;
    }
    
    /* Separator */
    hr {
        border-color: rgba(122, 95, 255, 0.3);
    }
    
    /* Checkbox */
    .stCheckbox {
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Cached singletons ----------
@st.cache_resource
def get_retriever():
    return Retriever()

@st.cache_resource
def get_generator():
    return GeminiGenerator()

@st.cache_resource
def get_vision():
    return VisionExtractor()

retriever = get_retriever()
generator = get_generator()
vision = get_vision()


# ---------- Sidebar ----------
with st.sidebar:
    st.title("⚡ Control Panel")
    st.markdown("---")
    top_k = st.slider("🔍 Top-K retrieved chunks", 1, 10, 3, 1)
    debug_mode = st.checkbox("🐛 Show retrieval debug", value=False)
    
    if st.button("🔄 Reload Knowledge Base"):
        with st.spinner("Rebuilding index from TXT files..."):
            retriever.reload()
        st.success("✅ Knowledge base reloaded!")

    st.markdown("---")
    st.markdown("### 📊 Statistics")
    st.caption(f"📚 Subjects: **{len(retriever.get_subjects())}**")
    st.caption(f"📄 Total chunks: **{retriever.get_chunk_count()}**")
    st.markdown("---")
    st.caption("💡 **Pro Tip:** Upload an image and ask your question!")

# ---------- Session State ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = None
if "last_query" not in st.session_state:
    st.session_state.last_query = None

st.markdown('<div style="text-align: center; margin-bottom: 1rem;"><svg width="80" height="80" viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg" style="display: inline-block;"><circle cx="256" cy="120" r="40" fill="#1a1a2e"/><circle cx="256" cy="280" r="180" fill="#ffffff" stroke="#1a1a2e" stroke-width="20"/><line x1="110" y1="200" x2="110" y2="120" stroke="#1a1a2e" stroke-width="15"/><circle cx="110" cy="120" r="20" fill="#1a1a2e"/><line x1="402" y1="200" x2="402" y2="120" stroke="#1a1a2e" stroke-width="15"/><circle cx="402" cy="120" r="20" fill="#1a1a2e"/><circle cx="90" cy="280" r="30" fill="#1a1a2e"/><circle cx="422" cy="280" r="30" fill="#1a1a2e"/><rect x="150" y="230" width="212" height="100" rx="50" fill="#00d4ff" stroke="#1a1a2e" stroke-width="15"/><circle cx="210" cy="280" r="25" fill="#1a1a2e"/><circle cx="302" cy="280" r="25" fill="#1a1a2e"/><path d="M 220 370 Q 256 390 292 370" fill="#1a1a2e"/><rect x="180" y="420" width="152" height="70" rx="15" fill="#ffffff" stroke="#1a1a2e" stroke-width="15"/><rect x="130" y="440" width="30" height="60" rx="15" fill="#ffffff" stroke="#1a1a2e" stroke-width="12"/><rect x="352" y="440" width="30" height="60" rx="15" fill="#ffffff" stroke="#1a1a2e" stroke-width="12"/><ellipse cx="256" cy="500" rx="50" ry="20" fill="#1a1a2e"/></svg><h1 style="margin: 0.5rem 0 0 0; color: #fff; font-weight: 800; font-size: 3rem;">NexusAI</h1><p style="color: #7a5fff; font-size: 1.2rem; margin-top: 0.5rem;">Advanced Image + Text Knowledge Retrieval</p></div>', unsafe_allow_html=True)

# Uploader resets automatically using dynamic key
uploaded_image = st.file_uploader(
    "📸 Upload an image to extract text",
    type=["png", "jpg", "jpeg"],
    key=f"uploader_{len(st.session_state.messages)}"
)

extracted_text = None  # for UI display


# ---------- Chat UI ----------
chat_container = st.container()
user_question = st.chat_input("💬 Ask me anything about your textbooks...")


if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})

    combined_query_parts = []

    if uploaded_image is not None:
        with st.spinner("🔍 Reading text from image..."):
            extracted_text = vision.extract_text(uploaded_image)
            if extracted_text:
                combined_query_parts.append(extracted_text)

    if extracted_text:
        st.markdown(f"<div style='background: rgba(122, 95, 255, 0.2); padding: 1rem; border-radius: 10px; border-left: 4px solid #7a5fff; margin: 1rem 0;'><strong>📌 Extracted Text:</strong><br><br>{extracted_text}</div>", unsafe_allow_html=True)

    combined_query_parts.append(user_question)
    final_query = "\n\n".join([p for p in combined_query_parts if p.strip()])

    with st.spinner("🔎 Retrieving relevant textbook content..."):
        retrieved = retriever.retrieve(final_query, top_k=top_k)

    st.session_state.last_retrieved = retrieved
    st.session_state.last_query = final_query

    if not retrieved:
        answer = CLASSIC_INSUFFICIENT_MSG
    else:
        with st.spinner("🤖 Thinking..."):
            answer = generator.generate(user_question, retrieved)

    st.session_state.messages.append({"role": "assistant", "content": answer})


# Render chat history
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ---------- Debug Panel ----------
if debug_mode and st.session_state.last_retrieved:
    st.markdown("---")
    with st.expander("🔍 Retrieval Debug Panel", expanded=False):
        st.code(st.session_state.last_query, language="text")
        for ch in st.session_state.last_retrieved:
            st.markdown(
                f"**📚 {ch['subject']}** • Page {ch['page']} • Chunk {ch['chunk_id']} • Score `{ch['score']:.4f}`"
            )
            st.caption(ch["text"][:350] + ("..." if len(ch["text"]) > 350 else ""))
            st.markdown("---")