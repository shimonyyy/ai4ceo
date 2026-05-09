"""
PDF 기반 멀티세션 RAG 챗봇

- Supabase 세션 저장/로드/삭제
- Supabase pgvector 기반 문서 검색
- OpenAI embedding 재사용
- 답변 스트리밍 출력
"""

import hashlib
import os
import re
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from supabase import Client, create_client


APP_TITLE = "PDF 기반 멀티세션 RAG 챗봇"
MODEL_OPTIONS = ["gpt-5.5", "claude-opus-4-7", "gemini-3-pro-preview"]
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_MATCH_COUNT = 8
DEFAULT_MATCH_THRESHOLD = 0.2
KST = timezone(timedelta(hours=9))


def disable_langsmith_remote() -> None:
    """LangChain 실행 중 LangSmith 원격 전송을 비활성화합니다."""
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_TRACING"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    for key in (
        "LANGSMITH_API_KEY",
        "LANGSMITH_ENDPOINT",
        "LANGSMITH_PROJECT",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_ENDPOINT",
        "LANGCHAIN_PROJECT",
        "LANGCHAIN_HUB_API_URL",
    ):
        os.environ.pop(key, None)


def load_environment() -> None:
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parents[1] if len(current_dir.parents) > 1 else current_dir
    for env_path in (current_dir / ".env", project_root / ".env", Path.cwd() / ".env"):
        if env_path.exists():
            load_dotenv(env_path, override=True)
    load_dotenv(override=True)


disable_langsmith_remote()
load_environment()
disable_langsmith_remote()


def sanitize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    cleaned = text.replace("\x00", "")
    cleaned = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    return cleaned.strip()


def compact_title(text: str, max_len: int = 48) -> str:
    text = sanitize_text(text)
    text = re.sub(r"\s+", " ", text)
    return text[:max_len].strip() or "새 세션"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_kst_time(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        normalized = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(KST).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value[:16].replace("T", " ")


def get_api_key(name: str) -> Optional[str]:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def get_supabase_key() -> Optional[str]:
    # publishable key를 우선 사용해 잘못 저장된 legacy anon key 때문에 인증이 깨지는 일을 피합니다.
    for key_name in ("SUPABASE_PUBLISHABLE_KEY", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_ANON_KEY", "SUPABASE_KEY"):
        value = get_api_key(key_name)
        if value and (not value.startswith("eyJ") or value.count(".") == 2):
            return value
    return None


@st.cache_resource(show_spinner=False)
def init_supabase() -> Optional[Client]:
    supabase_url = get_api_key("SUPABASE_URL")
    supabase_key = get_supabase_key()
    if not supabase_url or not supabase_key:
        return None
    return create_client(supabase_url, supabase_key)


@st.cache_resource(show_spinner=False)
def get_embeddings() -> OpenAIEmbeddings:
    api_key = get_api_key("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 필요합니다.")
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=api_key)


def get_chat_model(model_name: str, streaming: bool = False) -> Any:
    if model_name == "gpt-5.5":
        api_key = get_api_key("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY가 필요합니다.")
        return ChatOpenAI(
            model="gpt-5.5",
            temperature=1,
            streaming=streaming,
            openai_api_key=api_key,
        )

    if model_name == "claude-opus-4-7":
        api_key = get_api_key("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY가 필요합니다.")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model="claude-opus-4-7",
            temperature=1,
            streaming=streaming,
            anthropic_api_key=api_key,
        )

    if model_name == "gemini-3-pro-preview":
        api_key = get_api_key("GOOGLE_API_KEY") or get_api_key("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY 또는 GEMINI_API_KEY가 필요합니다.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model="gemini-3-pro-preview",
            temperature=1,
            streaming=streaming,
            google_api_key=api_key,
        )

    raise ValueError(f"지원하지 않는 모델입니다: {model_name}")


supabase = init_supabase()


st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")

st.markdown(
    """
<style>
.main .block-container {
    padding-top: 2.25rem !important;
    padding-bottom: 1rem !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 0.15rem !important;
    padding-bottom: 0.7rem !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 0.15rem !important;
}
div[data-testid="stVerticalBlock"] {
    gap: 0.65rem !important;
}
h1 {
    font-size: 1.45rem !important;
    font-weight: 800 !important;
    line-height: 1.35 !important;
    background: linear-gradient(90deg, #ff4da6, #ffb000, #47c765, #35a7ff, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 0 !important;
    margin-bottom: 0.35rem !important;
    padding-top: 0.1rem !important;
    overflow: visible !important;
}
h2 {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: #ffd700 !important;
}
h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #1f77b4 !important;
}
.stChatMessage {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
}
.stChatMessage * {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}
label, [data-testid="stMarkdownContainer"] p {
    color: #5b4b8a !important;
    font-weight: 650 !important;
}
section[data-testid="stSidebar"] label {
    color: #ffb000 !important;
    font-weight: 800 !important;
}
section[data-testid="stSidebar"] h1 {
    font-size: 1.65rem !important;
    line-height: 1.15 !important;
    margin-top: 0 !important;
    margin-bottom: 0.25rem !important;
}
section[data-testid="stSidebar"] .stAlert {
    padding: 0.55rem 0.75rem !important;
    border-radius: 0.8rem !important;
}
.stButton > button {
    border-radius: 0.5rem !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #ff67b3 0%, #c0267d 55%, #7c3aed 100%) !important;
    color: white !important;
    border: 0 !important;
    min-height: 2.55rem !important;
    font-size: 0.9rem !important;
    box-shadow: 0 4px 12px rgba(192, 47, 122, 0.25) !important;
    white-space: nowrap !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #ff8ac7 0%, #d13a88 50%, #8b5cf6 100%) !important;
    color: white !important;
}
section[data-testid="stSidebar"] .stButton > button:disabled {
    background: linear-gradient(135deg, #d86bab 0%, #b83b83 100%) !important;
    color: rgba(255, 255, 255, 0.65) !important;
    opacity: 0.85 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: #fff8db !important;
    border: 2px dashed #ffb000 !important;
    border-radius: 0.8rem !important;
    padding: 0.35rem !important;
}
section[data-testid="stSidebar"] hr {
    margin: 0.35rem 0 0.45rem 0 !important;
    border-color: #ffcf66 !important;
}
.small-muted {
    color: #777;
    font-size: 0.85rem;
}
</style>
""",
    unsafe_allow_html=True,
)


def init_state() -> None:
    defaults = {
        "current_session_id": str(uuid.uuid4()),
        "current_session_title": "새 세션",
        "chat_history": [],
        "processed_files": [],
        "selected_model": MODEL_OPTIONS[0],
        "loaded_session_selector": None,
        "show_vectordb": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def require_supabase() -> Client:
    if supabase is None:
        raise RuntimeError("SUPABASE_URL과 SUPABASE_SERVICE_ROLE_KEY 또는 SUPABASE_ANON_KEY가 필요합니다.")
    return supabase


def ensure_session_exists(session_id: str) -> None:
    sb = require_supabase()
    payload = {
        "id": session_id,
        "title": st.session_state.current_session_title or "새 세션",
        "model_name": st.session_state.selected_model,
        "processed_files": st.session_state.processed_files,
        "updated_at": utc_now_iso(),
    }
    sb.table("sessions").upsert(payload, on_conflict="id").execute()


def first_question_answer() -> Tuple[Optional[str], Optional[str]]:
    first_user: Optional[str] = None
    for message in st.session_state.chat_history:
        if message["role"] == "user" and first_user is None:
            first_user = message["content"]
        elif message["role"] == "assistant" and first_user:
            return first_user, message["content"]
    return first_user, None


def generate_session_title(user_question: str, ai_response: str) -> str:
    fallback = compact_title(user_question)
    try:
        llm = get_chat_model(st.session_state.selected_model, streaming=False)
        prompt = (
            "첫 번째 질문과 답변을 바탕으로 한국어 세션 제목을 18자 이내로 만드세요. "
            "따옴표, 마침표, 설명 없이 제목만 출력하세요.\n\n"
            f"질문: {user_question}\n\n답변: {ai_response[:1200]}"
        )
        response = llm.invoke(prompt)
        title = sanitize_text(getattr(response, "content", str(response)))
        title = re.sub(r"^[\"'`]+|[\"'`]+$", "", title).strip()
        return compact_title(title, 36) or fallback
    except Exception:
        return fallback


def save_session(session_id: Optional[str] = None) -> bool:
    try:
        sb = require_supabase()
        session_id = session_id or st.session_state.current_session_id
        question, answer = first_question_answer()
        if question and answer and st.session_state.current_session_title in ("새 세션", "", None):
            st.session_state.current_session_title = generate_session_title(question, answer)

        ensure_session_exists(session_id)
        sb.table("messages").delete().eq("session_id", session_id).execute()

        rows = []
        for idx, message in enumerate(st.session_state.chat_history):
            rows.append(
                {
                    "session_id": session_id,
                    "role": message["role"],
                    "content": sanitize_text(message["content"]),
                    "position": idx,
                }
            )
        if rows:
            sb.table("messages").insert(rows).execute()

        sb.table("sessions").update(
            {
                "title": st.session_state.current_session_title or "새 세션",
                "model_name": st.session_state.selected_model,
                "processed_files": st.session_state.processed_files,
                "updated_at": utc_now_iso(),
            }
        ).eq("id", session_id).execute()
        return True
    except Exception as exc:
        st.error(f"세션 저장 실패: {exc}")
        return False


def get_sessions() -> List[Dict[str, Any]]:
    if supabase is None:
        return []
    try:
        result = (
            supabase.table("sessions")
            .select("id,title,model_name,processed_files,created_at,updated_at")
            .order("updated_at", desc=True)
            .execute()
        )
        return result.data or []
    except Exception as exc:
        st.sidebar.error(f"세션 목록 로드 실패: {exc}")
        return []


def load_session(session_id: str) -> bool:
    try:
        sb = require_supabase()
        session_result = sb.table("sessions").select("*").eq("id", session_id).single().execute()
        session = session_result.data
        if not session:
            st.error("세션을 찾을 수 없습니다.")
            return False

        messages_result = (
            sb.table("messages")
            .select("role,content,position")
            .eq("session_id", session_id)
            .order("position")
            .execute()
        )
        st.session_state.current_session_id = session_id
        st.session_state.current_session_title = session.get("title") or "새 세션"
        st.session_state.selected_model = session.get("model_name") or st.session_state.selected_model
        st.session_state.processed_files = session.get("processed_files") or []
        st.session_state.chat_history = [
            {"role": row["role"], "content": row["content"]} for row in (messages_result.data or [])
        ]
        return True
    except Exception as exc:
        st.error(f"세션 로드 실패: {exc}")
        return False


def delete_session(session_id: str) -> bool:
    try:
        sb = require_supabase()
        sb.table("sessions").delete().eq("id", session_id).execute()
        check_result = sb.table("sessions").select("id").eq("id", session_id).limit(1).execute()
        if check_result.data:
            st.error("세션 삭제가 완료되지 않았습니다. Supabase 권한 또는 RLS 정책을 확인해주세요.")
            return False
        reset_screen()
        return True
    except Exception as exc:
        st.error(f"세션 삭제 실패: {exc}")
        return False


def reset_screen() -> None:
    st.session_state.current_session_id = str(uuid.uuid4())
    st.session_state.current_session_title = "새 세션"
    st.session_state.chat_history = []
    st.session_state.processed_files = []
    st.session_state.loaded_session_selector = None


def get_chunk_hash(content: str) -> str:
    return hashlib.sha256(sanitize_text(content).encode("utf-8")).hexdigest()


def existing_chunk_ids(content_hashes: List[str]) -> Dict[str, str]:
    if not content_hashes:
        return {}
    result = (
        require_supabase()
        .table("document_chunks")
        .select("id,content_hash")
        .in_("content_hash", list(set(content_hashes)))
        .execute()
    )
    return {row["content_hash"]: row["id"] for row in (result.data or [])}


def save_documents_to_supabase(chunks: List[Any], file_name: str, session_id: str) -> int:
    if not chunks:
        return 0

    sb = require_supabase()
    embeddings = get_embeddings()
    prepared = []
    for idx, chunk in enumerate(chunks):
        content = sanitize_text(chunk.page_content)
        if not content:
            continue
        metadata = dict(chunk.metadata or {})
        metadata.update({"file_name": file_name, "chunk_index": idx})
        prepared.append(
            {
                "content": content,
                "content_hash": get_chunk_hash(content),
                "file_name": file_name,
                "metadata": metadata,
                "chunk_index": idx,
            }
        )

    known_ids = existing_chunk_ids([row["content_hash"] for row in prepared])
    new_rows = [row for row in prepared if row["content_hash"] not in known_ids]
    if new_rows:
        vectors = embeddings.embed_documents([row["content"] for row in new_rows])
        insert_rows = []
        for row, vector in zip(new_rows, vectors):
            insert_rows.append(
                {
                    "content_hash": row["content_hash"],
                    "file_name": row["file_name"],
                    "content": row["content"],
                    "metadata": row["metadata"],
                    "embedding": vector,
                }
            )
        inserted = sb.table("document_chunks").insert(insert_rows).execute()
        for row in inserted.data or []:
            known_ids[row["content_hash"]] = row["id"]

    links = []
    for row in prepared:
        chunk_id = known_ids.get(row["content_hash"])
        if not chunk_id:
            continue
        links.append(
            {
                "session_id": session_id,
                "document_chunk_id": chunk_id,
                "file_name": file_name,
                "chunk_index": row["chunk_index"],
                "metadata": row["metadata"],
            }
        )
    if links:
        sb.table("session_documents").upsert(
            links,
            on_conflict="session_id,document_chunk_id",
        ).execute()
    return len(links)


def process_uploaded_files(uploaded_files: List[Any]) -> None:
    if not uploaded_files:
        st.warning("처리할 PDF 파일을 업로드해주세요.")
        return

    ensure_session_exists(st.session_state.current_session_id)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    total_chunks = 0
    processed_now = []

    progress = st.progress(0)
    status = st.empty()
    for index, uploaded_file in enumerate(uploaded_files):
        if uploaded_file.name in st.session_state.processed_files:
            status.info(f"이미 처리된 파일 재사용: {uploaded_file.name}")
            progress.progress((index + 1) / len(uploaded_files))
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            status.info(f"PDF 처리 중: {uploaded_file.name}")
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["file_name"] = uploaded_file.name
            chunks = splitter.split_documents(docs)
            saved_count = save_documents_to_supabase(
                chunks,
                uploaded_file.name,
                st.session_state.current_session_id,
            )
            total_chunks += saved_count
            processed_now.append(uploaded_file.name)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        progress.progress((index + 1) / len(uploaded_files))

    for file_name in processed_now:
        if file_name not in st.session_state.processed_files:
            st.session_state.processed_files.append(file_name)

    save_session(st.session_state.current_session_id)
    status.success(f"파일 처리 완료: {len(processed_now)}개 파일, {total_chunks}개 청크 연결")


def retrieve_context(question: str) -> Tuple[str, List[Dict[str, Any]]]:
    if supabase is None:
        return "", []
    try:
        query_embedding = get_embeddings().embed_query(question)
        result = supabase.rpc(
            "match_session_documents",
            {
                "query_embedding": query_embedding,
                "match_session_id": st.session_state.current_session_id,
                "match_threshold": DEFAULT_MATCH_THRESHOLD,
                "match_count": DEFAULT_MATCH_COUNT,
            },
        ).execute()
        rows = result.data or []
        context_blocks = []
        for idx, row in enumerate(rows, start=1):
            file_name = row.get("file_name") or (row.get("metadata") or {}).get("file_name") or "unknown"
            context_blocks.append(f"[문서 {idx} | {file_name}]\n{row.get('content', '')}")
        return "\n\n".join(context_blocks), rows
    except Exception as exc:
        st.warning(f"문서 검색 실패: {exc}")
        return "", []


def format_history(limit: int = 8) -> str:
    recent = st.session_state.chat_history[-limit:]
    lines = []
    for message in recent:
        role = "사용자" if message["role"] == "user" else "챗봇"
        lines.append(f"{role}: {message['content'][:1000]}")
    return "\n".join(lines)


def generate_followup_questions(question: str, answer: str, context_text: str) -> List[str]:
    try:
        llm = get_chat_model(st.session_state.selected_model, streaming=False)
        prompt = (
            "다음 PDF 기반 질의응답 흐름에서 사용자가 이어서 물어보면 좋은 질문 3개를 한국어로 만드세요. "
            "번호 없이 한 줄에 하나씩 질문만 출력하세요.\n\n"
            f"사용자 질문:\n{question}\n\n답변:\n{answer[:1600]}\n\n참고 문맥:\n{context_text[:1600]}"
        )
        response = llm.invoke(prompt)
        raw = sanitize_text(getattr(response, "content", str(response)))
        questions = []
        for line in raw.splitlines():
            cleaned = re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip()
            if cleaned:
                questions.append(cleaned)
        return questions[:3]
    except Exception:
        return [
            "이 문서에서 가장 중요한 핵심 내용은 무엇인가요?",
            "방금 답변과 관련된 근거 페이지를 더 자세히 설명해줄 수 있나요?",
            "실무에 적용하려면 어떤 점을 먼저 확인해야 하나요?",
        ]


def stream_answer(question: str) -> str:
    context_text, sources = retrieve_context(question)
    system_prompt = (
        "당신은 PDF 문서를 기반으로 답변하는 한국어 RAG 챗봇입니다. "
        "문서 문맥이 있으면 문맥을 우선 사용하고, 문맥에 없는 내용은 추측하지 말고 부족하다고 말하세요. "
        "답변은 읽기 쉽게 구조화하고 필요한 경우 근거 파일명을 언급하세요."
    )
    user_prompt = (
        f"현재 세션 제목: {st.session_state.current_session_title}\n\n"
        f"최근 대화:\n{format_history()}\n\n"
        f"검색된 문서 문맥:\n{context_text or '검색된 문서 문맥이 없습니다.'}\n\n"
        f"사용자 질문:\n{question}"
    )

    llm = get_chat_model(st.session_state.selected_model, streaming=True)
    placeholder = st.empty()
    answer = ""
    for chunk in llm.stream([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]):
        token = getattr(chunk, "content", "")
        if isinstance(token, list):
            token = "".join(str(item) for item in token)
        answer += str(token)
        placeholder.markdown(answer + "▌")

    followups = generate_followup_questions(question, answer, context_text)
    if followups:
        answer = answer.rstrip() + "\n\n### 향후 더 필요한 질문 3개\n"
        answer += "\n".join(f"{idx}. {item}" for idx, item in enumerate(followups, start=1))

    if sources:
        unique_files = []
        for row in sources:
            file_name = row.get("file_name") or (row.get("metadata") or {}).get("file_name")
            if file_name and file_name not in unique_files:
                unique_files.append(file_name)
        if unique_files:
            answer += "\n\n참고 파일: " + ", ".join(unique_files)

    placeholder.markdown(answer)
    return answer


def vector_file_names(session_id: Optional[str] = None) -> List[str]:
    if supabase is None:
        return []
    try:
        query = supabase.table("session_documents").select("file_name,session_id")
        if session_id:
            query = query.eq("session_id", session_id)
        result = query.execute()
        names = sorted({row["file_name"] for row in (result.data or []) if row.get("file_name")})
        return names
    except Exception as exc:
        st.sidebar.error(f"vectordb 조회 실패: {exc}")
        return []


def render_sidebar() -> None:
    st.sidebar.title("🌈 세션 관리 🐣")
    if supabase is None:
        st.sidebar.error("🧸 Supabase 환경 변수를 확인해주세요.")
    else:
        st.sidebar.success("🍀 Supabase 연결 준비 완료")

    st.session_state.selected_model = st.sidebar.selectbox(
        "🤖 LLM 선택",
        MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(st.session_state.selected_model)
        if st.session_state.selected_model in MODEL_OPTIONS
        else 0,
    )

    sessions = get_sessions()
    session_labels = {"새 세션": None}
    current_session_saved = False
    for item in sessions:
        updated_at = format_kst_time(item.get("updated_at"))
        title = item.get("title") or "제목 없음"
        session_labels[f"{title} · {updated_at}"] = item["id"]
        if item["id"] == st.session_state.current_session_id:
            current_session_saved = True

    selected_label = st.sidebar.selectbox("⭐ 세션 리스트", list(session_labels.keys()))
    selected_session_id = session_labels[selected_label]
    delete_target_id = selected_session_id or (
        st.session_state.current_session_id if current_session_saved else None
    )

    if selected_session_id and st.session_state.loaded_session_selector != selected_session_id:
        if load_session(selected_session_id):
            st.session_state.loaded_session_selector = selected_session_id
            st.rerun()

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("💾 세션저장", use_container_width=True):
            if save_session(st.session_state.current_session_id):
                st.sidebar.success("✨ 저장 완료")
                st.rerun()
    with col2:
        if st.button("📂 세션로드", use_container_width=True, disabled=selected_session_id is None):
            if selected_session_id and load_session(selected_session_id):
                st.session_state.loaded_session_selector = selected_session_id
                st.rerun()

    col3, col4 = st.sidebar.columns(2)
    with col3:
        if st.button("🗑️ 세션삭제", use_container_width=True, disabled=delete_target_id is None):
            if delete_target_id and delete_session(delete_target_id):
                st.sidebar.success("🧹 삭제 완료")
                st.rerun()
    with col4:
        if st.button("🫧 초기화", use_container_width=True):
            reset_screen()
            st.rerun()

    if st.sidebar.button("📋 세션 리스트토 메뉴", use_container_width=True):
        st.rerun()

    if st.sidebar.button("🧠 vectordb 버튼", use_container_width=True):
        st.session_state.show_vectordb = not st.session_state.show_vectordb

    if st.session_state.show_vectordb:
        st.sidebar.subheader("현재 세션 vectordb 파일")
        names = vector_file_names(st.session_state.current_session_id)
        if names:
            for name in names:
                st.sidebar.write(f"- {name}")
        else:
            st.sidebar.caption("현재 세션에 저장된 파일이 없습니다.")

    st.sidebar.divider()
    uploaded_files = st.sidebar.file_uploader(
        "📎 PDF 파일을 선택하세요",
        type=["pdf"],
        accept_multiple_files=True,
        help="처리된 문서 청크는 Supabase pgvector에 저장되어 다음 실행 때도 재사용됩니다.",
    )
    if st.sidebar.button("🚀 파일 처리하기", use_container_width=True):
        try:
            process_uploaded_files(uploaded_files or [])
        except Exception as exc:
            st.sidebar.error(f"파일 처리 실패: {exc}")

    if st.session_state.processed_files:
        st.sidebar.markdown("#### 처리된 파일")
        for file_name in st.session_state.processed_files:
            st.sidebar.write(f"- {file_name}")


def render_main() -> None:
    st.title(f"📚 {APP_TITLE} ✨")
    st.caption(f"🌟 현재 세션: {st.session_state.current_session_title}")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("질문을 입력하세요")
    if question:
        user_message = {"role": "user", "content": sanitize_text(question)}
        st.session_state.chat_history.append(user_message)
        save_session(st.session_state.current_session_id)

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                answer = stream_answer(question)
            except Exception as exc:
                answer = f"답변 생성 실패: {exc}"
                st.error(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        save_session(st.session_state.current_session_id)
        st.rerun()


render_sidebar()
render_main()
