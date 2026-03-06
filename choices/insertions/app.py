"""
Streamlit GUI for DeepSeek reasoning injection.

Usage:
    streamlit run choices/insertions/app.py
"""

import streamlit as st
from utils import (
    make_client,
    stream_complete,
    stream_complete_with_prefix,
    CompletionResult,
)

st.set_page_config(page_title="DeepSeek Reasoning Inspector", layout="wide")
st.markdown("#### DeepSeek Reasoning Inspector")

# Compact buttons
st.markdown(
    """
<style>
div[data-testid="stHorizontalBlock"] button[kind="secondary"],
div[data-testid="stHorizontalBlock"] button[kind="primary"] {
    padding: 0.1rem 0.5rem;
    font-size: 0.75rem;
    min-height: 0;
    line-height: 1.4;
}
</style>
""",
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("DeepSeek API Key", type="password")
    model = st.selectbox("Model", ["deepseek-reasoner", "deepseek-chat"], index=0)
    # deepseek-reasoner: max output 64K (reasoning + answer combined), default 32K
    max_tokens = st.slider(
        "Max tokens (reasoning + answer)", 512, 65536, 16000, step=512
    )
    system_prompt = st.text_area(
        "System prompt:", value="", height=100, key="system_prompt"
    )

# --- State ---
for key, default in [
    ("api_messages", []),
    ("last_result", None),
    ("rc", 0),
    ("pending", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

pending = st.session_state.pending
busy = pending is not None


def get_client():
    try:
        return make_client(api_key=api_key or None)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()


def full_messages():
    msgs = []
    if system_prompt.strip():
        msgs.append({"role": "system", "content": system_prompt.strip()})
    msgs.extend(st.session_state.api_messages)
    return msgs


def trigger(action_type, user, r_prefix="", c_prefix=""):
    st.session_state.pending = {
        "type": action_type,
        "user": user,
        "r_prefix": r_prefix,
        "c_prefix": c_prefix,
    }
    st.session_state.rc += 1
    st.rerun()


rc = st.session_state.rc
result = st.session_state.last_result

# If busy, update user message from pending action
if busy and st.session_state.api_messages:
    st.session_state.api_messages[-1]["content"] = pending["user"]

# Current values
user_val = (
    st.session_state.api_messages[-1]["content"]
    if st.session_state.api_messages
    else ""
)
thinking_val = (result.reasoning or "") if result else ""
answer_val = result.content if result else ""

# Which sections are being regenerated?
regen_thinking = busy and pending["type"] in ("after_user", "from_thinking")
regen_answer = busy

# ============================================================
# User section
# ============================================================
icon_col, content_col = st.columns([0.06, 0.94])
with icon_col:
    st.markdown("## :bust_in_silhouette:")
with content_col:
    edited_user = st.text_area(
        "user",
        value=user_val,
        height=80,
        key=f"u_{rc}",
        label_visibility="collapsed",
        disabled=busy,
    )
    _, btn_col = st.columns([0.9, 0.1])
    with btn_col:
        st.button(
            "send",
            key=f"send_{rc}",
            type="primary",
            disabled=busy,
            use_container_width=True,
            help="Send (or resend) this message and generate a response",
        )

# ============================================================
# Thinking section
# ============================================================
icon_col, content_col = st.columns([0.06, 0.94])
with icon_col:
    st.markdown("## :brain:")
with content_col:
    if regen_thinking:
        thinking_slot = st.empty()
        r_prefix = pending.get("r_prefix", "")
        if pending["type"] == "from_thinking" and r_prefix:
            thinking_slot.markdown(r_prefix + " :orange[...]")
        else:
            thinking_slot.markdown(":orange[generating...]")
    else:
        thinking_slot = None
        edited_reasoning = st.text_area(
            "thinking",
            value=thinking_val,
            height=220,
            key=f"t_{rc}",
            label_visibility="collapsed",
            disabled=busy or not result,
        )

    _, b1, b2 = st.columns([0.8, 0.1, 0.1])
    with b1:
        st.button(
            "continue",
            key=f"cf_t_{rc}",
            type="primary",
            disabled=busy or not result,
            use_container_width=True,
            help="Model picks up where your text left off",
        )
    with b2:
        st.button(
            "redo below",
            key=f"rb_t_{rc}",
            disabled=busy or not result,
            use_container_width=True,
            help="Lock this thinking, regenerate the answer",
        )

# ============================================================
# Answer section
# ============================================================
icon_col, content_col = st.columns([0.06, 0.94])
with icon_col:
    st.markdown("## :speech_balloon:")
with content_col:
    if regen_answer:
        answer_slot = st.empty()
        c_prefix = pending.get("c_prefix", "")
        if pending["type"] == "from_answer" and c_prefix:
            answer_slot.markdown(c_prefix + " :orange[...]")
        else:
            answer_slot.markdown(":orange[waiting...]")
    else:
        answer_slot = None
        edited_content = st.text_area(
            "answer",
            value=answer_val,
            height=180,
            key=f"a_{rc}",
            label_visibility="collapsed",
            disabled=busy or not result,
        )

    _, btn_col = st.columns([0.9, 0.1])
    with btn_col:
        st.button(
            "continue",
            key=f"cf_a_{rc}",
            type="primary",
            disabled=busy or not result,
            use_container_width=True,
            help="Model picks up where your text left off",
        )

# ============================================================
# Handle button clicks
# ============================================================
if not busy:
    # "send" — (re)send the user message, generate everything fresh
    if st.session_state.get(f"send_{rc}") and edited_user.strip():
        if not st.session_state.api_messages:
            st.session_state.api_messages.append(
                {"role": "user", "content": edited_user}
            )
        else:
            st.session_state.api_messages[-1]["content"] = edited_user
        trigger("after_user", edited_user)

    if result:
        if st.session_state.get(f"cf_t_{rc}"):
            trigger("from_thinking", edited_user, r_prefix=edited_reasoning)
        if st.session_state.get(f"rb_t_{rc}"):
            trigger("after_thinking", edited_user, r_prefix=edited_reasoning)
        if st.session_state.get(f"cf_a_{rc}"):
            trigger(
                "from_answer",
                edited_user,
                r_prefix=edited_reasoning,
                c_prefix=edited_content,
            )

# ============================================================
# Execute streaming
# ============================================================
if busy:
    client = get_client()
    r_prefix = pending.get("r_prefix", "")
    c_prefix = pending.get("c_prefix", "")
    use_prefix = bool(r_prefix or c_prefix)

    r_buf, c_buf = [], []

    if use_prefix:
        it = stream_complete_with_prefix(
            client=client,
            messages=full_messages(),
            reasoning_prefix=r_prefix,
            content_prefix=c_prefix,
            model=model,
            max_tokens=max_tokens,
        )
    else:
        it = stream_complete(
            client=client,
            messages=full_messages(),
            model=model,
            max_tokens=max_tokens,
        )

    for kind, text in it:
        if kind == "reasoning":
            r_buf.append(text)
            if thinking_slot:
                thinking_slot.markdown(r_prefix + "".join(r_buf))
        else:
            c_buf.append(text)
            if answer_slot:
                answer_slot.markdown(c_prefix + "".join(c_buf))

    full_reasoning = r_prefix + "".join(r_buf)
    full_content = c_prefix + "".join(c_buf)

    if pending["type"] == "after_thinking":
        full_reasoning = r_prefix

    if thinking_slot:
        thinking_slot.markdown(full_reasoning or "*(no reasoning)*")
    if answer_slot:
        answer_slot.markdown(full_content)

    st.session_state.last_result = CompletionResult(
        reasoning=full_reasoning or None,
        content=full_content,
    )
    st.session_state.pending = None
    st.session_state.rc += 1
    st.rerun()
