import streamlit as st
import requests
import json
from typing import List, Dict

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "llama-3.2-1b-instruct"

st.set_page_config(
    page_title="AI Chatbot with Web Search",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Chatbot with Web Search")

def initialize_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def add_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})

def get_recent_messages(max_messages: int = 5) -> List[Dict]:
    if len(st.session_state.messages) > max_messages:
        return st.session_state.messages[-max_messages:]
    return st.session_state.messages

#Get the streaming data from the api
def stream_response_from_api(messages: List[Dict], use_search: bool = False):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "use_search": use_search
    }

    try:
        response = requests.post(API_URL, json=payload, stream=True, headers=headers, timeout=300)
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                try:
                    data = json.loads(line)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        
                        if delta.get("search_performed"):
                            yield {"type": "search_indicator"}
                            continue
                        
                        content = delta.get("content")
                        if content:
                            yield {"type": "content", "content": content}
                        
                        finish_reason = choices[0].get("finish_reason")
                        if finish_reason == "stop":
                            break
                except json.JSONDecodeError:
                    continue
                        
    except Exception as e:
        yield {"type": "error", "content": f"Unexpected error: {str(e)}"}

#Generate frontend according to streaming response coming
def response_generator(messages: List[Dict], use_search: bool = False):
    full_response = ""
    search_shown = False
    
    for chunk in stream_response_from_api(messages, use_search):
        if chunk["type"] == "search_indicator" and not search_shown:
            yield "🔍 Searching the web...\n\n"
            search_shown = True
        elif chunk["type"] == "content":
            content = chunk["content"]
            full_response += content
            yield content
        elif chunk["type"] == "error":
            yield chunk["content"]
    
    if full_response.strip():
        add_message("assistant", full_response)

initialize_chat_history()

with st.sidebar:
    force_search = st.checkbox(
        "🔍 Force Web Search",
        value=False,
        help="Force web search for every query instead of automatic keyword-based detection"
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):
    add_message("user", prompt)
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    recent_messages = get_recent_messages(max_messages=5)
    
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(recent_messages, force_search))
