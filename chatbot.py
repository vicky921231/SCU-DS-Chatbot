import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from rag import get_relevant_context

load_dotenv()

try:
    import streamlit as st
    HF_TOKEN = st.secrets.get("HUGGINGFACE_ACCESS_TOKEN") or os.getenv("HUGGINGFACE_ACCESS_TOKEN")
except Exception:
    HF_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
client = InferenceClient(model=MODEL_NAME, token=HF_TOKEN)

def ask_question(query: str) -> str:
    context = get_relevant_context(query)

    system_prompt = """你是一個專為東吳大學設計的專業行政助手。
        請嚴格遵守以下規則：
        1. 你只能根據下方 [參考資料] 區塊中提供的資訊來回答問題。
        2. 如果參考資料中沒有包含答案，請直接回答：「很抱歉，目前的資料庫中沒有這方面的資訊。」，絕對不可以自己編造或推測答案。
        3. 請用親切、專業的台灣繁體中文回答。"""

    user_prompt = f"""【參考資料】\n{context}\n【使用者問題】\n{query}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    response = client.chat_completion(messages=messages, max_tokens=512)
    return response.choices[0].message.content.strip()
