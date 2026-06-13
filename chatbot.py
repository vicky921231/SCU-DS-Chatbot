import os
from dotenv import load_dotenv
from groq import Groq
from rag import get_relevant_context

load_dotenv()

MODEL_NAME = "llama-3.1-8b-instant"

def _get_key():
    try:
        import streamlit as st
        return st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    except Exception:
        return os.getenv("GROQ_API_KEY")

def ask_question(query: str) -> str:
    context = get_relevant_context(query)

    system_prompt = """你是一個專為東吳大學資料科學系設計的專業行政助手。
請嚴格遵守以下規則：
1. 你只能根據下方 [參考資料] 區塊中提供的資訊來回答問題。
2. 重要背景知識：「資科系」、「資料科學系」、「巨資學院資料科學系」、「巨資學院」指的都是同一個系所，回答時請靈活對應，不要因為名稱不同就說找不到資料。
3. 如果參考資料中有相關資訊，請直接根據資料回答，不要說找不到。只有在參考資料完全沒有相關內容時，才回答：「很抱歉，目前的資料庫中沒有這方面的資訊。」
4. 絕對不可以自己編造或推測資料中沒有的內容。
5. 請用親切、專業的台灣繁體中文回答。"""

    user_prompt = f"【參考資料】\n{context}\n【使用者問題】\n{query}"

    client = Groq(api_key=_get_key())
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()
