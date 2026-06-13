import streamlit as st
import time
from chatbot import ask_question  # 核心改動：改接 Yue 寫的大腦

# 1. 網頁基本設定
st.set_page_config(page_title="東吳資科智慧助手", page_icon="🎓", layout="wide")

# 自定義 CSS 讓介面更美觀
st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .stChatInput { border-top: 1px solid #ddd; }
    </style>
""", unsafe_allow_html=True)

st.title("💡 東吳資科智慧校園助手")
st.caption("🚀 Powered by Llama 3.1 & RAG Architecture")

# 2. 初始化對話紀錄
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. 側邊欄優化：系統狀態與功能區
with st.sidebar:
    st.image("https://www.scu.edu.tw/images/logo.png", width=200)
    st.header("🛠️ 助手控制台")
    
    # 優化 A：系統診斷看板
    with st.expander("📡 系統狀態", expanded=True):
        st.success("環境連線：正常 (HuggingFace)")
        st.success("知識庫：Numpy 向量庫已掛載")
        st.info("目前大腦：Llama-3.1-8B（Groq 雲端推論）")

    # 優化 B：快捷問題提示
    st.markdown("### 🔍 常用問題快捷鍵")
    quick_queries = [
        "體育課算學分嗎？",
        "畢業門檻除了學分還有其他規定嗎？",
        "每學期學雜費大約多少？",
        "如何申請獎學金？",
        "雙主修成績門檻",
    ]
    for q in quick_queries:
        if st.button(q, use_container_width=True):
            st.session_state.temp_prompt = q

    st.markdown("---")
    
    # 優化 C：實用小工具
    if st.button("🗑️ 清除對話紀錄", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    if st.button("📋 匯出本次對話", use_container_width=True):
        if st.session_state.messages:
            chat_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
            st.download_button("點此下載 .txt 檔", chat_text, file_name="chat_history.txt")
        else:
            st.warning("目前沒有對話紀錄可匯出")

# 4. 處理快捷鍵輸入邏輯
chat_input = st.chat_input("請問關於選課、學分、獎學金的問題...")
if "temp_prompt" in st.session_state:
    prompt = st.session_state.pop("temp_prompt")
else:
    prompt = chat_input

# 5. 顯示歷史與處理新訊息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt:
    # 存入使用者問題
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成 AI 回答
    with st.chat_message("assistant"):
        with st.spinner("🔍 正在翻閱資科系規章資料庫..."):
            start_time = time.time()
            
            # 核心改動：呼叫 Yue 的函式來取得答案
            try:
                response = ask_question(prompt)
            except Exception as e:
                response = f"系統發生錯誤：{e}"
                
            end_time = time.time()
            
            st.markdown(response)
            st.caption(f"⏱️ 檢索與推論耗時: {end_time - start_time:.2f} 秒 | 來源: 東吳資科規章向量庫")

    # 存入 AI 回答
    st.session_state.messages.append({"role": "assistant", "content": response})