import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

# ── 資料來源設定 ────────────────────────────────────────────────
# 每個來源指定：檔名、要 embed 的欄位、要回傳的欄位（None = 同 embed 欄）
SOURCES = [
    {
        "file": "東吳資科常見問題彙整.csv",
        "embed_col": "query",
        "answer_col": "answer",
        "schema": "qa",
        "no_header": True,       # 此檔無標題列，第一欄=query，第二欄=answer
    },
    {
        "file": "東吳資科_5000筆高品質學生問法資料集.csv",
        "embed_col": "學生問法",
        "answer_col": "答案",
        "schema": "qa",
    },
    {
        "file": "Knowledge_Base_Final_v5.csv",
        "embed_template": "{類別} {子類別} {關鍵標籤}",  # 短關鍵字組合，比長段落更易命中
        "answer_col": "知識內容",
        "schema": "knowledge",
    },
]

def _load_source(src: dict):
    """讀取單一來源，回傳 (embed_texts, answer_texts) 兩個 list。"""
    try:
        if src.get("no_header"):
            df = pd.read_csv(src["file"], header=None)
            df.columns = [src["embed_col"], src["answer_col"]] + list(range(len(df.columns) - 2))
        else:
            df = pd.read_csv(src["file"])
        df = df.dropna(subset=[src["answer_col"]])
    except Exception as e:
        print(f"[警告] 讀取 {src['file']} 失敗：{e}")
        return [], []

    if src.get("embed_template"):
        embed_texts = df.apply(
            lambda row: src["embed_template"].format(**{k: str(row[k]) for k in row.index}),
            axis=1
        ).tolist()
    else:
        embed_texts = df[src["embed_col"]].astype(str).tolist()

    answer_texts = df[src["answer_col"]].astype(str).tolist()
    print(f"  ✓ {src['file']}：{len(embed_texts)} 筆")
    return embed_texts, answer_texts

@st.cache_resource
def build_vector_db():
    print("正在載入所有知識來源...")
    all_embed, all_answer = [], []
    for src in SOURCES:
        e, a = _load_source(src)
        all_embed  += e
        all_answer += a

    if not all_embed:
        print("無可用資料，資料庫建置失敗。")
        return None, None, None

    print(f"共載入 {len(all_embed)} 筆資料，載入 Embedding 模型中...")
    model = SentenceTransformer("shibing624/text2vec-base-chinese")

    print("建立向量索引中（首次啟動約需 1-2 分鐘）...")
    embeddings = model.encode(all_embed, normalize_embeddings=True, show_progress_bar=False)
    print("向量索引建置完成。\n")
    return model, embeddings, all_answer

model, embeddings, all_answers = build_vector_db()

_ANS_FU = (
    "資科系輔系申請條件（114學年度）：前一學期總成績平均75分以上"
    "（前學期無成績者不得申請）。申請時需繳：歷年成績單、個人基本資料"
    "（含自傳與修課規劃）、其他有利審查資料。達成績標準後審酌錄取。"
)
_ANS_DUAL = (
    "資科系雙主修申請條件（114學年度）：在學每一學期總成績平均80分以上"
    "（允許前學期無成績），審酌錄取。申請時需繳：歷年成績單、個人基本資料"
    "（含自傳與修課規劃）、其他有利審查資料。注意事項："
    "（1）抵免後必修科目不足40學分需補修本系指定科目；"
    "（2）大四「專題實作」須依本系規定修讀，詳見系官網課程資訊之專題實作專區。"
)

def get_relevant_context(query: str, k: int = 5) -> str:
    if model is None:
        return "無法檢索資料，資料庫尚未建立。"

    pinned = []
    has_fu   = "輔系" in query
    has_dual = "雙主修" in query

    if has_fu and not has_dual:
        pinned.append(_ANS_FU)
    elif has_dual and not has_fu:
        pinned.append(_ANS_DUAL)
    elif has_fu and has_dual:
        pinned.extend([_ANS_FU, _ANS_DUAL])

    remaining = max(0, k - len(pinned))
    query_vec = model.encode([query], normalize_embeddings=True)
    scores = (embeddings @ query_vec.T).flatten()
    top_k = np.argsort(scores)[::-1][:remaining]

    context = ""
    for i, ans in enumerate(pinned):
        context += f"[資料{i+1}]\n{ans}\n\n"
    for j, idx in enumerate(top_k):
        context += f"[資料{len(pinned)+j+1}]\n{all_answers[idx]}\n\n"
    return context
