import numpy as np
import pandas as pd
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
        "file": "東吳資科_新增知識資料集_1000plus.csv",
        "embed_col": "知識內容",
        "answer_col": "知識內容",
        "schema": "knowledge",   # 純知識段落
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
        df = df.dropna(subset=[src["embed_col"], src["answer_col"]])
    except Exception as e:
        print(f"[警告] 讀取 {src['file']} 失敗：{e}")
        return [], []
    embed_texts  = df[src["embed_col"]].astype(str).tolist()
    answer_texts = df[src["answer_col"]].astype(str).tolist()
    print(f"  ✓ {src['file']}：{len(embed_texts)} 筆")
    return embed_texts, answer_texts

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

def get_relevant_context(query: str, k: int = 5) -> str:
    if model is None:
        return "無法檢索資料，資料庫尚未建立。"

    query_vec = model.encode([query], normalize_embeddings=True)
    scores = (embeddings @ query_vec.T).flatten()
    top_k = np.argsort(scores)[::-1][:k]

    context = ""
    for i, idx in enumerate(top_k):
        context += f"[資料{i+1}]\n{all_answers[idx]}\n\n"
    return context
