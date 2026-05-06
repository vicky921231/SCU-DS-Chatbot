import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

FILE_PATH = "東吳資科常見問題彙整.csv"

def build_vector_db():
    print(f"正在讀取檔案：{FILE_PATH}")
    try:
        df = pd.read_csv(FILE_PATH)
        df = df.iloc[:, :2]
        df.columns = ['query', 'answer']
        df = df.dropna().reset_index(drop=True)
        print(f"成功讀取 {len(df)} 筆有效的 QA 資料")
    except Exception as e:
        print(f"讀取 CSV 發生錯誤：{e}")
        return None, None, None

    texts = [f"問題：{row['query']}\n答案：{row['answer']}" for _, row in df.iterrows()]

    print("載入中文 Embedding 模型中")
    model = SentenceTransformer("shibing624/text2vec-base-chinese")

    print("建立向量索引中")
    embeddings = model.encode(texts, normalize_embeddings=True)
    print("向量索引建置完成\n")
    return model, embeddings, df

model, embeddings, df = build_vector_db()

def get_relevant_context(query: str, k: int = 3) -> str:
    if model is None:
        return "無法檢索資料，資料庫尚未建立。"

    query_vec = model.encode([query], normalize_embeddings=True)
    scores = (embeddings @ query_vec.T).flatten()
    top_k = np.argsort(scores)[::-1][:k]

    context = ""
    for i, idx in enumerate(top_k):
        row = df.iloc[idx]
        context += f"[資料{i+1}]\n問：{row['query']}\n答：{row['answer']}\n\n"
    return context
