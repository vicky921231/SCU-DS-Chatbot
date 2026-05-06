import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 設定資料檔案路徑 (請確保 CSV 檔與此程式碼在同一個資料夾)
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
        print(f"讀取 CSV 發生錯誤，請檢查檔案路徑：{e}")
        return None

    documents = []
    for index, row in df.iterrows():
        text_content = f"問題：{row['query']}\n答案：{row['answer']}"
        metadata = {"query": str(row['query']), "answer": str(row['answer'])}
        doc = Document(page_content=text_content, metadata=metadata)
        documents.append(doc)

    print("載入中文 Embedding 模型中")
    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

    print("建立 Chroma 向量資料庫中")
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings
    )
    print("向量資料庫建置完成\n")
    return vector_db

# 在模組被 import 時，初始化一次資料庫，避免每次問問題都重建
vector_db = build_vector_db()

def get_relevant_context(query: str, k: int = 3) -> str:

    if vector_db is None:
        return "無法檢索資料，資料庫尚未建立。"
        
    docs = vector_db.similarity_search(query, k=k)
    context = ""
    for i, doc in enumerate(docs):
        context += f"[資料{i+1}]\n問：{doc.metadata['query']}\n答：{doc.metadata['answer']}\n\n"
    return context