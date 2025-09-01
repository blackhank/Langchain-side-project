# -- 修正 Windows Unicode 輸出問題 --
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# -------------------------------------

import os
import json
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- 全域設定 ---
# 定義模型、原始文件路徑和向量資料庫路徑
MODEL_NAME = "llama3"
SOURCE_DOCS_PATH = "./source_documents"
VECTOR_STORE_PATH = "./vector_store"
HALLUCINATION_LOG_FILE = "hallucination_log.jsonl"

def log_hallucination(log_data):
    """將標註的幻覺資料寫入日誌檔案"""
    try:
        with open(HALLUCINATION_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"寫入日誌時發生錯誤: {e}")

def load_documents(directory_path):
    """從指定目錄載入所有 .txt 文件"""
    print(f"從 '{directory_path}' 目錄載入文件中...")
    # 使用 DirectoryLoader 載入所有 txt 文件
    loader = DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    documents = loader.load()
    print(f"成功載入 {len(documents)} 份文件。")
    return documents

def create_and_save_vector_store(documents, embeddings):
    """建立向量儲存並將其儲存到磁碟"""
    print("正在分割文件並建立新的向量儲存...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(documents)
    
    vector_store = FAISS.from_documents(split_documents, embeddings)
    
    print(f"正在將向量儲存到 '{VECTOR_STORE_PATH}'...")
    vector_store.save_local(VECTOR_STORE_PATH)
    print("儲存成功。")
    return vector_store

def get_retriever(embeddings):
    """載入或建立向量儲存，並回傳一個檢索器"""
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"正在從 '{VECTOR_STORE_PATH}' 載入已存在的向量儲存...")
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        print("載入成功。")
    else:
        print("找不到已存在的向量儲存。")
        documents = load_documents(SOURCE_DOCS_PATH)
        if not documents:
            print(f"錯誤：在 '{SOURCE_DOCS_PATH}' 中找不到任何文件。請先新增 SOP 文件。")
            sys.exit()
        vector_store = create_and_save_vector_store(documents, embeddings)
        
    return vector_store.as_retriever()

def get_rag_chain(retriever, llm):
    """建立並回傳 RAG 鏈"""
    # 為藥品製造業客製化的提示
    prompt_template = '''
你是一個專業的藥品製造工廠操作員助手。
請只根據以下提供的標準作業程序 (SOP) 文件來精確回答問題。
如果文件內容無法回答問題，請明確地說「根據現有SOP文件，無法找到相關資訊」。
請保持回答簡潔、專業。

<context>
{context}
</context>

問題: {input}
'''
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

def main():
    """主執行函式"""
    print("--- 藥品製造 SOP 問答系統啟動 ---")
    print(f"(錯誤標註將記錄於: {HALLUCINATION_LOG_FILE})")

    # 1. 初始化模型
    print("[1/3] 正在初始化模型...")
    llm = ChatOllama(model=MODEL_NAME)
    embeddings = OllamaEmbeddings(model=MODEL_NAME)

    # 2. 取得檢索器 (自動載入或建立向量儲存)
    print("[2/3] 正在準備檢索器...")
    if not os.path.exists(SOURCE_DOCS_PATH):
        print(f"錯誤：找不到來源文件目錄 '{SOURCE_DOCS_PATH}'。")
        sys.exit()
    retriever = get_retriever(embeddings)

    # 3. 建立 RAG 鏈
    print("[3/3] 正在建立 RAG 鏈...")
    rag_chain = get_rag_chain(retriever, llm)

    print("\n--- 系統準備就緒 ---")
    print("您可以開始針對 SOP 文件內容提問了。例如：「Metformin 的硬度目標是多少?」")
    print("--------------------------------------------------------------------------")

    # --- 互動式問答迴圈 ---
    while True:
        question = input("\n> 請提問 (或輸入 'exit' 來結束): ")

        if question.lower() == 'exit':
            print("正在結束程式...")
            break
        
        if not question.strip():
            continue

        print("\n--- 模型回答 ---")
        
        input_data = {"input": question}
        response = rag_chain.invoke(input_data)
        
        answer = response.get("answer", "無法生成答案。")
        
        # 處理並附加答案來源
        source_documents = response.get("context", [])
        sources_text = ""
        if source_documents:
            unique_sources = set(os.path.basename(doc.metadata.get("source")) for doc in source_documents)
            sources_text = f"\n\n**來源文件:** {', '.join(unique_sources)}"
        
        # 輸出整合後的答案與來源
        print(f"{answer}{sources_text}")
        
        print("--------------------")

        # --- 使用者回饋與標註機制 ---
        feedback = input("> 若答案錯誤或不佳，請輸入 'flag' 來標註，或按 Enter 繼續: ")
        if feedback.lower().strip() == 'flag':
            log_data = {
                "question": question,
                "answer": answer,
                "context": [doc.dict() for doc in source_documents]
            }
            log_hallucination(log_data)
            print(">>> 已標註此筆回覆，感謝您的反饋。")

if __name__ == "__main__":
    main()
