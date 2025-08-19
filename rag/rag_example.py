# -- 修正 Windows Unicode 輸出問題 --
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# -------------------------------------

import os
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- RAG 設置 (只執行一次) ---

# 檢查文件是否存在
doc_path = "my_document.txt"
if not os.path.exists(doc_path):
    print(f"錯誤：找不到文件 '{doc_path}'。請確認檔案是否與程式在同一個目錄下。")
    exit()

print("RAG 程式啟動...")

# 1. 初始化 LLM 和 Embeddings 模型
print("[1/5] 正在初始化模型...")
llm = ChatOllama(model="llama3")
embeddings = OllamaEmbeddings(model="llama3")

# 2. 載入與處理文件
print(f"[2/5] 正在載入與處理文件: {doc_path}...")
loader = TextLoader(doc_path, encoding="utf-8")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

# 3. 建立向量儲存
print("[3/5] 正在建立向量儲存...")
vector = FAISS.from_documents(documents, embeddings)

# 4. 建立 RAG 鏈
print("[4/5] 正在建立 RAG 鏈...")
prompt = ChatPromptTemplate.from_template('''
請只根據以下上下文來回答問題：

<context>
{context}
</context>

問題: {input}
''')
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

print("[5/5] RAG 引擎準備就緒！可以開始提問了。")
print("--------------------------------------------------")

# --- 互動式問答迴圈 ---

while True:
    question = input("\n> 請針對文件內容提問 (或輸入 'exit' 來結束): ")

    if question.lower() == 'exit':
        print("正在結束程式...")
        break
    
    if not question.strip():
        continue

    print("--- 模型回答 ---")
    
    # 建立一個字典來傳遞給 invoke，只包含 'input'
    input_data = {"input": question}
    
    # 執行鏈並取得包含答案的字典
    response_chain = retrieval_chain.invoke(input_data)
    
    # 從結果中提取答案
    answer = response_chain.get("answer", "無法找到答案。")
    
    print(answer)
    print("--------------------")
