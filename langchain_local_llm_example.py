# -- 修正 Windows Unicode 輸出問題 --
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# -------------------------------------

# 匯入我們需要的 LangChain 元件
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 初始化模型
print("正在初始化 Ollama 模型...")
llm = ChatOllama(model="llama3")
print("模型初始化完成。可以開始提問了。")

# 2. 建立提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個有用的AI助理。請用繁體中文回答。"),
    ("user", "{question}")
])

# 3. 建立一個簡單的鏈
chain = prompt | llm | StrOutputParser()

# 4. 建立一個無限迴圈來持續接收使用者問題
while True:
    # 從使用者獲取問題
    question = input("\n> 請輸入您的問題 (或輸入 'exit' 來結束): ")

    # 如果使用者輸入 'exit'，就跳出迴圈
    if question.lower() == 'exit':
        print("正在結束程式...")
        break

    # 執行鏈並即時輸出結果
    print("--- 模型回答 ---")
    # 使用 stream 方法來讓回答一個字一個字地顯示，體驗更好
    full_response = ""
    for chunk in chain.stream({"question": question}):
        print(chunk, end="", flush=True)
        full_response += chunk
    print("\n--------------------")
