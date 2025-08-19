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

# --- 模型名稱設定 ---
FINETUNED_MODEL_NAME = "pirate-model"
# ---------------------

# 1. 初始化模型
print(f"正在初始化微調模型: {FINETUNED_MODEL_NAME}...")
llm = ChatOllama(model=FINETUNED_MODEL_NAME)
print("模型初始化完成。可以開始提問了。")

# 2. 建立提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個根據特定指示微調過的 AI 模型。"),
    ("user", "{question}")
])

# 3. 建立鏈
chain = prompt | llm | StrOutputParser()

# 4. 建立一個無限迴圈來持續接收使用者問題
while True:
    question = input("\n> 請對微調後的模型提問 (或輸入 'exit' 來結束): ")

    if question.lower() == 'exit':
        print("正在結束程式...")
        break

    print("--- 微調模型回答 ---")
    full_response = ""
    for chunk in chain.stream({"question": question}):
        print(chunk, end="", flush=True)
        full_response += chunk
    print("\n---------------------")
