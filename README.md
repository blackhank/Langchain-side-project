# 本地端 LLM 應用程式範例 (LangChain + Ollama)

這個專案包含了一系列使用 LangChain 和 Ollama 在本地端運行大型語言模型的 Python 範例。
本專案為利用gemini cli進行vibe coding，使用prompt進行程式架構生成、除錯與推送並由作者監控其是否合理

---

## 專案結構

```
.
├── assets
│   ├── finetune_diagram.svg
│   └── rag_diagram.svg
├── finetune
│   ├── ... (微調相關檔案)
├── rag
│   ├── source_documents        (RAG 的知識來源文件)
│   │   ├── SOP-LIQUID-15.txt
│   │   └── SOP-TABLET-42.txt
│   ├── rag_example.py          (RAG 應用主腳本)
│   └── vector_store            (持久化向量儲存資料夾 - 自動生成)
├── langchain_local_llm_example.py
└── README.md
```

---

## 環境設定

1.  **安裝 Ollama**
    請至 [https://ollama.com/](https://ollama.com/) 下載並安裝 Ollama。安裝後請確保 Ollama 應用程式正在背景執行。

2.  **下載基礎模型**
    本專案主要使用 `llama3` 模型。請透過 Ollama 應用程式的 GUI 或在終端機執行以下指令來下載模型：
    ```bash
    ollama pull llama3
    ```

3.  **安裝 Python 函式庫**
    本專案需要 LangChain 及其他相關函式庫。請執行以下指令安裝：
    ```bash
    pip install langchain langchain-community faiss-cpu
    ```
    若要執行微調，還需要安裝額外的函式庫：
    ```bash
    pip install torch datasets transformers peft trl bitsandbytes accelerate
    ```

---

## 如何執行範例

**重要：** 所有互動式腳本都需要由您自己在本機的終端機中執行。

### 1. 基本範例 (互動式問答)

這個腳本會直接與 `llama3` 模型進行對話。

```bash
python langchain_local_llm_example.py
```

### 2. RAG 範例 (藥品製造 SOP 問答系統)

這個範例模擬一個產業應用，作為藥廠操作員的輔助工具，能根據 `rag/source_documents` 資料夾中的 SOP 文件內容，回答生產相關問題。

**請注意：** `source_documents` 資料夾內的 SOP 文件內容為 AI 生成的虛構範例，僅用於功能展示，並非真實的工業標準作業程序。

![RAG Architecture](./assets/rag_diagram.svg)

**特色功能:**
*   **持久化向量儲存**: 首次執行時會自動建立 `vector_store` 資料夾來存放文件索引，加速後續啟動。
*   **答案來源追蹤**: 每次回答都會附上參考的 SOP 文件來源，方便使用者驗證資訊。

**執行步驟:**

1.  首先，切換到 RAG 的資料夾：
    ```bash
    cd rag
    ```
2.  執行 RAG 腳本：
    ```bash
    python rag_example.py
    ```
3.  程式啟動後，您可以試著提問，例如：
    *   `Metformin XR 錠劑的硬度目標是多少?`
    *   `Amoxicillin 口服懸液的 pH 值應在什麼範圍?`

### 3. 微調 (Fine-Tuning) 完整流程

此流程旨在建立一個具有獨特風格 (例如：海盜語氣) 的自訂模型。

![Fine-Tuning Process](./assets/finetune_diagram.svg)

(執行步驟與原版相同...)

---

## 處理模型的不可靠性：幻覺 (Hallucinations)

大型語言模型有時會「產生幻覺」，也就是捏造看似合理但實際上不正確或無從考證的資訊。在專業或關鍵應用中，處理幻覺至關重要。以下是本專案中採用的幾種策略：

### 1. 透過提示工程進行約束 (Prompt Engineering)

最直接的方式，就是透過精確的指令來限制模型的行為。在我們的 RAG 範例中，提示 (Prompt) 明確要求模型：

> "請只根據以下提供的標準作業程序 (SOP) 文件來精確回答問題。如果文件內容無法回答問題，請明確地說「根據現有SOP文件，無法找到相關資訊」。"

這個指令大幅降低了模型在知識範圍外自由發揮的可能性。

### 2. 提供答案來源進行溯源 (Grounding with Citations)

信任來自於驗證。如果模型只給出答案，我們無從得知其依據。新版的 RAG 範例在提供答案的同時，會一併列出它參考了哪些 `source_documents` 中的文件。

這讓使用者可以快速追溯到原始資訊，自行判斷答案的可靠性。這是建立可信賴 AI 系統的關鍵一步。

### 3. 透過微調深化領域知識 (Fine-Tuning)

雖然我們的微調範例是為了好玩的「海盜語氣」，但在產業應用上，微調是讓模型深度學習特定領域知識、術語和溝通風格的強大工具。

一個經過大量內部文件 (如：維修手冊、病例報告、法律條文) 微調的模型，會因為更「熟悉」該領域的知識，而較不容易產生與該領域無關的幻覺。

### 4. 透過使用者反饋進行標註與改進 (Labeling and Improvement via User Feedback)

預防幻覺的機制並非萬無一失。因此，建立一個讓使用者可以輕鬆回報錯誤的機制，是讓系統能「持續進化」的關鍵。這被稱為**人工回饋循環 (Human-in-the-Loop Feedback)**。

在新版的 RAG 範例中，我們加入了一個簡單的標註功能：
*   在模型回答後，您可以輸入 `flag` 指令。
*   程式會自動將該次對話的「問題」、「模型的錯誤答案」和「參考的上下文」完整記錄到 `rag/hallucination_log.jsonl` 檔案中。

這個日誌檔案非常有價值，它可以幫助您：
1.  **分析失敗案例**：了解系統是在哪個環節出錯（是文件檢索不準確，還是模型理解錯誤？）。
2.  **建立高品質的微調資料集**：這些被標註的「問題-錯誤答案」組合，是未來進行微調、讓模型從錯誤中學習的最佳教材，能有效提升模型在特定領域的準確度。
