# Archive Notes

- **移動日**: 2026-07-21
- **理由**: CLI先行開発 ➔ MCPサーバーラッピング設計への段階的移行

### 移動したファイルと役割

- **`archive/agent.py`**:
  - LangGraph `StateGraph` を用いて、カメラの状況を内省評価・プラン決定・ツール実行を行っていた意思決定グラフクラス。
- **`archive/agent_tools.py`**:
  - LLMエージェントが呼び出すためのツール（`set_tracking_target`, `clear_tracking_target`, `trigger_visual_query`, `recall_memory`, `store_memory`）の具現化クラス。
- **`archive/cognition.py`**:
  - Ollama VLM に画像を送信して探索ルール（例：「帽子をかぶった人」）に合致するIDの特定を行っていた非同期認知処理エンジン。
- **`archive/perception_buffer.py`**:
  - YOLOおよびトラッカーの検出結果を蓄積し、警告ゾーンの判定や、LLMに渡すための可読テキストにフォーマットしていた知覚バッファ。
- **`archive/run_agent.py`**:
  - システムのエントリーポイント。RTSPストリームの受信、YOLO検出、トラッカー更新、PID制御（反射ループ）と、LangGraphエージェント（非同期認知ループ）を並行して動かしていたメインループ。

### 参照方法
Phase 3 で MCP 経由の自律脳エージェント（LangGraph等）を再度統合する際、これらのファイルのツール定義やプロンプト設計、Epochガードレールロジックを参考にします。
