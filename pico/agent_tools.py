import logging
import cv2
import numpy as np
from typing import Optional, Any, Dict, List
from pico.onvif_client import PTZController
from pico.ollama_client import OllamaVisionClient
from pico.memory import MemoryStore

logger = logging.getLogger(__name__)

class AgentTools:
    """LLMエージェントが自律プランの中で呼び出す、物理サーボ、能動知覚（VLM）、長期記憶のツールセット。

    V2設計 §2 および §3 のツール仕様に準拠。
    """

    def __init__(
        self,
        ptz_controller: PTZController,
        vlm_client: OllamaVisionClient,
        memory_store: MemoryStore
    ) -> None:
        self.ptz: PTZController = ptz_controller
        self.vlm: OllamaVisionClient = vlm_client
        self.memory: MemoryStore = memory_store
        
        # クロップ画像用の共有変数（最新の生フレームを外部から一時的にセットしてもらう）
        self.last_raw_frame: Optional[np.ndarray] = None
        self.last_active_tracks: List[Dict[str, Any]] = []

    def set_tracking_target(self, track_id: int) -> str:
        """指定された track_id のオブジェクトを最優先で追従・ロックオンターゲットにする。

        Args:
            track_id (int): 追尾対象のID
        """
        self.ptz.lock_on_id = track_id
        logger.info(f"🔧 [Tool Call: set_tracking_target] Target set to ID: {track_id}")
        return f"Successfully locked physical tracking onto target ID {track_id}."

    def clear_tracking_target(self) -> str:
        """ロックオンターゲットを解除し、デフォルト（最大面積）の自動追従モードに戻す。"""
        self.ptz.lock_on_id = None
        logger.info("🔧 [Tool Call: clear_tracking_target] Lock-on cleared. Reverted to default tracking.")
        return "Lock-on target cleared. System returned to default automatic tracking mode."

    async def trigger_visual_query(self, track_id: int, prompt: str) -> str:
        """指定されたIDのバウンディングボックス周辺をクロップ（ソフトウェアズーム）し、VLMに入力して意味解析を行う。

        Args:
            track_id (int): 解析対象のID
            prompt (str): VLMへの意味解析用のビジュアルクエリプロンプト
        """
        logger.info(f"🔧 [Tool Call: trigger_visual_query] Target ID: {track_id}, Prompt: '{prompt}'")
        
        if self.last_raw_frame is None:
            return "Error: No camera frame available in perception buffer to analyze."

        # 対象オブジェクトのbboxを取得
        target_info = next((t for t in self.last_active_tracks if t["track_id"] == track_id), None)
        if not target_info:
            return f"Error: Target ID {track_id} is not currently visible in the scene. Visible IDs: {[t['track_id'] for t in self.last_active_tracks]}"

        x, y, w, h = target_info["bbox"]
        img_h, img_w = self.last_raw_frame.shape[:2]

        # 境界クランプしつつ、周辺を少し広め(30%余白)にクロップしてソフトウェアズーム
        padding_w = int(w * 0.3)
        padding_h = int(h * 0.3)
        
        x1 = max(0, x - padding_w)
        y1 = max(0, y - padding_h)
        x2 = min(img_w, x + w + padding_w)
        y2 = min(img_h, y + h + padding_h)

        cropped_img = self.last_raw_frame[y1:y2, x1:x2]
        if cropped_img.size == 0:
            return "Error: Failed to crop the target region. Image size is zero."

        # VLMによるオンデマンド解析の実行
        logger.info(f"👁️ [Visual Query] Sending cropped image of ID {track_id} to VLM (Ollama)...")
        analysis_result = await self.vlm.analyze_scene(cropped_img, prompt)
        
        if not analysis_result:
            return "Error: VLM returned an empty response or analysis failed."
            
        logger.info(f"👁️ [Visual Query Result]: {analysis_result}")
        return f"VLM Analysis Result for ID {track_id}:\n{analysis_result}"

    def recall_memory(self, query: str) -> str:
        """長期記憶 (SQLite FTS5 Trigram Wiki) から過去の関連知識を想起（検索）する。

        Args:
            query (str): 想起キーとなるテキストやタグ
        """
        logger.info(f"🔧 [Tool Call: recall_memory] Query: '{query}'")
        results = self.memory.search(query, limit=2)
        if not results:
            return f"No memories recalled for query: '{query}'."

        formatted_memories = []
        for idx, r in enumerate(results):
            formatted_memories.append(
                f"Memory {idx+1}: {r['title']} (tags: {', '.join(r['tags'])})\n"
                f"Source: {r['provenance']['source']} (Confidence: {r['provenance']['confidence']})\n"
                f"Content:\n{r['content']}"
            )
        return "\n\n".join(formatted_memories)

    def store_memory(self, title: str, content: str, tags: str) -> str:
        """新しい情報や分析結果を OKF (Open Knowledge Format) に準拠した構造で長期記憶に書き込む。

        Args:
            title (str): 知識のタイトル
            content (str): 本文
            tags (str): スペース区切りのタグ
        """
        logger.info(f"🔧 [Tool Call: store_memory] Title: '{title}', Tags: '{tags}'")
        try:
            # OKF Markdown構造を作成
            okf_content = (
                f"---\n"
                f"title: {title}\n"
                f"tags: {tags}\n"
                f"doc_type: knowledge\n"
                f"provenance_source: Agent V2 active_perception\n"
                f"provenance_confidence: High\n"
                f"---\n\n"
                f"{content}"
            )
            filepath = f"wiki/auto_{title.lower().replace(' ', '_')}.md"
            
            # MemoryStoreに格納
            self.memory.add_entry(
                filepath=filepath,
                doc_type="knowledge",
                title=title,
                tags=tags,
                content=okf_content
            )
            return f"Successfully stored new knowledge to '{filepath}'."
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return f"Error: Failed to store memory: {e}"
