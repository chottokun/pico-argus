import asyncio
import logging
import numpy as np
from typing import Dict, Any
from pico.ollama_client import OllamaVisionClient
from pico.memory import MemoryStore

logger = logging.getLogger(__name__)

class CognitionEngine:
    """非同期の認知処理ループを実行し、VLM (Ollama) と長期記憶 (MemoryStore) を用いたシーン理解を処理するクラス。"""

    def __init__(self, vlm_client: OllamaVisionClient, memory_store: MemoryStore) -> None:
        self.vlm: OllamaVisionClient = vlm_client
        self.memory: MemoryStore = memory_store
        
        # 非同期イベントキュー
        self.queue: asyncio.Queue = asyncio.Queue()
        self.stop_event: asyncio.Event = asyncio.Event()

    async def trigger_event(self, event: Dict[str, Any]) -> None:
        """認知イベント（フレーム画像、追跡対象IDなど）をキューに投入する。
        
        反射ループ（スレッド）などから呼び出す。
        """
        await self.queue.put(event)
        logger.debug(f"Cognition event triggered for class: {event.get('class_name')}")

    def trigger_event_from_thread(self, event: Dict[str, Any], loop: asyncio.AbstractEventLoop) -> None:
        """別スレッドから非同期キューへイベントをスレッド安全に投入する。"""
        asyncio.run_coroutine_threadsafe(self.trigger_event(event), loop)

    async def run(self) -> None:
        """キューからイベントを取り出し、順次非同期で解析を行うループ。"""
        logger.info("CognitionEngine loop started.")
        while not self.stop_event.is_set():
            try:
                # タイムアウト付きでキューから取得し、停止シグナルを定期チェック可能にする
                event = await asyncio.wait_for(self.queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            
            try:
                # 1. イベントのパース
                class_name: str = event.get("class_name", "unknown")
                track_id: int = event.get("track_id", -1)
                frame: np.ndarray = event["frame"]

                logger.info(f"🧠 [Cognition Engine] Starting analysis for object: {class_name}#{track_id}")

                # 2. 長期記憶の想起 (FAG)
                # 検出されたオブジェクトカテゴリ名に基づき、関連ドキュメントを想起する
                memories = self.memory.search(class_name, limit=1)
                memory_context = ""
                if memories:
                    memory_context = f"\n[想起された関連記憶]:\n- Title: {memories[0]['title']}\n- Content: {memories[0]['content']}\n"
                    logger.info(f"🧠 [Memory Recalled] Title: {memories[0]['title']}")

                # 3. VLM プロンプトの構築
                # 想起した記憶があれば文脈として追加する
                prompt = (
                    f"あなたは防犯エッジAIの認知ループエージェントです。\n"
                    f"現在、カメラ画像の中に {class_name} (ID: {track_id}) を検出しました。\n"
                    f"{memory_context}"
                    f"画像内の様子を詳細に分析し、何をしているか、および懸念事項や安全上のルールをふまえた警告があるかを1-2文の日本語で要約して出力してください。"
                )

                # 4. Ollama VLM 推論の非同期実行
                # 重い推論だが非同期呼び出しなので反射ループスレッドをブロッキングしない
                analysis_result = await self.vlm.analyze_scene(frame, prompt)
                
                if analysis_result:
                    # 5. 受動的ログモードとしての結果出力
                    logger.info(f"🧠 [VLM Scene Summary] Output:\n{analysis_result.strip()}")
                else:
                    logger.warning("🧠 [VLM Analysis Failed] VLM returned empty response.")

            except Exception as e:
                logger.error(f"Error processing cognition event: {e}")
            finally:
                self.queue.task_done()

        logger.info("CognitionEngine loop stopped.")

    async def stop(self) -> None:
        """認知ループを安全に停止する。"""
        self.stop_event.set()
