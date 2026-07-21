import asyncio
import logging
import re
import json
import cv2
import numpy as np
from typing import Dict, Any, Optional, List
from pico.ollama_client import OllamaVisionClient
from pico.memory import MemoryStore

logger = logging.getLogger(__name__)

class CognitionEngine:
    """非同期の認知処理ループを実行し、VLM (Ollama) と長期記憶 (MemoryStore) を用いたシーン理解および能動的ロックオン（Barge-In）を処理するクラス。"""

    def __init__(self, vlm_client: OllamaVisionClient, memory_store: MemoryStore, ptz_controller: Optional[Any] = None) -> None:
        self.vlm: OllamaVisionClient = vlm_client
        self.memory: MemoryStore = memory_store
        self.ptz = ptz_controller
        
        # 非同期イベントキュー
        self.queue: asyncio.Queue = asyncio.Queue()
        self.stop_event: asyncio.Event = asyncio.Event()

    async def trigger_event(self, event: Dict[str, Any]) -> None:
        """認知イベント（フレーム画像、検出オブジェクトリストなど）をキューに投入する。
        
        反射ループ（スレッド）などから呼び出す。
        """
        await self.queue.put(event)

    def trigger_event_from_thread(self, event: Dict[str, Any], loop: asyncio.AbstractEventLoop) -> None:
        """別スレッドから非同期キューへイベントをスレッド安全に投入する。"""
        asyncio.run_coroutine_threadsafe(self.trigger_event(event), loop)

    def _draw_annotations(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """画像上の各検出オブジェクトに対して、ID番号と矩形（アノテーション）を描画する。"""
        annotated = frame.copy()
        for det in detections:
            track_id = det.get("track_id", -1)
            bbox = det.get("bbox")
            if bbox is None:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                annotated, f"ID: {track_id}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
        return annotated

    def _parse_lock_on_id(self, text: str) -> Optional[int]:
        """VLMのテキスト応答からJSON表現を抽出し、lock_on_idを取得する。"""
        try:
            # 波括弧で囲まれた最小のJSON部分を正規表現で探す
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                val = data.get("lock_on_id")
                if val is not None:
                    # nullやNoneの場合はNoneを返す
                    if str(val).lower() in ("null", "none"):
                        return None
                    return int(val)
        except Exception as e:
            logger.error(f"🧠 [VLM Parser Error] Failed to parse lock_on_id from: '{text}'. Error: {e}")
        return None

    async def run(self) -> None:
        """キューからイベントを取り出し、順次非同期で解析およびロックオン指示を行うループ。"""
        logger.info("CognitionEngine loop started.")
        while not self.stop_event.is_set():
            try:
                # タイムアウト付きでキューから取得
                event = await asyncio.wait_for(self.queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            
            try:
                # 1. イベントのパース
                frame: np.ndarray = event["frame"]
                detections: List[Dict[str, Any]] = event.get("detections", [])
                
                if not detections:
                    continue

                # 現在の探索ルールを取得 (PTZControllerから動的取得、なければ環境変数またはデフォルト)
                target_rule = "a person wearing a hat"
                if self.ptz and hasattr(self.ptz, "target_rule"):
                    target_rule = self.ptz.target_rule

                logger.info(f"🧠 [Cognition Engine] Analyzing {len(detections)} targets for rule: '{target_rule}'")

                # 2. VLM用のアノテーション付き画像の作成
                annotated_frame = self._draw_annotations(frame, detections)

                # 3. 長期記憶の想起 (FAG)
                # 検出リスト内の最初のクラス名などをキーに記憶を検索
                main_class = detections[0].get("class_name", "person")
                memories = self.memory.search(main_class, limit=1)
                memory_context = ""
                if memories:
                    memory_context = f"\n[想起された防犯情報]:\n- Title: {memories[0]['title']}\n- Content: {memories[0]['content']}\n"

                # 4. VLMプロンプトの構築（ID枠の特定とJSON強制指示）
                prompt = (
                    f"あなたは防犯ドームカメラ映像を監視する認知AIエージェントです。\n"
                    f"現在、カメラ画像上に {len(detections)} 個の検出対象が赤い枠とID表示（例: 'ID: 1'）で描画されています。\n"
                    f"{memory_context}\n"
                    f"アノテーションされた画像を確認し、以下の[探索ルール]に最も合致するオブジェクトのID番号を特定してください。\n\n"
                    f"[探索ルール]: {target_rule}\n\n"
                    f"合致するIDが画像中に存在するならば、必ず以下のJSONフォーマットのみで返答してください。余計な挨拶や説明、思考の出力は一切禁止します。\n"
                    f'{{"lock_on_id": ID番号}}\n\n'
                    f"合致するIDが存在しない、または判断できない場合は、以下のJSONのみを返答してください。\n"
                    f'{{"lock_on_id": null}}\n'
                )

                # 5. Ollama VLM 推論の非同期実行
                analysis_result = await self.vlm.analyze_scene(annotated_frame, prompt)
                
                if analysis_result:
                    logger.info(f"🧠 [VLM Cognition Output]:\n{analysis_result.strip()}")
                    
                    # 6. JSONの抽出とロックオン指示の反映
                    lock_on_id = self._parse_lock_on_id(analysis_result)
                    
                    if lock_on_id is not None:
                        # 検出されたIDが現在追跡対象リスト内にあるかを二重防御で確認
                        valid_ids = [d.get("track_id") for d in detections]
                        if lock_on_id in valid_ids:
                            logger.info(f"🎯 [Barge-In Lock-On Request] VLM detected matching target! Requesting Lock-On for ID: {lock_on_id}")
                            if self.ptz:
                                self.ptz.lock_on_id = lock_on_id
                        else:
                            logger.warning(f"🧠 [VLM Hallucination Prevented] VLM suggested lock-on ID {lock_on_id}, but it is not active in tracker detections: {valid_ids}")
                    else:
                        logger.debug("🧠 [VLM Cognition] No matching targets detected for rule.")
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
