import asyncio
import numpy as np
from unittest.mock import AsyncMock, MagicMock
from pico.cognition import CognitionEngine

def test_cognition_engine_event_processing() -> None:
    # クライアントおよびデータベースのモック作成
    # 探索ルール "a person wearing a hat" に合致する ID: 5 を返すJSONを模擬応答とする
    mock_vlm = MagicMock()
    mock_vlm.analyze_scene = AsyncMock(return_value='{"lock_on_id": 5}')
    
    mock_memory = MagicMock()
    mock_memory.search.return_value = [
        {"title": "防犯ルール", "content": "夜間の不審者に注意。"}
    ]

    mock_ptz = MagicMock()
    mock_ptz.target_rule = "a person wearing a hat"
    mock_ptz.lock_on_id = None

    # エンジンの初期化
    engine = CognitionEngine(vlm_client=mock_vlm, memory_store=mock_memory, ptz_controller=mock_ptz)
    
    # 擬似イベント（検出された人物情報とフレーム画像）の作成
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    event = {
        "frame": dummy_frame,
        "detections": [
            {
                "track_id": 5,
                "bbox": [10, 10, 50, 50],
                "class_name": "person"
            },
            {
                "track_id": 9,
                "bbox": [60, 60, 90, 90],
                "class_name": "person"
            }
        ]
    }
    
    async def run_test():
        task = asyncio.create_task(engine.run())
        
        # イベントをキューに追加
        await engine.trigger_event(event)
        
        # 処理が走るのを少し待つ
        await asyncio.sleep(0.1)
        
        # モックの呼び出し検証
        mock_memory.search.assert_called_with("person", limit=1)
        
        # VLMに投げられたプロンプトの中身を検証
        called_args, called_kwargs = mock_vlm.analyze_scene.call_args
        prompt_arg = called_args[1]
        assert "a person wearing a hat" in prompt_arg
        assert "夜間の不審者に注意" in prompt_arg
        
        # lock_on_id が正常に mock_ptz にフィードバックされたか検証
        assert mock_ptz.lock_on_id == 5

        # エンジンを停止
        await engine.stop()
        await task

    asyncio.run(run_test())


def test_cognition_engine_hallucination_prevention() -> None:
    # 画面上の detections に存在しない ID（例: ID 99）をVLMが返してきた場合、ハルシネーションと判定して無視することを確認
    mock_vlm = MagicMock()
    mock_vlm.analyze_scene = AsyncMock(return_value='{"lock_on_id": 99}')
    
    mock_memory = MagicMock()
    mock_memory.search.return_value = []

    mock_ptz = MagicMock()
    mock_ptz.target_rule = "a person wearing a hat"
    mock_ptz.lock_on_id = None

    engine = CognitionEngine(vlm_client=mock_vlm, memory_store=mock_memory, ptz_controller=mock_ptz)
    
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    event = {
        "frame": dummy_frame,
        "detections": [
            {
                "track_id": 5,
                "bbox": [10, 10, 50, 50],
                "class_name": "person"
            }
        ]
    }
    
    async def run_test():
        task = asyncio.create_task(engine.run())
        await engine.trigger_event(event)
        await asyncio.sleep(0.1)
        
        # ID 99 は detections に無いため、lock_on_id に反映されてはならない
        assert mock_ptz.lock_on_id is None

        await engine.stop()
        await task

    asyncio.run(run_test())
