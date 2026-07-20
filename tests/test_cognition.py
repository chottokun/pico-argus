import asyncio
import numpy as np
from unittest.mock import AsyncMock, MagicMock
from pico.cognition import CognitionEngine

def test_cognition_engine_event_processing() -> None:
    # クライアントおよびデータベースのモック作成
    mock_vlm = MagicMock()
    mock_vlm.analyze_scene = AsyncMock(return_value="男の人がカメラを見て立っています。")
    
    mock_memory = MagicMock()
    mock_memory.search.return_value = [
        {"title": "防犯ルール", "content": "夜間の不審者に注意。"}
    ]

    # エンジンの初期化
    engine = CognitionEngine(vlm_client=mock_vlm, memory_store=mock_memory)
    
    # 擬似イベント（検出された人物情報とフレーム画像）の作成
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    event = {
        "class_name": "person",
        "track_id": 1,
        "frame": dummy_frame
    }
    
    async def run_test():
        # バックグラウンドタスクを回すため、一時的にエンジンタスクを起動して終了させる設計にする
        task = asyncio.create_task(engine.run())
        
        # イベントをキューに追加
        await engine.trigger_event(event)
        
        # 処理が走るのを少し待つ
        await asyncio.sleep(0.1)
        
        # モックの呼び出し検証
        mock_memory.search.assert_called_with("person", limit=1)
        # 想起された記憶（夜間の不審者に注意）がプロンプトに含まれて推論に回されたか検証
        called_args, called_kwargs = mock_vlm.analyze_scene.call_args
        prompt_arg = called_args[1]
        assert "person" in prompt_arg
        assert "夜間の不審者に注意" in prompt_arg

        
        # エンジンを停止
        await engine.stop()
        await task

    asyncio.run(run_test())

