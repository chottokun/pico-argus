import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock
from pico.agent_tools import AgentTools

def test_agent_tools_tracking_control() -> None:
    mock_ptz = MagicMock()
    mock_ptz.lock_on_id = None
    
    mock_vlm = MagicMock()
    mock_memory = MagicMock()
    
    tools = AgentTools(ptz_controller=mock_ptz, vlm_client=mock_vlm, memory_store=mock_memory)
    
    # ロックオンの設定検証
    res = tools.set_tracking_target(5)
    assert "locked physical tracking onto target ID 5" in res
    assert mock_ptz.lock_on_id == 5
    
    # ロックオンの解除検証
    res = tools.clear_tracking_target()
    assert "Lock-on target cleared" in res
    assert mock_ptz.lock_on_id is None

@pytest.mark.anyio
async def test_agent_tools_trigger_visual_query() -> None:
    mock_ptz = MagicMock()
    
    mock_vlm = MagicMock()
    mock_vlm.analyze_scene = AsyncMock(return_value="Detected a black cap.")
    
    mock_memory = MagicMock()
    
    tools = AgentTools(ptz_controller=mock_ptz, vlm_client=mock_vlm, memory_store=mock_memory)
    
    # テスト用画像 (480x640x3) と知覚バッファ内のターゲット
    tools.last_raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tools.last_active_tracks = [
        {
            "track_id": 10,
            "bbox": [100, 100, 50, 60]  # x, y, w, h
        }
    ]
    
    # 正常パターンの検証
    res = await tools.trigger_visual_query(10, "Is the person wearing a hat?")
    assert "Detected a black cap." in res
    
    # クロップ引数の検証（analyze_scene の第1引数はクロップされた numpy 配列）
    mock_vlm.analyze_scene.assert_called_once()
    called_args, _ = mock_vlm.analyze_scene.call_args
    cropped_img_arg = called_args[0]
    
    # パディング50%追加時の元解像度アスペクト比維持のサイズ（幅100, 高さ120）になっていること
    assert cropped_img_arg.shape[0] == 120
    assert cropped_img_arg.shape[1] == 100

    # 存在しないターゲットID指定時のエラーハンドリング検証
    res_err = await tools.trigger_visual_query(99, "Query")
    assert "Error" in res_err

def test_agent_tools_memory_handling() -> None:
    mock_ptz = MagicMock()
    mock_vlm = MagicMock()
    
    mock_memory = MagicMock()
    mock_memory.search.return_value = [
        {
            "title": "家主ルール",
            "tags": ["owner", "takashi"],
            "content": "たかしさんは黄色い帽子を好む",
            "provenance": {"source": "rules.md", "confidence": "High"}
        }
    ]
    
    tools = AgentTools(ptz_controller=mock_ptz, vlm_client=mock_vlm, memory_store=mock_memory)
    
    # 想起 (Recall) の検証
    recall_res = tools.recall_memory("takashi")
    assert "たかしさんは黄色い帽子を好む" in recall_res
    mock_memory.search.assert_called_with("takashi", limit=2)
    
    # 保存 (Store) の検証
    store_res = tools.store_memory("new_info", "Testing content", "test tag")
    assert "Successfully stored new knowledge" in store_res
    mock_memory.add_entry.assert_called_once()
