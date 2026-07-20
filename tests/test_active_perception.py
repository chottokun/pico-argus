import pytest
import numpy as np
import json
from unittest.mock import MagicMock, AsyncMock
from pico.agent import SurveillanceAgent, AgentState
from pico.perception_buffer import PerceptionBuffer
from pico.agent_tools import AgentTools

@pytest.fixture
def mock_surveillance_setup():
    mock_ptz = MagicMock()
    mock_ptz.lock_on_id = None
    
    mock_vlm = MagicMock()
    
    # 段階的な LLM の自律プラン応答を side_effect でシミュレート
    # 1周目: 警戒エリアに入った物体(ID 5)について trigger_visual_query をキック
    # 2周目: 解析結果を受けて store_memory で知識蓄積をキック
    # 3周目: 完了したため end
    mock_vlm.analyze_scene = AsyncMock(
        side_effect=[
            # 1周目のプランニング
            '{"action": "execute", "tool_name": "trigger_visual_query", "args": {"track_id": 5, "prompt": "Identify object detail"}}',
            # 1周目の trigger_visual_query に対する VLM クエリ実行 (VLMが物体を特定)
            'The target ID 5 is a yellow courier package left at the door.',
            # 2周目のプランニング
            '{"action": "execute", "tool_name": "store_memory", "args": {"title": "yellow_package_id5", "content": "A yellow courier package was left at the door.", "tags": "package warning"}}',
            # 3周目のプランニング
            '{"action": "end", "reason": "Investigation completed and stored"}'
        ]
    )
    
    mock_memory = MagicMock()
    mock_memory.search.return_value = []
    
    tools = AgentTools(ptz_controller=mock_ptz, vlm_client=mock_vlm, memory_store=mock_memory)
    
    # 警戒エリア x: [0.1, 0.9], y: [0.4, 0.9]
    perception = PerceptionBuffer(warning_zone_x=(0.1, 0.9), warning_zone_y=(0.4, 0.9))
    
    agent = SurveillanceAgent(tools=tools, perception_buffer=perception, ollama_client=mock_vlm)
    return agent, tools, mock_vlm, mock_memory

@pytest.mark.anyio
async def test_active_zoom_scan_full_sequence(mock_surveillance_setup) -> None:
    agent, tools, mock_vlm, mock_memory = mock_surveillance_setup
    
    # 1. 警戒エリア内に低確信度の物体を検出した状況の再現
    # bbox: [x, y, w, h] (cx=0.5, cy=0.75 -> 警戒ゾーン内部)
    from pico.tracker import TrackedObject
    dummy_tracked = [TrackedObject(track_id=5, class_id=24, bbox=(290, 330, 60, 60), confidence=0.52)]
    
    # フレームシェイプ (480x640x3)
    frame_shape = (480, 640, 3)
    agent.perception.update(dummy_tracked, frame_shape)
    
    # 知覚メタデータに warning_zone_triggered が入っているか検証
    tracks_json = agent.perception.get_active_tracks_json()
    assert tracks_json[0]["warning_zone_triggered"] is True
    assert "[⚠️WARNING ZONE DETECTED]" in agent.perception.get_active_tracks_text()

    # 2. 1ステップ (自律能動知覚ループの1周) を実行
    dummy_frame = np.zeros(frame_shape, dtype=np.uint8)
    
    # stepを実行して、LangGraphの最初から最後までを駆動
    res = await agent.step(tracks_json, dummy_frame)
    
    # 最終的な tool_output が 3周目の終了メッセージになっていること
    assert res["tool_output"] == "Plan ended or no actions requested."
    
    # MemoryStore への add_entry が正しく呼び出されていること
    mock_memory.add_entry.assert_called_once()
    called_args, called_kwargs = mock_memory.add_entry.call_args
    assert called_kwargs["title"] == "yellow_package_id5"
    assert "yellow_package_id5" in called_kwargs["filepath"]
    assert "A yellow courier package was left at the door." in called_kwargs["content"]
