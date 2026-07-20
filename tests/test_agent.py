import pytest
import numpy as np
import asyncio
from unittest.mock import MagicMock, AsyncMock
from pico.agent import SurveillanceAgent, AgentState
from pico.perception_buffer import PerceptionBuffer

@pytest.fixture
def mock_agent_setup():
    mock_ptz = MagicMock()
    mock_ptz.lock_on_id = None
    
    mock_vlm = MagicMock()
    # プランナーとしてのLLMテキスト応答をモック
    # 1回目はツール実行、2回目はENDとなるように設定して無限ループを防止
    mock_vlm.analyze_scene = AsyncMock(
        side_effect=[
            '{"action": "execute", "tool_name": "set_tracking_target", "args": {"track_id": 2}}',
            '{"action": "end", "reason": "Target already locked"}'
        ]
    )
    
    mock_memory = MagicMock()
    mock_memory.search.return_value = []
    
    from pico.agent_tools import AgentTools
    tools = AgentTools(ptz_controller=mock_ptz, vlm_client=mock_vlm, memory_store=mock_memory)
    perception = PerceptionBuffer()
    
    agent = SurveillanceAgent(tools=tools, perception_buffer=perception, ollama_client=mock_vlm)
    return agent, tools, mock_vlm, mock_ptz

@pytest.mark.anyio
async def test_agent_graph_execution_flow(mock_agent_setup) -> None:
    agent, tools, mock_vlm, mock_ptz = mock_agent_setup
    
    # ターゲットを検知したとする
    # bbox: [x, y, w, h]
    from pico.tracker import TrackedObject
    dummy_tracked = [TrackedObject(track_id=2, class_id=0, bbox=(10, 10, 100, 100), confidence=0.90)]
    agent.perception.update(dummy_tracked, (480, 640, 3))
    
    # 1ステップ駆動
    dummy_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    res = await agent.step(agent.perception.get_active_tracks_json(), dummy_frame)
    
    # ロックオンIDが2にフィードバックされたか検証
    assert mock_ptz.lock_on_id == 2
    assert res["tool_output"] == "Plan ended or no actions requested."

@pytest.mark.anyio
async def test_agent_epoch_guard_failsafe(mock_agent_setup) -> None:
    agent, tools, mock_vlm, mock_ptz = mock_agent_setup
    
    # ツール実行時のエポックが古いまま（ユーザー緊急介入によるエポック増加が発生した場合）のシミュレーション
    state: AgentState = {
        "active_tracks": [],
        "active_tracks_text": "No tracks",
        "lockon_mode": "auto",
        "target_track_id": None,
        "agent_goal": "Test goal",
        "recalled_knowledge": [],
        "conversation_history": [],
        "state_epoch": 0,  # 計画開始時のエポック
        "next_step": "execute",
        "next_tool_call": {
            "action": "execute",
            "tool_name": "set_tracking_target",
            "args": {"track_id": 99}
        },
        "tool_output": ""
    }

    # ツール実行前に、ユーザー介入イベント (Barge-In) が発生してエージェント側のエポックが 1 にジャンプしたとする
    await agent.update_by_user_barge_in(7) # ユーザー指定ID 7 に強制固定
    
    # ツール実行ノードを呼び出す
    res = await agent.node_execute_tool(state)
    
    # 古いエポック (0) でのツール実行コミットが破棄され、アボートしたことを確認
    assert res["tool_output"] == "Aborted due to epoch mismatch."
    # 物理カメラのロックオン先はユーザー指定の 7 で維持されていること
    assert mock_ptz.lock_on_id == 7
