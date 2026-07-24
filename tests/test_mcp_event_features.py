import pytest
import json
from unittest.mock import MagicMock, patch
from pico.mcp.server import handle_call_tool, handle_list_tools

@pytest.mark.anyio
async def test_mcp_list_tools_contains_new_tools():
    """新規ツールが list_tools に登録されているか確認"""
    tools = await handle_list_tools()
    tool_names = [t.name for t in tools]
    assert "get_perception_status" in tool_names
    assert "configure_event_filter" in tool_names

@pytest.mark.anyio
@patch("pico.mcp.server.get_perception")
@patch("pico.mcp.server.get_ptz")
async def test_mcp_get_perception_status(mock_get_ptz, mock_get_perception):
    """get_perception_status ツールの呼び出し検証"""
    mock_perception = MagicMock()
    mock_get_perception.return_value = mock_perception
    mock_perception.get_perception_status_data.return_value = {
        "engine_status": "RUNNING",
        "fps": 29.5,
        "active_track_count": 1,
        "cooldown_sec": 5.0,
        "recent_events": []
    }

    res = await handle_call_tool("get_perception_status", {})
    assert len(res) == 1
    data = json.loads(res[0].text)
    assert data["engine_status"] == "RUNNING"
    assert data["fps"] == 29.5

@pytest.mark.anyio
@patch("pico.mcp.server.get_perception")
@patch("pico.mcp.server.get_ptz")
async def test_mcp_configure_event_filter(mock_get_ptz, mock_get_perception):
    """configure_event_filter ツールの呼び出し検証"""
    mock_perception = MagicMock()
    mock_get_perception.return_value = mock_perception
    mock_perception.configure_event_filter_data.return_value = {
        "cooldown_sec": 10.0,
        "allowed_classes": ["person"],
        "recent_events": []
    }

    res = await handle_call_tool("configure_event_filter", {
        "cooldown_sec": 10.0,
        "allowed_classes": ["person"]
    })
    assert len(res) == 1
    assert "Success: Event filter configured" in res[0].text
    mock_perception.configure_event_filter_data.assert_called_once_with(10.0, ["person"])
