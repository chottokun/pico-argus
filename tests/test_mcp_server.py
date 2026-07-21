import pytest
from unittest.mock import patch
import mcp.types as types
from pico.mcp.server import handle_list_tools, handle_call_tool

@pytest.mark.anyio
@patch("pico.mcp.server.perception")
@patch("pico.mcp.server.ptz")
@patch("pico.mcp.server.memory")
async def test_handle_list_tools(mock_memory, mock_ptz, mock_perception):
    tools = await handle_list_tools()
    
    assert len(tools) == 4
    tool_names = [t.name for t in tools]
    assert "get_active_tracks" in tool_names
    assert "analyze_crop_image" in tool_names
    assert "set_tracking_target" in tool_names
    assert "search_wiki" in tool_names

@pytest.mark.anyio
@patch("pico.mcp.server.perception")
@patch("pico.mcp.server.ptz")
@patch("pico.mcp.server.memory")
async def test_call_get_active_tracks(mock_memory, mock_ptz, mock_perception):
    # Setup
    mock_perception.get_tracks_data.return_value = [
        {"track_id": 1, "class": "person", "bbox": [10, 20, 30, 40], "confidence": 0.9}
    ]

    # Run
    res = await handle_call_tool("get_active_tracks", {})

    # Assert
    assert len(res) == 1
    assert isinstance(res[0], types.TextContent)
    assert "track_id" in res[0].text
    mock_perception.get_tracks_data.assert_called_once()

@pytest.mark.anyio
@patch("pico.mcp.server.perception")
@patch("pico.mcp.server.ptz")
@patch("pico.mcp.server.memory")
async def test_call_analyze_crop_image(mock_memory, mock_ptz, mock_perception):
    # Setup
    mock_perception.analyze_crop_data.return_value = {
        "status": "success",
        "response": "A small dog sitting."
    }

    # Run
    res = await handle_call_tool("analyze_crop_image", {"track_id": 1, "query": "What is this?"})

    # Assert
    assert len(res) == 1
    assert res[0].text == "A small dog sitting."
    mock_perception.analyze_crop_data.assert_called_once_with(1, "What is this?")

@pytest.mark.anyio
@patch("pico.mcp.server.perception")
@patch("pico.mcp.server.ptz")
@patch("pico.mcp.server.memory")
async def test_call_set_tracking_target_null(mock_memory, mock_ptz, mock_perception):
    # Setup
    # mock lockon task
    global lockon_task
    
    # Run
    res = await handle_call_tool("set_tracking_target", {"track_id": None})

    # Assert
    assert len(res) == 1
    assert "解除" in res[0].text
    mock_ptz.stop_lockon.assert_called_once()
    mock_ptz.emergency_stop.assert_called_once()

@pytest.mark.anyio
@patch("pico.mcp.server.perception")
@patch("pico.mcp.server.ptz")
@patch("pico.mcp.server.memory")
async def test_call_search_wiki(mock_memory, mock_ptz, mock_perception):
    # Setup
    mock_memory.search_knowledge_data.return_value = [
        {"filepath": "wiki/test.md", "title": "Test Title", "content": "Test content", "score": 1.0, "provenance": {"source": "test", "confidence": "high"}}
    ]

    # Run
    res = await handle_call_tool("search_wiki", {"query": "Test"})

    # Assert
    assert len(res) == 1
    assert "Test Title" in res[0].text
    mock_memory.search_knowledge_data.assert_called_once_with("Test")
