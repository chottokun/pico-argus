import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import mcp.types as types
from pico.mcp.server import handle_list_tools, handle_call_tool

@pytest.mark.anyio
@patch("pico.mcp.server.get_perception")
@patch("pico.mcp.server.get_ptz")
@patch("pico.mcp.server.get_memory")
async def test_handle_list_tools(mock_get_memory, mock_get_ptz, mock_get_perception):
    tools = await handle_list_tools()
    
    assert len(tools) == 4
    tool_names = [t.name for t in tools]
    assert "get_active_tracks" in tool_names
    assert "analyze_crop_image" in tool_names
    assert "set_tracking_target" in tool_names
    assert "search_wiki" in tool_names

@pytest.mark.anyio
@patch("pico.mcp.server.get_perception")
@patch("pico.mcp.server.get_ptz")
@patch("pico.mcp.server.get_memory")
@patch("pico.mcp.server.get_yolo_semaphore")
async def test_call_get_active_tracks(mock_get_yolo_sem, mock_get_memory, mock_get_ptz, mock_get_perception):
    # Setup
    mock_perception = MagicMock()
    mock_get_perception.return_value = mock_perception
    mock_perception.get_tracks_data.return_value = [
        {"track_id": 1, "class": "person", "bbox": [10, 20, 30, 40], "confidence": 0.9}
    ]

    mock_sem = AsyncMock()
    mock_get_yolo_sem.return_value = mock_sem

    # Run
    res = await handle_call_tool("get_active_tracks", {})

    # Assert
    assert len(res) == 1
    assert isinstance(res[0], types.TextContent)
    assert "track_id" in res[0].text
    mock_perception.get_tracks_data.assert_called_once()
    mock_sem.__aenter__.assert_called_once()

@pytest.mark.anyio
@patch("pico.mcp.server.get_perception")
@patch("pico.mcp.server.get_ptz")
@patch("pico.mcp.server.get_memory")
@patch("pico.mcp.server.get_vlm_semaphore")
@patch("pico.mcp.server.get_vlm_rpm_limiter")
async def test_call_analyze_crop_image(mock_get_rpm, mock_get_vlm_sem, mock_get_memory, mock_get_ptz, mock_get_perception):
    # Setup
    mock_perception = MagicMock()
    mock_get_perception.return_value = mock_perception
    mock_perception.analyze_crop_data.return_value = {
        "status": "success",
        "response": "A small dog sitting."
    }

    mock_sem = AsyncMock()
    mock_get_vlm_sem.return_value = mock_sem

    mock_rpm = AsyncMock()
    mock_get_rpm.return_value = mock_rpm

    # Run
    res = await handle_call_tool("analyze_crop_image", {"track_id": 1, "query": "What is this?"})

    # Assert
    assert len(res) == 1
    assert res[0].text == "A small dog sitting."
    mock_perception.analyze_crop_data.assert_called_once_with(1, "What is this?")
    mock_rpm.acquire.assert_called_once()
    mock_sem.__aenter__.assert_called_once()

@pytest.mark.anyio
@patch("pico.mcp.server.get_perception")
@patch("pico.mcp.server.get_ptz")
@patch("pico.mcp.server.get_memory")
async def test_call_set_tracking_target_null(mock_get_memory, mock_get_ptz, mock_get_perception):
    # Setup
    mock_ptz = MagicMock()
    mock_get_ptz.return_value = mock_ptz
    
    # Run
    res = await handle_call_tool("set_tracking_target", {"track_id": None})

    # Assert
    assert len(res) == 1
    assert "解除" in res[0].text
    mock_ptz.stop_lockon.assert_called_once()
    mock_ptz.emergency_stop.assert_called_once()

@pytest.mark.anyio
@patch("pico.mcp.server.get_perception")
@patch("pico.mcp.server.get_ptz")
@patch("pico.mcp.server.get_memory")
async def test_call_search_wiki(mock_get_memory, mock_get_ptz, mock_get_perception):
    # Setup
    mock_memory = MagicMock()
    mock_get_memory.return_value = mock_memory
    mock_memory.search_knowledge_data.return_value = [
        {"filepath": "wiki/test.md", "title": "Test Title", "content": "Test content", "score": 1.0, "provenance": {"source": "test", "confidence": "high"}}
    ]

    # Run
    res = await handle_call_tool("search_wiki", {"query": "Test"})

    # Assert
    assert len(res) == 1
    assert "Test Title" in res[0].text
    mock_memory.search_knowledge_data.assert_called_once_with("Test")
