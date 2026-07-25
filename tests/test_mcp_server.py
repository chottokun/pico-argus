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
    
    assert len(tools) == 8  # 新規2ツール追加により 8 つ
    tool_names = [t.name for t in tools]
    assert "get_active_tracks" in tool_names
    assert "analyze_crop_image" in tool_names
    assert "set_tracking_target" in tool_names
    assert "get_live_snapshot" in tool_names
    assert "search_wiki" in tool_names
    assert "write_wiki" in tool_names
    assert "get_perception_status" in tool_names
    assert "configure_event_filter" in tool_names

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
    mock_perception.analyze_crop_data.assert_called_once_with(1, None, "What is this?")
    mock_rpm.acquire.assert_called_once()
    mock_sem.__aenter__.assert_called_once()

@pytest.mark.anyio
@patch("pico.mcp.server.get_perception")
@patch("pico.mcp.server.get_ptz")
@patch("pico.mcp.server.get_memory")
@patch("pico.mcp.server.get_vlm_semaphore")
@patch("pico.mcp.server.get_vlm_rpm_limiter")
async def test_call_analyze_crop_image_class_filter(mock_get_rpm, mock_get_vlm_sem, mock_get_memory, mock_get_ptz, mock_get_perception):
    # Setup
    mock_perception = MagicMock()
    mock_get_perception.return_value = mock_perception
    mock_perception.analyze_crop_data.return_value = {
        "status": "success",
        "response": "A red suitcase."
    }

    mock_sem = AsyncMock()
    mock_get_vlm_sem.return_value = mock_sem
    mock_rpm = AsyncMock()
    mock_get_rpm.return_value = mock_rpm

    # Run
    res = await handle_call_tool("analyze_crop_image", {"class_filter": "suitcase", "query": "What color?"})

    # Assert
    assert len(res) == 1
    assert res[0].text == "A red suitcase."
    mock_perception.analyze_crop_data.assert_called_once_with(None, "suitcase", "What color?")

@pytest.mark.anyio
@patch("pico.mcp.server.get_perception")
@patch("pico.mcp.server.get_ptz")
@patch("pico.mcp.server.get_memory")
@patch("pico.mcp.server.get_shared_reader")
async def test_call_set_tracking_target_class_filter(mock_get_shared_reader, mock_get_memory, mock_get_ptz, mock_get_perception):
    # Setup
    mock_ptz = MagicMock()
    mock_get_ptz.return_value = mock_ptz
    mock_get_shared_reader.return_value = MagicMock()
    
    # Run
    res = await handle_call_tool("set_tracking_target", {"class_filter": "person"})

    # Assert
    assert len(res) == 1
    assert "person" in res[0].text
    mock_ptz.start_lockon.assert_called_with(track_id=None, class_filter="person")

@pytest.mark.anyio
@patch("pico.mcp.server.get_perception")
@patch("pico.mcp.server.get_ptz")
@patch("pico.mcp.server.get_memory")
async def test_call_get_live_snapshot(mock_get_memory, mock_get_ptz, mock_get_perception):
    # Setup
    mock_perception = MagicMock()
    mock_get_perception.return_value = mock_perception
    mock_perception.get_live_snapshot_data.return_value = {
        "status": "success",
        "filepath": "monitor/live_snapshot.jpg"
    }

    # Run
    res = await handle_call_tool("get_live_snapshot", {})

    # Assert
    assert len(res) == 1
    assert "Live Snapshot" in res[0].text
    mock_perception.get_live_snapshot_data.assert_called_once()

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

@pytest.mark.anyio
@patch("pico.mcp.server.get_perception")
@patch("pico.mcp.server.get_ptz")
@patch("pico.mcp.server.get_memory")
async def test_call_write_wiki(mock_get_memory, mock_get_ptz, mock_get_perception):
    # Setup
    mock_memory = MagicMock()
    mock_get_memory.return_value = mock_memory
    mock_memory.write_knowledge_data.return_value = {
        "status": "success",
        "filepath": "wiki/test_write.md"
    }

    # Run
    res = await handle_call_tool("write_wiki", {
        "filepath": "wiki/test_write.md",
        "title": "New Knowledge",
        "content": "Test knowledge content",
        "tags": "test knowledge"
    })

    # Assert
    assert len(res) == 1
    assert "Success" in res[0].text or "success" in res[0].text
    mock_memory.write_knowledge_data.assert_called_once_with(
        "wiki/test_write.md", "New Knowledge", "Test knowledge content", "test knowledge", None
    )

@pytest.mark.anyio
@patch("pico.mcp.server.get_perception")
@patch("pico.mcp.server.get_ptz")
@patch("pico.mcp.server.get_memory")
async def test_call_write_wiki_missing_args(mock_get_memory, mock_get_ptz, mock_get_perception):
    # Run with missing content
    res = await handle_call_tool("write_wiki", {
        "filepath": "wiki/test_write.md",
        "title": "New Knowledge"
    })

    # Assert
    assert len(res) == 1
    assert "Error" in res[0].text
