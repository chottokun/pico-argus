import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pico.mcp.server import handle_call_tool

@pytest.mark.anyio
@patch("pico.mcp.server.get_perception")
@patch("pico.mcp.server.get_ptz")
@patch("pico.mcp.server.get_memory")
async def test_mcp_invalid_tool_name(mock_get_memory, mock_get_ptz, mock_get_perception):
    """存在しない未知のツール呼び出し時の防御テスト"""
    res = await handle_call_tool("non_existent_tool", {})
    assert len(res) == 1
    assert "Critical Error" in res[0].text or "Unknown tool" in res[0].text

@pytest.mark.anyio
@patch("pico.mcp.server.get_perception")
@patch("pico.mcp.server.get_ptz")
@patch("pico.mcp.server.get_memory")
async def test_mcp_write_wiki_invalid_types(mock_get_memory, mock_get_ptz, mock_get_perception):
    """write_wiki に型不一致パラメータが渡された場合の堅牢性テスト"""
    mock_memory = MagicMock()
    mock_get_memory.return_value = mock_memory
    mock_memory.write_knowledge_data.return_value = {"status": "success", "filepath": "wiki/test.md"}

    # 引数がリストや数数値などの場合
    res = await handle_call_tool("write_wiki", {
        "filepath": "wiki/test.md",
        "title": "Title",
        "content": "Content",
        "aliases": "alias1, alias2"  # リストではなくカンマ区切り文字列が来た場合
    })
    assert len(res) == 1
    assert "Success" in res[0].text

    mock_memory.write_knowledge_data.assert_called_once_with(
        "wiki/test.md", "Title", "Content", "", ["alias1", "alias2"]
    )

@pytest.mark.anyio
@patch("pico.mcp.server.get_perception")
@patch("pico.mcp.server.get_ptz")
@patch("pico.mcp.server.get_memory")
@patch("pico.mcp.server.get_vlm_semaphore")
@patch("pico.mcp.server.get_vlm_rpm_limiter")
async def test_mcp_vlm_exception_cleanup(mock_get_rpm, mock_get_vlm_sem, mock_get_memory, mock_get_ptz, mock_get_perception):
    """analyze_crop_image で例外が発生した際、セマフォ/リミッターが正しく通過・解放されるかの検証"""
    mock_perception = MagicMock()
    mock_get_perception.return_value = mock_perception
    mock_perception.analyze_crop_data.side_effect = RuntimeError("VLM unexpected crash")

    mock_sem = AsyncMock()
    mock_get_vlm_sem.return_value = mock_sem
    mock_rpm = AsyncMock()
    mock_get_rpm.return_value = mock_rpm

    res = await handle_call_tool("analyze_crop_image", {"query": "Test query"})
    assert len(res) == 1
    assert "Critical Error" in res[0].text or "Error" in res[0].text

    # セマフォの context manager が正しく入出されたか
    mock_sem.__aenter__.assert_called_once()
    mock_sem.__aexit__.assert_called_once()


@pytest.mark.anyio
@patch("pico.mcp.server.get_perception")
@patch("pico.mcp.server.get_ptz")
@patch("pico.mcp.server.get_memory")
async def test_mcp_structured_error_responses(mock_get_memory, mock_get_ptz, mock_get_perception) -> None:
    """ConnectionError や TimeoutError などの各種エラー時に構造化JSONエラーレスポンスが返ることを検証するテスト"""
    import json
    mock_perception = MagicMock()
    mock_get_perception.return_value = mock_perception

    # 1. ConnectionError の検証
    mock_perception.get_tracks_data.side_effect = ConnectionError("Could not connect to Tap camera")
    res = await handle_call_tool("get_active_tracks", {})
    assert len(res) == 1
    err_data = json.loads(res[0].text)
    assert err_data["status"] == "error"
    assert err_data["error_type"] == "CONNECTION_ERROR"
    assert "Failed to connect" in err_data["message"]
    assert "Could not connect" in err_data["details"]

    # 2. TimeoutError の検証
    mock_perception.get_tracks_data.side_effect = TimeoutError("Camera response timed out")
    res = await handle_call_tool("get_active_tracks", {})
    assert len(res) == 1
    err_data = json.loads(res[0].text)
    assert err_data["status"] == "error"
    assert err_data["error_type"] == "TIMEOUT_ERROR"
    assert "Operation timed out" in err_data["message"]
    assert "Camera response timed out" in err_data["details"]

    # 3. 一般的な例外の検証
    mock_perception.get_tracks_data.side_effect = RuntimeError("Something went wrong internally")
    res = await handle_call_tool("get_active_tracks", {})
    assert len(res) == 1
    err_data = json.loads(res[0].text)
    assert err_data["status"] == "error"
    assert err_data["error_type"] == "INTERNAL_ERROR"
    assert "Critical Error" in err_data["message"]
    assert "Something went wrong" in err_data["details"]
