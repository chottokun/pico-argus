import asyncio
import json
import os
import sys

async def run_mcp_integration_test():
    """MCP サーバーを stdio サブプロセスとして起動し、JSON-RPC プロトコルで E2E テストを実施するスクリプト"""
    print("🚀 MCP サーバー統合テストを開始します...")

    # サブプロセスの起動
    cmd = [sys.executable, "-m", "pico.mcp.server"]
    env = dict(os.environ)
    env["PYTHONPATH"] = "."

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env
    )

    request_id = 1

    async def send_request(method: str, params: dict = None):
        nonlocal request_id
        req = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method
        }
        if params is not None:
            req["params"] = params
        request_id += 1

        line = json.dumps(req) + "\n"
        proc.stdin.write(line.encode("utf-8"))
        await proc.stdin.drain()

        # レスポンス受信
        resp_line = await proc.stdout.readline()
        if not resp_line:
            raise RuntimeError("No response from MCP server process.")
        return json.loads(resp_line.decode("utf-8"))

    try:
        # 1. initialize
        print("\n1. Initialize リクエスト送信...")
        init_res = await send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"}
        })
        print(f"   [RESULT] Server Name: {init_res.get('result', {}).get('serverInfo', {}).get('name')}")
        assert "serverInfo" in init_res.get("result", {}), "initialize failed"

        # initialized 通知
        notification = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        proc.stdin.write(notification.encode("utf-8"))
        await proc.stdin.drain()

        # 2. tools/list
        print("\n2. tools/list リクエスト送信...")
        tools_res = await send_request("tools/list")
        tools = tools_res.get("result", {}).get("tools", [])
        tool_names = [t["name"] for t in tools]
        print(f"   [RESULT] 利用可能ツール ({len(tools)}件): {tool_names}")
        assert "write_wiki" in tool_names, "write_wiki が tools/list に見つかりません"
        assert "search_wiki" in tool_names, "search_wiki が tools/list に見つかりません"

        # 3. tools/call (write_wiki)
        print("\n3. tools/call (write_wiki) の実行...")
        test_file = "wiki/test_mcp_e2e_output.md"
        write_res = await send_request("tools/call", {
            "name": "write_wiki",
            "arguments": {
                "filepath": test_file,
                "title": "E2E テスト用ナレッジ",
                "content": "MCP インテグレーションテストによる書き込み検証。",
                "tags": "test e2e mcp"
            }
        })
        content_text = write_res.get("result", {}).get("content", [{}])[0].get("text", "")
        print(f"   [RESULT] Response: {content_text}")
        assert "Success" in content_text or "success" in content_text, "write_wiki execution failed"
        assert os.path.exists(test_file), "書き込みファイルがディスク上に生成されていません"

        # 4. tools/call (search_wiki)
        print("\n4. tools/call (search_wiki) で書き込んだ記憶の想起検証...")
        search_res = await send_request("tools/call", {
            "name": "search_wiki",
            "arguments": {
                "query": "E2E テスト用ナレッジ"
            }
        })
        search_text = search_res.get("result", {}).get("content", [{}])[0].get("text", "")
        print(f"   [RESULT] Search Result: {search_text}")
        assert "E2E テスト用ナレッジ" in search_text, "書き込んだ記憶が search_wiki で検索できませんでした"

        print("\n✅ MCP サーバー統合テスト (E2E) すべて成功いたしました！")

    finally:
        # クリーンアップ
        proc.terminate()
        await proc.wait()
        if os.path.exists("wiki/test_mcp_e2e_output.md"):
            try:
                os.remove("wiki/test_mcp_e2e_output.md")
            except Exception:
                pass

if __name__ == "__main__":
    asyncio.run(run_mcp_integration_test())
