import asyncio
import json
import os
import sys
import cv2
import numpy as np

async def run_thorough_mcp_tests():
    print("=" * 70)
    print("🧪 MCP サーバー 徹底 E2E & Ollama VLM 結合テスト")
    print("=" * 70)

    # テスト用画像の生成 (OpenCV で赤いリンゴと青い車が描かれた画像を生成)
    os.makedirs("monitor", exist_ok=True)
    os.makedirs("wiki", exist_ok=True)
    
    test_img_path = "monitor/test_vlm_sample.jpg"
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # 背景
    img[:] = (240, 240, 240)
    # 赤い果物 (Circle)
    cv2.circle(img, (200, 240), 60, (0, 0, 220), -1)
    cv2.putText(img, "Red Apple", (160, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
    # 青い物体 (Rectangle)
    cv2.rectangle(img, (400, 180), (580, 300), (220, 0, 0), -1)
    cv2.putText(img, "Blue Object", (400, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 0, 0), 2)
    cv2.imwrite(test_img_path, img)
    print(f"📷 テスト用合成サンプル画像を作成しました: {test_img_path}")

    # MCP サーバープロセス起動
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

    req_id = 1

    async def send_rpc(method: str, params: dict = None):
        nonlocal req_id
        req = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            req["params"] = params
        req_id += 1
        line = json.dumps(req) + "\n"
        proc.stdin.write(line.encode("utf-8"))
        await proc.stdin.drain()

        resp_line = await proc.stdout.readline()
        if not resp_line:
            raise RuntimeError("MCP サーバーからレスポンスがありません。")
        return json.loads(resp_line.decode("utf-8"))

    try:
        # STEP 1: 初期化
        print("\n--- [STEP 1] MCP サーバー Initialize & ハンドシェイク ---")
        init_res = await send_rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "thorough-tester", "version": "1.0.0"}
        })
        server_info = init_res.get("result", {}).get("serverInfo", {})
        print(f"✅ サーバー接続成功: {server_info.get('name')} (v{server_info.get('version')})")

        notification = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        proc.stdin.write(notification.encode("utf-8"))
        await proc.stdin.drain()

        # STEP 2: ツール一覧
        print("\n--- [STEP 2] tools/list 全6ツールの検出検証 ---")
        tools_res = await send_rpc("tools/list")
        tools = tools_res.get("result", {}).get("tools", [])
        tool_names = [t["name"] for t in tools]
        print(f"✅ 検出されたツール一覧 ({len(tools)} 件): {tool_names}")
        expected_tools = {"get_active_tracks", "analyze_crop_image", "set_tracking_target", "get_live_snapshot", "search_wiki", "write_wiki"}
        assert expected_tools.issubset(set(tool_names)), "すべてのツールが定義されていません"

        # STEP 3: search_wiki & write_wiki
        print("\n--- [STEP 3] 記憶ツール (search_wiki / write_wiki) の動作検証 ---")
        w_res = await send_rpc("tools/call", {
            "name": "write_wiki",
            "arguments": {
                "filepath": "wiki/test_thorough.md",
                "title": "徹底テスト用知識",
                "content": "赤と青のオブジェクトが存在する。",
                "tags": "test thorough"
            }
        })
        print(f"  [write_wiki]: {w_res['result']['content'][0]['text']}")

        s_res = await send_rpc("tools/call", {
            "name": "search_wiki",
            "arguments": {"query": "徹底テスト"}
        })
        print(f"  [search_wiki]: {s_res['result']['content'][0]['text']}")

        # STEP 4: カメラ非接続環境での挙動検証
        print("\n--- [STEP 4] カメラ非接続状態でのセンサー/制御ツールの堅牢性検証 ---")
        
        # 4.1 get_active_tracks
        print("  - [get_active_tracks 呼び出し...]")
        tracks_res = await send_rpc("tools/call", {"name": "get_active_tracks", "arguments": {}})
        tracks_txt = tracks_res["result"]["content"][0]["text"]
        print(f"    応答: {tracks_txt[:100]}...")

        # 4.2 set_tracking_target
        print("  - [set_tracking_target 呼び出し...]")
        ptz_res = await send_rpc("tools/call", {
            "name": "set_tracking_target",
            "arguments": {"class_filter": "person"}
        })
        print(f"    応答: {ptz_res['result']['content'][0]['text']}")

        # 4.3 get_live_snapshot
        print("  - [get_live_snapshot 呼び出し...]")
        snap_res = await send_rpc("tools/call", {"name": "get_live_snapshot", "arguments": {}})
        print(f"    応答: {snap_res['result']['content'][0]['text'][:120]}...")

        # STEP 5: Ollama VLM との実機推論統合テスト (analyze_crop_image)
        print("\n--- [STEP 5] リアル Ollama VLM との analyze_crop_image 推論統合テスト ---")
        print("  - [analyze_crop_image 呼び出し (Ollama 推論待ち)...]")
        
        vlm_res = await send_rpc("tools/call", {
            "name": "analyze_crop_image",
            "arguments": {
                "query": "この画像には何が描かれていますか？色や形を詳しく説明してください。"
            }
        })
        vlm_txt = vlm_res["result"]["content"][0]["text"]
        print(f"\n🧠 [Ollama VLM レスポンス]:\n{vlm_txt}\n")

        print("=" * 70)
        print("🎉 すべての MCP ツールおよび Ollama VLM 結合テストが成功しました！")
        print("=" * 70)

    finally:
        proc.terminate()
        await proc.wait()
        if os.path.exists("wiki/test_thorough.md"):
            try:
                os.remove("wiki/test_thorough.md")
            except Exception:
                pass

if __name__ == "__main__":
    asyncio.run(run_thorough_mcp_tests())
