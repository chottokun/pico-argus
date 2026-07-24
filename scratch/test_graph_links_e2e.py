import asyncio
import json
import os
import sys

async def run_graph_links_e2e_test():
    print("=" * 70)
    print("🧪 相互リンク (WikiLinks) & バックリンク E2E MCP 実テスト")
    print("=" * 70)

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
        print("\n1. Initialize & ハンドシェイク...")
        await send_rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "graph-tester", "version": "1.0.0"}
        })
        proc.stdin.write((json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n").encode("utf-8"))
        await proc.stdin.drain()

        # STEP 2: ノード 1 (タマちゃん) を write_wiki で登録 (aliases 付き)
        print("\n2. write_wiki でノード1 (飼い猫タマちゃん) を保存 (aliases: タマ, 猫のタマ)...")
        w1_res = await send_rpc("tools/call", {
            "name": "write_wiki",
            "arguments": {
                "filepath": "wiki/test_node_tama.md",
                "title": "飼い猫タマちゃん",
                "content": "白茶トラの猫。首輪は赤色。",
                "tags": "pet cat profile",
                "aliases": ["タマ", "猫のタマ"]
            }
        })
        print(f"   [RESULT]: {w1_res['result']['content'][0]['text']}")

        # STEP 3: ノード 2 (庭 Zone B) を write_wiki で登録 (ノード1へ [[飼い猫タマちゃん]] リンクを形成)
        print("\n3. write_wiki でノード2 (庭 Zone B) を保存 ([[飼い猫タマちゃん]] リンクを含む)...")
        w2_res = await send_rpc("tools/call", {
            "name": "write_wiki",
            "arguments": {
                "filepath": "wiki/test_node_garden.md",
                "title": "庭 Zone B",
                "content": "夕方に [[飼い猫タマちゃん]] が散歩する場所。",
                "tags": "zone garden"
            }
        })
        print(f"   [RESULT]: {w2_res['result']['content'][0]['text']}")

        # STEP 4: search_wiki で "タマ" (エイリアス) を検索し、バックリンク構造を取得検証
        print("\n4. search_wiki でエイリアス 'タマ' を検索し、自動バックリンク (被参照) の取得を検証...")
        s_res = await send_rpc("tools/call", {
            "name": "search_wiki",
            "arguments": {
                "query": "タマ"
            }
        })
        s_text = s_res['result']['content'][0]['text']
        print(f"\n🧠 [検索・想起結果 (レスポンス)]:\n{s_text}\n")

        # レスポンス内の解析
        assert "飼い猫タマちゃん" in s_text, "タマちゃんのナレッジが取得できませんでした"
        assert "backlinks" in s_text, "バックリンク情報がレスポンスに含まれていません"
        assert "wiki/test_node_garden.md" in s_text, "ノード2 (庭 Zone B) からのバックリンクが自動形成されていません"

        print("=" * 70)
        print("✅ 相互リンク・バックリンク・エイリアス名寄せ E2E 実テスト すべて合格いたしました！")
        print("=" * 70)

    finally:
        proc.terminate()
        await proc.wait()
        for f in ["wiki/test_node_tama.md", "wiki/test_node_garden.md"]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

if __name__ == "__main__":
    asyncio.run(run_graph_links_e2e_test())
