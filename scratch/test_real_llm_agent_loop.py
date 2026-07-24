import asyncio
import json
import os
import sys
import httpx

async def run_real_llm_agent_test():
    print("=" * 75)
    print("🧠 [実 LLM 駆動テスト] Ollama (gemma4:e2b) エージェント E2E ループ")
    print("=" * 75)

    ollama_url = "http://localhost:11434/api/chat"
    model = "gemma4:e2b"

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

    async def call_mcp(method: str, params: dict = None):
        nonlocal req_id
        req = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            req["params"] = params
        req_id += 1
        proc.stdin.write((json.dumps(req) + "\n").encode("utf-8"))
        await proc.stdin.drain()
        line = await proc.stdout.readline()
        if not line:
            raise RuntimeError("No output from MCP server")
        return json.loads(line.decode("utf-8"))

    try:
        # 1. MCP ハンドシェイク
        await call_mcp("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "llm-loop-tester", "version": "1.0.0"}
        })
        proc.stdin.write((json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n").encode("utf-8"))
        await proc.stdin.drain()

        # MCP ツール一覧の取得
        tools_res = await call_mcp("tools/list")
        available_tools = tools_res.get("result", {}).get("tools", [])
        
        # Ollama /api/chat 用の tools フォーマットに変換
        ollama_tools = []
        for t in available_tools:
            ollama_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["inputSchema"]
                }
            })

        system_instruction = (
            "あなたは自律型カメラエージェントの司令塔です。"
            "提供された MCP ツール (write_wiki, search_wiki, analyze_crop_image 等) を活用して、"
            "観察結果やユーザーの指示を長期記憶 (Wiki) に記録したり想起したりしてください。"
            "新しい場所や現象を追記する際は、既存のエンティティを [[エンティティ名]] のように WikiLink 化して相互接続してください。"
        )

        messages = [{"role": "system", "content": system_instruction}]

        async with httpx.AsyncClient(timeout=120.0) as http_client:

            # ------------------------------------------------------------------
            # TURN 1: ユーザーからの指示入力 ➔ LLM が自律的に write_wiki を呼び出すか
            # ------------------------------------------------------------------
            print("\n🗣️ 【TURN 1】 ユーザー指示:")
            user_msg_1 = (
                "夕方18時前後に庭（Zone B）にやってくる白茶トラの猫は、近所で可愛がられているタマちゃんだよ。"
                "タマちゃんが来たら静かに見守るルールにして、長期記憶 wiki/known_objects_tama.md に保存しておいて。"
            )
            print(f"   ユーザー: \"{user_msg_1}\"")
            messages.append({"role": "user", "content": user_msg_1})

            print("\n🤖 【TURN 1】 Ollama LLM の思考・ツール選択中...")
            resp1 = await http_client.post(ollama_url, json={
                "model": model,
                "messages": messages,
                "tools": ollama_tools,
                "stream": False
            })
            msg1 = resp1.json().get("message", {})
            messages.append(msg1)

            tool_calls1 = msg1.get("tool_calls", [])
            print(f"   LLM の自律判断: {len(tool_calls1)} 件のツールをキック")

            for tc in tool_calls1:
                fn = tc.get("function", {})
                tool_name = fn.get("name")
                tool_args = fn.get("arguments", {})
                print(f"   ⚡ LLM が呼び出した MCP ツール: {tool_name}")
                print(f"      引数: {json.dumps(tool_args, ensure_ascii=False)}")

                # MCP サーバーで実際にツールを実行！
                mcp_out = await call_mcp("tools/call", {"name": tool_name, "arguments": tool_args})
                res_text = mcp_out.get("result", {}).get("content", [{}])[0].get("text", "")
                print(f"      実行結果: {res_text[:100]}...")

                messages.append({
                    "role": "tool",
                    "content": res_text
                })

            # LLM にツール実行結果を踏まえた最終回答を出させる
            resp1_final = await http_client.post(ollama_url, json={
                "model": model,
                "messages": messages,
                "tools": ollama_tools,
                "stream": False
            })
            final_msg1 = resp1_final.json().get("message", {}).get("content", "")
            print(f"\n💬 【TURN 1 LLM 応答】:\n{final_msg1}\n")
            messages.append({"role": "assistant", "content": final_msg1})

            # ------------------------------------------------------------------
            # TURN 2: カメラ観察イベント ➔ LLM が search_wiki で想起 ➔ write_wiki で相互リンク保存
            # ------------------------------------------------------------------
            print("\n📹 【TURN 2】 観察イベント発生:")
            event_msg = (
                "【カメラ検知イベント】 17:50 庭 (Zone B) にて白茶トラの猫を検出しました。"
                "過去の規則を想起して適切なアクションを講じ、観察記録を wiki/observation_log_20260724.md に記録してください。"
                "記録時は [[飼い猫タマちゃん]] などの相互リンクを活用してください。"
            )
            print(f"   イベント: \"{event_msg}\"")
            messages.append({"role": "user", "content": event_msg})

            print("\n🤖 【TURN 2】 Ollama LLM の思考・想起・記録ループ...")
            resp2 = await http_client.post(ollama_url, json={
                "model": model,
                "messages": messages,
                "tools": ollama_tools,
                "stream": False
            })
            msg2 = resp2.json().get("message", {})
            messages.append(msg2)

            tool_calls2 = msg2.get("tool_calls", [])
            print(f"   LLM の自律判断: {len(tool_calls2)} 件のツールをキック")

            for tc in tool_calls2:
                fn = tc.get("function", {})
                tool_name = fn.get("name")
                tool_args = fn.get("arguments", {})
                print(f"   ⚡ LLM が呼び出した MCP ツール: {tool_name}")
                print(f"      引数: {json.dumps(tool_args, ensure_ascii=False)}")

                mcp_out = await call_mcp("tools/call", {"name": tool_name, "arguments": tool_args})
                res_text = mcp_out.get("result", {}).get("content", [{}])[0].get("text", "")
                print(f"      実行結果: {res_text[:150]}...")

                messages.append({
                    "role": "tool",
                    "content": res_text
                })

            resp2_final = await http_client.post(ollama_url, json={
                "model": model,
                "messages": messages,
                "tools": ollama_tools,
                "stream": False
            })
            final_msg2 = resp2_final.json().get("message", {}).get("content", "")
            print(f"\n💬 【TURN 2 LLM 最終応答】:\n{final_msg2}\n")

            # ------------------------------------------------------------------
            # TURN 3: 実装された自動相互リンク (WikiLinks & Backlinks) の検証
            # ------------------------------------------------------------------
            print("\n🔍 【TURN 3】 ディスクおよび DB 内のナレッジグラフ（相互リンク）構造の検証...")
            search_verify = await call_mcp("tools/call", {
                "name": "search_wiki",
                "arguments": {"query": "タマ"}
            })
            verify_text = search_verify.get("result", {}).get("content", [{}])[0].get("text", "")
            print(f"   [search_wiki('タマ') 想起結果]:\n{verify_text}\n")

            assert "飼い猫タマちゃん" in verify_text or "tama" in verify_text.lower(), "タマちゃんの記憶が想起されませんでした"
            print("=" * 75)
            print("🎉 実 LLM (Ollama gemma4:e2b) によるマルチターン E2E 実テスト全行程成功！")
            print("=" * 75)

    finally:
        proc.terminate()
        await proc.wait()
        for f in ["wiki/known_objects_tama.md", "wiki/observation_log_20260724.md"]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

if __name__ == "__main__":
    asyncio.run(run_real_llm_agent_test())
