import asyncio
import json
import httpx

async def run_experiment():
    print("🧪 [LLM 実験] WikiLinks 自動抽出 & 相互関係抽出の実験開始...")
    
    url = "http://localhost:11434/api/generate"
    model = "gemma4:e2b"  # もしくは起動中の Ollama モデル
    
    sample_text = (
        "夕方の庭（Zone B）に飼い猫のタマちゃんが現れました。"
        "タマちゃんは赤色の首輪をしており、西日が差し込むテラス付近を散歩しています。"
        "家主のたかしさんは置き配のダンボールを玄関アプローチ（Zone A）に置くよう希望しています。"
    )
    
    known_entities = ["Zone B (庭)", "Zone A (玄関)", "タマちゃん (飼い猫)", "たかしさん (家主)"]
    
    prompt = f"""以下はカメラ観察記録およびユーザー指示テキストです。
テキストから主要な「エンティティ（対象・場所・人物）」および「相互の関係性（関係トリプル）」を抽出し、
本文中のエンティティを Obsidian 形式の WikiLink ([[エンティティ名]]) に自動補完したテキストを作成してください。

【テキスト】
{sample_text}

【既知エンティティ】
{known_entities}

【出力フォーマット (JSON)】
{{
  "linked_content": "WikiLink補完後の本文",
  "extracted_links": ["リンク1", "リンク2"],
  "relations": [
    {{"source": "エンティティA", "relation": "関係性", "target": "エンティティB"}}
  ]
}}
JSON のみを出力してください。
"""

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(url, json={
                "model": model,
                "prompt": prompt,
                "format": "json",
                "stream": False
            })
            if resp.status_code == 200:
                result = resp.json().get("response", "")
                print(f"✅ Ollama 応答取得成功:\n{result}")
                try:
                    parsed = json.loads(result)
                    print("\n解析結果:")
                    print("・Linked Content:", parsed.get("linked_content"))
                    print("・Extracted Links:", parsed.get("extracted_links"))
                    print("・Relations:", parsed.get("relations"))
                except Exception as parse_err:
                    print("JSON パースエラー:", parse_err)
            else:
                print("Ollama API Error:", resp.status_code, resp.text)
        except Exception as e:
            print("Ollama 呼び出し例外:", e)

if __name__ == "__main__":
    asyncio.run(run_experiment())
