import asyncio
import os
import cv2
import sys
from pico.ollama_client import OllamaVisionClient
from pico.config import AppConfig

# 標準エラーではなく標準出力に出す
logging_format = "%(asctime)s [%(levelname)s] %(message)s"

async def main():
    try:
        config = AppConfig()
    except Exception as e:
        print(f"Config error: {e}")
        sys.exit(1)
        
    vlm = OllamaVisionClient(base_url=config.ollama_base_url, model=config.ollama_model)
    
    # 二つの画像をそれぞれVLMに投げて分析します
    for name in ["latest_crop.jpg", "live_snapshot.jpg"]:
        path = f"monitor/{name}"
        if not os.path.exists(path):
            print(f"⚠️ ファイルが存在しません: {path}")
            continue
            
        print(f"\n==========================================")
        print(f"🔍 画像解析中: {path} (サイズ: {os.path.getsize(path)} bytes)")
        print(f"==========================================")
        
        img = cv2.imread(path)
        if img is None:
            print(f"❌ 画像の読み込みに失敗しました: {path}")
            continue
            
        query = "この画像にサーフボードが映っていますか？それとも他のもの（壁、ドア、ドアの枠、家具など）の誤認識でしょうか？写っているものが何であるか、日本語で詳しく説明してください。"
        print(f"質問: '{query}'\n")
        
        try:
            response = await vlm.analyze_scene(img, query)
            print("💡 VLMの回答:")
            print(response)
        except Exception as e:
            print(f"❌ VLM解析エラー: {e}")
            
    await vlm.close()

if __name__ == "__main__":
    asyncio.run(main())
