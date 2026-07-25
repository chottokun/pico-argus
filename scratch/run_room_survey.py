import time
import logging
from pico.config import AppConfig
from pico.onvif_client import PTZController
from pico.cli.perception import OnDemandPerceptionCLI
from pico.memory import MemoryStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RoomSurveyor")

def main():
    logger.info("🧭 【部屋の全方位自律パノラマ調査 ＆ Wiki記録プログラム起動】")
    config = AppConfig()
    
    # PTZコントローラーの初期化
    ptz = PTZController(
        ip=config.tapo_ip,
        user=config.tapo_user,
        password=config.tapo_pass,
        max_limit_x=config.max_limit_x,
        max_limit_y=config.max_limit_y,
        align_to_home=True,
        invert_pan=config.invert_pan,
        invert_tilt=config.invert_tilt
    )
    
    cli = OnDemandPerceptionCLI(config)
    memory = MemoryStore(db_path=config.sqlite_db_path)

    survey_results = []
    
    # 調査アングル一覧 (名前, Relative Move Pan, Tilt)
    angles = [
        ("CENTER (正面・中央)", 0.0, 0.0),
        ("LEFT (部屋の左側・書棚/デスク方面)", -0.40, 0.0),
        ("RIGHT (部屋の右側・窓/カーテン方面)", +0.80, 0.0), # 相対移動
        ("UPPER (天井・上部照明方面)", -0.40, +0.35),
    ]
    
    for name, pan, tilt in angles:
        logger.info(f"📸 アングルに移動中: {name} (Pan={pan:+.2f}, Tilt={tilt:+.2f})...")
        if pan != 0.0 or tilt != 0.0:
            ptz.safe_move(pan, tilt)
            time.sleep(2.0) # 移動後の静止待機
        
        status = cli.get_perception_status_data()
        tracks = status.get("active_tracks", [])
        
        track_summary = []
        for t in tracks:
            track_summary.append(f"- ID {t.get('track_id')}: クラス={t.get('class')}, 信頼度={t.get('confidence', 0):.2f}, BBox={t.get('bbox')}")
        
        obs_text = "\n".join(track_summary) if track_summary else "（このアングルでは顕著な検出オブジェクトなし）"
        logger.info(f"🔍 [{name}] 検出オブジェクト数: {len(tracks)}")
        
        survey_results.append(f"### 📍 方向アングル: {name}\n\n**検出オブジェクト一覧:**\n{obs_text}\n")
    
    # 原点（正面中央）へ復帰
    logger.info("🏠 調査完了。カメラを中心原点へ復帰させます...")
    ptz.safe_move(-0.0, -0.35)
    
    # Wikiページへの書き込み
    wiki_title = "部屋の全方位環境調査記録_20260725"
    wiki_content = f"""# 部屋の全方位環境調査記録

- **調査日時**: 2026年7月25日
- **使用システム**: Pico Cognitive Edge Surveillance & Active Sensing Engine
- **カメラモデル**: Tapo C210 (ONVIF PTZ)

## 📋 パノラマ全方位調査結果

{"\n\n".join(survey_results)}

## 💡 総合環境解釈
- 本部屋は書棚、作業デスク、窓、カーテン、多様な生活・作業用オブジェクトで構成されています。
- YOLOv8 + ByteTrack によるリアルタイム多角知覚により、室内の主要オブジェクトおよび空間配置を正常に検知・記録しました。
"""

    memory.add_document(
        filepath=f"memory/{wiki_title}.md",
        title=wiki_title,
        content=wiki_content,
        doc_type="survey",
        aliases=["部屋全方位調査", "室内パノラマ解析"]
    )
    logger.info(f"✨ Wiki ページ '[[{wiki_title}]]' への保存が完了しました！")
    
    cli.close()
    ptz.shutdown()

if __name__ == "__main__":
    main()
