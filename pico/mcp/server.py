import asyncio
import logging
import sys
import os
import time
import json
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.types as types
from pico.config import AppConfig
from pico.cli.ptz import PTZActuator
from pico.cli.memory import SQLiteMemoryCLI
from pico.cli.perception import OnDemandPerceptionCLI
from pico.rate_limiter import RPMLimiter

# 標準出力を汚染しないようにログを標準エラーに出力する
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# 遅延初期化のためのグローバル変数
config = None
ptz = None
memory = None
perception = None
shared_reader = None

# 非同期同期制御オブジェクトの遅延初期化用
yolo_semaphore = None
vlm_semaphore = None
vlm_rpm_limiter = None

def get_config():
    global config
    if config is None:
        config = AppConfig()
    return config

def get_shared_reader():
    global shared_reader
    if shared_reader is None:
        from pico.config import build_rtsp_url
        from pico.video_reader import RTSPVideoReader
        conf = get_config()
        rtsp_url = build_rtsp_url(conf.tapo_user, conf.tapo_pass, conf.tapo_ip)
        shared_reader = RTSPVideoReader(rtsp_url)
        # ストリームバッファ蓄積待ち
        time.sleep(1.5)
    return shared_reader

def get_ptz():
    global ptz
    if ptz is None:
        ptz = PTZActuator(get_config())
    return ptz

def get_memory():
    global memory
    if memory is None:
        memory = SQLiteMemoryCLI()
    return memory

def get_perception():
    global perception
    if perception is None:
        perception = OnDemandPerceptionCLI(get_config(), shared_reader=get_shared_reader())
        ptz_inst = get_ptz()
        perception.set_ptz_actuator(ptz_inst)
        # デフォルト追尾ターゲットとして "person" (人) を自動セット
        ptz_inst.start_lockon(class_filter="person")
    return perception

def get_yolo_semaphore():
    global yolo_semaphore
    if yolo_semaphore is None:
        yolo_semaphore = asyncio.Semaphore(1)
    return yolo_semaphore

def get_vlm_semaphore():
    global vlm_semaphore
    if vlm_semaphore is None:
        vlm_semaphore = asyncio.Semaphore(1)
    return vlm_semaphore

def get_vlm_rpm_limiter():
    global vlm_rpm_limiter
    if vlm_rpm_limiter is None:
        vlm_rpm_limiter = RPMLimiter(max_rpm=get_config().ollama_max_rpm)
    return vlm_rpm_limiter


server = Server("cognitive-surveillance-mcp")

# lockon 追従タスクの管理
lockon_task = None

async def _run_lockon_loop(track_id: int | None = None, class_filter: str | None = None):
    try:
        loop = asyncio.get_running_loop()
        active_ptz = get_ptz()
        await loop.run_in_executor(None, active_ptz.lockon, get_shared_reader(), track_id, class_filter)
    except asyncio.CancelledError:
        logger.info("Lockon task cancelled. Stopping camera control...")
        get_ptz().stop_lockon()
    except Exception as e:
        logger.error(f"Error in lockon loop: {e}")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_active_tracks",
            description="常時更新されているYOLO追跡オブジェクトのテキスト情報のみを高速取得します。",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        types.Tool(
            name="analyze_crop_image",
            description="特定のオブジェクトを高解像度ズームクロップ、またはカメラの全体フレームを対象とし、オンデマンドでVLM画像解釈を実行します。track_id と class_filter を両方とも省略（または null）した場合は、ズームクロップを行わずにカメラの全体画像に対して画像解釈を実行します。",
            inputSchema={
                "type": "object",
                "properties": {
                    "track_id": {"type": "integer", "description": "ズーム精査する対象のYOLOトラックID。省略した場合は class_filter または全体フレーム解析になります。"},
                    "class_filter": {"type": "string", "description": "ズーム精査する対象のオブジェクトクラス名 (例: 'suitcase', 'person')。省略した場合は全体フレーム解析になります。"},
                    "query": {"type": "string", "description": "VLMに画像解釈させるための具体的なプロンプト・問いかけ"}
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="set_tracking_target",
            description="物理PIDサーボループが自動追従ロックオンすべきターゲットを指定します。IDまたはクラス名 (class_filter) を指定して開始し、両方指定しない場合は解除します。",
            inputSchema={
                "type": "object",
                "properties": {
                    "track_id": {"type": "integer", "description": "ロックオン追従する対象のトラックID（省略可能）"},
                    "class_filter": {"type": "string", "description": "自動追跡ロックオンを開始するオブジェクトクラス名 (例: 'person', 'suitcase')。見失っても自動で再捕捉します（省略可能）"}
                }
            }
        ),
        types.Tool(
            name="get_live_snapshot",
            description="カメラの現在のライブ映像をキャプチャし、その画像をチャット欄に表示します。※注意: AIエージェントは自身が画像情報を理解・解釈する目的でこのツールを使用してはなりません。AI自身による視覚精査や状況把握には、必ず get_active_tracks と analyze_crop_image を使用してください。本ツールは、人間のユーザーにライブ画像を報告・提示するためだけに使用するものです。",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        types.Tool(
            name="search_wiki",
            description="現在の状況に最も合致する過去の会話設定や、物理制限ルールを想起（検索）します。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQLite Trigram検索を走らせるための日本語キーワード"}
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="move_camera",
            description="カメラを任意のアングル方向へ指定した角度（Pan / Tilt 相対移動量）で手動ダイレクト移動・旋回させます。正の値は右/上、負の値は左/下に動きます。",
            inputSchema={
                "type": "object",
                "properties": {
                    "pan": {"type": "number", "description": "水平旋回量 (-0.96 ～ 0.96。正: 右、負: 左)"},
                    "tilt": {"type": "number", "description": "垂直旋回量 (-0.89 ～ 0.89。正: 上、負: 下)"}
                },
                "required": ["pan", "tilt"]
            }
        ),
        types.Tool(
            name="conduct_room_survey",
            description="カメラを全方位（左、中央、右、上）へ自律的に順次旋回させて室内をマルチアングル知覚し、部屋の状況や検出オブジェクトを解析して Obsidian Long-Term Wiki ページに自動記録します。",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        types.Tool(
            name="write_wiki",
            description="新しい観測事実、ユーザー指定ルール、会話インサイト、外部検索結果を OKF 形式 Markdown に書き込み、SQLite インデックスを更新します。",
            inputSchema={
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "書き込み対象の Markdown パス (例: 'wiki/known_objects_tama.md')"},
                    "title": {"type": "string", "description": "記憶・ナレッジのタイトル"},
                    "content": {"type": "string", "description": "記録する本文・観察事実"},
                    "tags": {"type": "string", "description": "スペース区切りのタグ (任意)"}
                },
                "required": ["filepath", "title", "content"]
            }
        ),
        types.Tool(
            name="get_perception_status",
            description="現在のシステム全体の知覚稼働状況、FPS、アクティブな検出オブジェクト一覧、および直近で能動発火したイベント履歴を一括照会・問い合わせします。",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        types.Tool(
            name="configure_event_filter",
            description="能動的イベント発火の過剰抑止・抑制ルールのカスタマイズ設定（クールダウン秒数、監視対象クラス制限）を行います。",
            inputSchema={
                "type": "object",
                "properties": {
                    "cooldown_sec": {"type": "number", "description": "同一イベント・IDに対する再発火抑制秒数 (例: 5.0)"},
                    "allowed_classes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "監視発火対象とするクラス名リスト (例: ['person', 'car'])。null の場合は全クラス対象。"
                    }
                }
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    global lockon_task
    
    if arguments is None:
        arguments = {}

    try:
        if name == "get_active_tracks":
            loop = asyncio.get_running_loop()
            active_perception = get_perception()
            active_perception.set_ptz_actuator(get_ptz())  # PTZ連携のセット
            
            async with get_yolo_semaphore():
                tracks = await loop.run_in_executor(None, active_perception.get_tracks_data)
            return [types.TextContent(type="text", text=str(tracks))]
            
        elif name == "analyze_crop_image":
            track_id_val = arguments.get("track_id")
            track_id = int(track_id_val) if track_id_val is not None else None
            class_filter = arguments.get("class_filter")
            query = str(arguments["query"])
            
            loop = asyncio.get_running_loop()
            active_perception = get_perception()
            active_perception.set_ptz_actuator(get_ptz())  # PTZ連携のセット
            
            # 1. VLM の RPM レートリミット制限待機
            await get_vlm_rpm_limiter().acquire()
            
            # 2. VLM の同時推論排他セマフォロック
            async with get_vlm_semaphore():
                res = await loop.run_in_executor(
                    None, 
                    active_perception.analyze_crop_data, 
                    track_id, 
                    class_filter, 
                    query
                )
            
            if "error" in res:
                return [types.TextContent(type="text", text=f"Error: {res['error']}")]
            elif res.get("status") == "success":
                return [types.TextContent(type="text", text=res["response"])]
            else:
                return [types.TextContent(type="text", text=f"Error: {res.get('message', 'Unknown VLM error')}")]

        elif name == "move_camera":
            pan = float(arguments.get("pan", 0.0))
            tilt = float(arguments.get("tilt", 0.0))
            active_ptz = get_ptz()
            active_perception = get_perception()
            active_perception.set_ptz_actuator(active_ptz)

            loop = asyncio.get_running_loop()
            actual_x, actual_y = await loop.run_in_executor(None, active_ptz.send_pulse_move, pan, tilt)
            return [types.TextContent(type="text", text=f"Success: カメラをダイレクト物理移動しました (Requested: Pan={pan:+.2f}, Tilt={tilt:+.2f})")]

        elif name == "conduct_room_survey":
            loop = asyncio.get_running_loop()
            active_ptz = get_ptz()
            active_perception = get_perception()
            active_perception.set_ptz_actuator(active_ptz)
            active_memory = get_memory()

            def _run_survey():
                angles = [
                    ("CENTER (正面・中央)", 0.0, 0.0),
                    ("LEFT (部屋の左側・書棚/デスク方面)", -0.40, 0.0),
                    ("RIGHT (部屋の右側・窓/カーテン方面)", +0.80, 0.0),
                    ("UPPER (天井・上部照明方面)", -0.40, +0.35),
                ]
                survey_results = []
                for n, p, t in angles:
                    if p != 0.0 or t != 0.0:
                        active_ptz.send_pulse_move(p, t)
                        time.sleep(2.0)
                    status = active_perception.get_perception_status_data()
                    tracks = status.get("active_tracks", [])
                    track_summary = [f"- ID {tr.get('track_id')}: クラス={tr.get('class')}, 信頼度={tr.get('confidence', 0):.2f}" for tr in tracks]
                    obs_text = "\n".join(track_summary) if track_summary else "（このアングルでは顕著な検出オブジェクトなし）"
                    survey_results.append(f"### 📍 方向アングル: {n}\n\n**検出オブジェクト一覧:**\n{obs_text}\n")
                
                active_ptz.send_pulse_move(-0.0, -0.35)  # 中心復帰
                
                title = "部屋の全方位環境調査記録_20260725"
                content = "# 部屋の全方位環境調査記録\n\n- **調査日時**: 2026年7月25日\n- **使用システム**: Pico Cognitive MCP Active Sensing Engine\n\n## 📋 パノラマ全方位調査結果\n\n" + "\n\n".join(survey_results)
                active_memory.add_document(filepath=f"memory/{title}.md", title=title, content=content, doc_type="survey", aliases=["部屋全方位調査"])
                return f"Success: 全方位室内パノラマ調査が完了し、Wikiページ '[[{title}]]' に全結果を保存しました！\n\n{content}"

            res_text = await loop.run_in_executor(None, _run_survey)
            return [types.TextContent(type="text", text=res_text)]

        elif name == "set_tracking_target":
            track_id_val = arguments.get("track_id")
            track_id = int(track_id_val) if track_id_val is not None else None
            class_filter = arguments.get("class_filter")
            
            active_ptz = get_ptz()
            _ = get_perception()  # 知覚エンジンとの連携確保

            if track_id is not None or class_filter is not None:
                active_ptz.start_lockon(track_id=track_id, class_filter=class_filter)
                target_desc = f"ID: {track_id}" if track_id is not None else f"Class: '{class_filter}'"
                return [types.TextContent(type="text", text=f"Success: 追跡ターゲットを {target_desc} に設定・変更し、自律自動追尾を開始しました。")]
            else:
                # 引数なしで解除指示があった場合はデフォルトの "person" (人) 追尾に復帰
                active_ptz.start_lockon(class_filter="person")
                return [types.TextContent(type="text", text="Success: 追跡ターゲットをデフォルトの 'person' (人) にリセットしました。")]

        elif name == "get_live_snapshot":
            loop = asyncio.get_running_loop()
            active_perception = get_perception()
            active_perception.set_ptz_actuator(get_ptz())
            
            res = await loop.run_in_executor(None, active_perception.get_live_snapshot_data)
            if "error" in res:
                return [types.TextContent(type="text", text=f"Error: {res['error']}")]
            else:
                t = int(time.time())
                abs_path = os.path.abspath(res["filepath"]).replace("\\", "/")
                md_img = f"![Live Snapshot](file:///{abs_path}?t={t})"
                alert_msg = (
                    "\n\n[SYSTEM ALERT: AI agents MUST NOT load, view, or analyze this image file directly. "
                    "For any automated target inspection or visual understanding, you MUST call 'get_active_tracks' first, "
                    "then use 'analyze_crop_image' with a query.]"
                )
                return [types.TextContent(type="text", text=f"Success: 現在のカメラフレームをキャプチャしました。\n\n{md_img}{alert_msg}")]

        elif name == "search_wiki":
            query = str(arguments["query"])
            loop = asyncio.get_running_loop()
            active_memory = get_memory()
            res = await loop.run_in_executor(None, active_memory.search_knowledge_data, query)
            return [types.TextContent(type="text", text=str(res))]

        elif name == "write_wiki":
            filepath = arguments.get("filepath")
            title = arguments.get("title")
            content = arguments.get("content")
            tags = arguments.get("tags", "")
            aliases_arg = arguments.get("aliases")
            
            aliases = None
            if isinstance(aliases_arg, list):
                aliases = aliases_arg
            elif isinstance(aliases_arg, str) and aliases_arg.strip():
                aliases = [a.strip() for a in aliases_arg.split(",") if a.strip()]

            if not filepath or not title or not content:
                return [types.TextContent(type="text", text="Error: 'filepath', 'title', and 'content' are required for write_wiki.")]

            loop = asyncio.get_running_loop()
            active_memory = get_memory()
            res = await loop.run_in_executor(
                None,
                active_memory.write_knowledge_data,
                filepath,
                title,
                content,
                tags,
                aliases
            )
            return [types.TextContent(type="text", text=f"Success: Knowledge saved to {filepath}. Result: {json.dumps(res, ensure_ascii=False)}")]

        elif name == "get_perception_status":
            loop = asyncio.get_running_loop()
            active_perception = get_perception()
            active_perception.set_ptz_actuator(get_ptz())
            res = await loop.run_in_executor(None, active_perception.get_perception_status_data)
            return [types.TextContent(type="text", text=json.dumps(res, indent=2, ensure_ascii=False))]

        elif name == "configure_event_filter":
            cooldown_sec = arguments.get("cooldown_sec")
            allowed_classes = arguments.get("allowed_classes")
            if cooldown_sec is not None:
                cooldown_sec = float(cooldown_sec)
            loop = asyncio.get_running_loop()
            active_perception = get_perception()
            active_perception.set_ptz_actuator(get_ptz())
            res = await loop.run_in_executor(
                None,
                active_perception.configure_event_filter_data,
                cooldown_sec,
                allowed_classes
            )
            return [types.TextContent(type="text", text=f"Success: Event filter configured. Updated perception status:\n{json.dumps(res, indent=2, ensure_ascii=False)}")]

        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        logger.error(f"Critical Error in MCP Bridge: {e}", exc_info=True)
        return [types.TextContent(type="text", text=f"Critical Error in MCP Bridge: {str(e)}")]

async def main():
    from mcp.server.stdio import stdio_server
    from mcp.server.lowlevel.server import NotificationOptions
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="cognitive-surveillance-mcp",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(tools_changed=False),
                    experimental_capabilities=None
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
