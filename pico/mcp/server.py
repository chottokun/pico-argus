import asyncio
import logging
import sys
import os
import time
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
            description="特定のオブジェクトを高解像度ズームクロップし、オンデマンドでVLM画像解釈を実行します。ID指定のほか、class_filter にオブジェクト名 (例: 'suitcase', 'person') を指定可能です。",
            inputSchema={
                "type": "object",
                "properties": {
                    "track_id": {"type": "integer", "description": "ズーム精査する対象のYOLOトラックID。省略した場合は class_filter を使用"},
                    "class_filter": {"type": "string", "description": "ズーム精査する対象のオブジェクトクラス名 (例: 'suitcase', 'person')"},
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
            description="カメラの現在のライブ映像をキャプチャし、その画像をチャット欄に表示します。",
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

        elif name == "set_tracking_target":
            track_id_val = arguments.get("track_id")
            track_id = int(track_id_val) if track_id_val is not None else None
            class_filter = arguments.get("class_filter")
            
            active_ptz = get_ptz()
            
            # すでに実行中の追従タスクがあればキャンセル
            if lockon_task and not lockon_task.done():
                lockon_task.cancel()
                try:
                    await lockon_task
                except asyncio.CancelledError:
                    pass
                lockon_task = None
                
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, active_ptz.stop_lockon)
            await loop.run_in_executor(None, active_ptz.emergency_stop)

            if track_id is not None or class_filter is not None:
                lockon_task = asyncio.create_task(_run_lockon_loop(track_id, class_filter))
                target_desc = f"ID: {track_id}" if track_id is not None else f"Class: '{class_filter}'"
                return [types.TextContent(type="text", text=f"Success: 追跡ターゲットを {target_desc} に設定し、自律自動追尾を開始しました。")]
            else:
                return [types.TextContent(type="text", text="Success: 追跡ターゲットを解除しました。")]

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
                return [types.TextContent(type="text", text=f"Success: 現在のカメラフレームをキャプチャしました。\n\n{md_img}")]

        elif name == "search_wiki":
            query = str(arguments["query"])
            loop = asyncio.get_running_loop()
            active_memory = get_memory()
            res = await loop.run_in_executor(None, active_memory.search_knowledge_data, query)
            return [types.TextContent(type="text", text=str(res))]

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
