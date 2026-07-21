import asyncio
import logging
import sys
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

# 非同期同期制御オブジェクトの遅延初期化用
yolo_semaphore = None
vlm_semaphore = None
vlm_rpm_limiter = None

def get_config():
    global config
    if config is None:
        config = AppConfig()
    return config

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
        perception = OnDemandPerceptionCLI(get_config())
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

async def _run_lockon_loop(track_id: int):
    try:
        loop = asyncio.get_running_loop()
        active_ptz = get_ptz()
        await loop.run_in_executor(None, active_ptz.lockon, track_id)
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
            description="特定のオブジェクトを高解像度ズームクロップし、オンデマンドでVLM画像解釈を実行します。",
            inputSchema={
                "type": "object",
                "properties": {
                    "track_id": {"type": "integer", "description": "ズームして精査する対象のYOLOトラックID"},
                    "query": {"type": "string", "description": "VLMに画像解釈させるための具体的なプロンプト・問いかけ"}
                },
                "required": ["track_id", "query"]
            }
        ),
        types.Tool(
            name="set_tracking_target",
            description="物理PIDサーボループが自動追従ロックオンすべきYOLOトラックIDを動的指定します。null/Noneを指定した場合は解除します。",
            inputSchema={
                "type": "object",
                "properties": {
                    "track_id": {"type": ["integer", "null"], "description": "ロックオン追従する対象のトラックID。None/nullを指定した場合は解除"}
                },
                "required": ["track_id"]
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
            # YOLO推論の同時実行を防ぐためセマフォを取得
            async with get_yolo_semaphore():
                tracks = await loop.run_in_executor(None, active_perception.get_tracks_data)
            return [types.TextContent(type="text", text=str(tracks))]
            
        elif name == "analyze_crop_image":
            track_id = int(arguments["track_id"])
            query = str(arguments["query"])
            
            loop = asyncio.get_running_loop()
            active_perception = get_perception()
            
            # 1. VLM の RPM レートリミット制限待機 (過熱・熱スロットリング防止)
            await get_vlm_rpm_limiter().acquire()
            
            # 2. VLM の同時推論排他セマフォロック (VRAMのOOM防止)
            async with get_vlm_semaphore():
                res = await loop.run_in_executor(None, active_perception.analyze_crop_data, track_id, query)
            
            if "error" in res:
                return [types.TextContent(type="text", text=f"Error: {res['error']}")]
            elif res.get("status") == "success":
                return [types.TextContent(type="text", text=res["response"])]
            else:
                return [types.TextContent(type="text", text=f"Error: {res.get('message', 'Unknown VLM error')}")]

        elif name == "set_tracking_target":
            track_id_val = arguments.get("track_id")
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

            if track_id_val is not None:
                track_id = int(track_id_val)
                lockon_task = asyncio.create_task(_run_lockon_loop(track_id))
                return [types.TextContent(type="text", text=f"Success: 追跡ターゲットを ID: {track_id} に固定しました。")]
            else:
                return [types.TextContent(type="text", text="Success: 追跡ターゲットを解除しました。")]

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
