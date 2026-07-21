import asyncio
import logging
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.types as types
from pico.config import AppConfig
from pico.cli.ptz import PTZActuator
from pico.cli.memory import SQLiteMemoryCLI
from pico.cli.perception import OnDemandPerceptionCLI

logger = logging.getLogger(__name__)

# シングルトンインスタンスの設定
config = AppConfig()
ptz = PTZActuator(config)
memory = SQLiteMemoryCLI()
perception = OnDemandPerceptionCLI(config)

server = Server("cognitive-surveillance-mcp")

# lockon 追従タスクの管理
lockon_task = None

async def _run_lockon_loop(track_id: int):
    try:
        # loop.run_in_executor を使って同期的な lockon ループを別スレッドで走らせる
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, ptz.lockon, track_id)
    except asyncio.CancelledError:
        logger.info("Lockon task cancelled. Stopping camera control...")
        ptz.stop_lockon()
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
            tracks = await loop.run_in_executor(None, perception.get_tracks_data)
            return [types.TextContent(type="text", text=str(tracks))]
            
        elif name == "analyze_crop_image":
            track_id = int(arguments["track_id"])
            query = str(arguments["query"])
            
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(None, perception.analyze_crop_data, track_id, query)
            
            if "error" in res:
                return [types.TextContent(type="text", text=f"Error: {res['error']}")]
            elif res.get("status") == "success":
                return [types.TextContent(type="text", text=res["response"])]
            else:
                return [types.TextContent(type="text", text=f"Error: {res.get('message', 'Unknown VLM error')}")]

        elif name == "set_tracking_target":
            track_id_val = arguments.get("track_id")
            
            # すでに実行中の追従タスクがあればキャンセル
            if lockon_task and not lockon_task.done():
                lockon_task.cancel()
                try:
                    await lockon_task
                except asyncio.CancelledError:
                    pass
                lockon_task = None
                
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, ptz.stop_lockon)
            await loop.run_in_executor(None, ptz.emergency_stop)

            if track_id_val is not None:
                track_id = int(track_id_val)
                lockon_task = asyncio.create_task(_run_lockon_loop(track_id))
                return [types.TextContent(type="text", text=f"Success: 追跡ターゲットを ID: {track_id} に固定しました。")]
            else:
                return [types.TextContent(type="text", text="Success: 追跡ターゲットを解除しました。")]

        elif name == "search_wiki":
            query = str(arguments["query"])
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(None, memory.search_knowledge_data, query)
            return [types.TextContent(type="text", text=str(res))]

        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        logger.error(f"Critical Error in MCP Bridge: {e}", exc_info=True)
        return [types.TextContent(type="text", text=f"Critical Error in MCP Bridge: {str(e)}")]

async def main():
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="cognitive-surveillance-mcp",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
