import logging
import json
import re
import asyncio
import numpy as np
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from pico.ollama_client import OllamaVisionClient
from pico.agent_tools import AgentTools
from pico.perception_buffer import PerceptionBuffer

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    # 知覚レイヤー (Perception Buffer)
    active_tracks: List[Dict[str, Any]]       # YOLO/ByteTrack からのメタデータ
    active_tracks_text: str                   # 人間可読なテキスト表現
    
    # 物理制御・状態レイヤー
    lockon_mode: str                          # "auto" | "id"
    target_track_id: Optional[int]            # ロックオン対象ID
    
    # 認知・長期記憶
    agent_goal: str                           # 現在のプラン・タスク目標
    recalled_knowledge: List[str]             # 想起された過去Wiki知識
    conversation_history: List[Dict[str, str]]# 対話ログ
    
    # 割り込み・防衛レイヤー
    state_epoch: int                          # 先祖返り防止エポック数
    
    # プランニング制御フラグ
    next_step: str                            # グラフの遷移指示
    tool_output: str                          # 直前のツール実行結果
    next_tool_call: Optional[Dict[str, Any]]  # 次に実行するツールの呼び出し情報

class SurveillanceAgent:
    """LangGraph 1.0 に基づく、LLMエージェント統括型能動知覚 (Active Perception) 司令塔クラス。

    V2設計 §3 に準拠。
    """

    def __init__(
        self,
        tools: AgentTools,
        perception_buffer: PerceptionBuffer,
        ollama_client: OllamaVisionClient
    ) -> None:
        self.tools: AgentTools = tools
        self.perception: PerceptionBuffer = perception_buffer
        self.vlm: OllamaVisionClient = ollama_client
        
        # 非同期排他ロック
        self.lock: asyncio.Lock = asyncio.Lock()
        
        # エポック管理
        self.state_epoch: int = 0
        
        # 現在実行中の非同期推論タスクの参照
        self.current_thinking_task: Optional[asyncio.Task] = None

        # グラフの構築
        self.graph = self._build_graph()

    def _build_graph(self):
        """LangGraph 1.0 の StateGraph を用いた意思決定ノードとエッジの定義"""
        workflow = StateGraph(AgentState)

        # ノードの追加
        workflow.add_node("evaluate_situation", self.node_evaluate_situation)
        workflow.add_node("agent_planner", self.node_agent_planner)
        workflow.add_node("execute_tool", self.node_execute_tool)

        # エントリーポイント
        workflow.set_entry_point("evaluate_situation")

        # 状態評価からプランナーへの遷移
        workflow.add_edge("evaluate_situation", "agent_planner")

        # 条件付きエッジ: プランナーがツール実行を求めるか、終了するか
        workflow.add_conditional_edges(
            "agent_planner",
            self.route_after_planner,
            {
                "execute": "execute_tool",
                "end": END
            }
        )

        # ツール実行後、再度状況評価（またはプランナー）へループ
        workflow.add_edge("execute_tool", "evaluate_situation")

        return workflow.compile()

    async def update_by_user_barge_in(self, forced_track_id: int) -> None:
        """ユーザーの緊急介入（Barge-In）発生時、進行中の思考プロセスを即時中断し、エポックを更新する。

        V2設計 §4.1 のエポックガード機構の実装。
        """
        async with self.lock:
            logger.warning(f"🎤 [Barge-In Event] User forced lock-on target ID: {forced_track_id}")
            
            # 第1防衛線: 実行中のLLM推論タスクを即時キャンセル
            if self.current_thinking_task and not self.current_thinking_task.done():
                logger.warning("🔥 [Task Cancellation] Aborting active agent planning task...")
                self.current_thinking_task.cancel()

            # 第2防衛線: エポックをインクリメントし、古い書き込みを無効化
            self.state_epoch += 1
            
            # 即時物理ロックオンの反映
            self.tools.set_tracking_target(forced_track_id)
            logger.info(f"⚡ [Epoch Jump] New epoch established: {self.state_epoch}")

    async def node_evaluate_situation(self, state: AgentState) -> Dict[str, Any]:
        """内省思考ノード: 知覚バッファとWiki長期記憶を想起して統合する。"""
        # 画面内に人物がいる場合、関連するWiki知識を自動想起
        people_tracks = [t for t in state["active_tracks"] if t["class_name"] == "person"]
        recalled = []
        if people_tracks:
            # 簡易キーワードで検索
            logger.info("🧠 [Memory Recall] Person detected, searching memory store...")
            memories_text = self.tools.recall_memory("person")
            if "Memory" in memories_text:
                recalled.append(memories_text)

        return {
            "active_tracks_text": self.perception.get_active_tracks_text(),
            "recalled_knowledge": recalled
        }

    async def node_agent_planner(self, state: AgentState) -> Dict[str, Any]:
        """意思決定ノード: 目標と現状に照らし合わせ、次のツール命令をプランニングする。"""
        prompt = (
            f"あなたはTapo監視エージェントの司令塔（プランナー）です。\n"
            f"現在の状況と、これまでの記憶をもとに、次に取るべきアクションを選択してください。\n\n"
            f"--- [現在の視界メタデータ] ---\n{state['active_tracks_text']}\n\n"
            f"--- [想起された長期記憶Wiki] ---\n{chr(10).join(state['recalled_knowledge']) if state['recalled_knowledge'] else '記憶なし'}\n\n"
            f"--- [現在の目的 (Goal)] ---\n{state.get('agent_goal', '不審人物の検知および追従')}\n\n"
            f"--- [自律警戒ルール] ---\n"
            f"- 視界メタデータ内に '[⚠️WARNING ZONE DETECTED]' がついているターゲットは警戒域内にいます。\n"
            f"- 警戒域内のターゲットの確信度が低い、または詳細が不明な場合は、まず 'trigger_visual_query' ツールで対象画像（ソフトウェアズーム）を詳しく解析（VLM指示）してください。\n"
            f"- 解析結果を得た場合は、将来の監視に役立てるために 'store_memory' を使って Wiki にその結果（例: タイトル、詳細内容、タグ）を即座に保存・蓄積してください。\n\n"
            f"--- [使用可能なツールリスト] ---\n"
            f"1. set_tracking_target(track_id: int): 対象IDにカメラ視線を固定\n"
            f"2. clear_tracking_target(): カメラのターゲット固定を解除\n"
            f"3. trigger_visual_query(track_id: int, prompt: str): 対象の画像をスポットクロップ・アップスケーリング解析\n"
            f"4. store_memory(title: str, content: str, tags: str): Wikiに新たな知識をOKF Markdown形式で追加\n\n"
            f"次のJSONフォーマットのみで厳密に返答してください（他の説明テキストは一切出力しないでください）:\n"
            f'{{"action": "execute", "tool_name": "set_tracking_target", "args": {{"track_id": 1}}}}\n'
            f'または、特に対処が必要ない、あるいは完了した場合は:\n'
            f'{{"action": "end", "reason": "No actions needed"}}\n'
        )

        try:
            # VLMクライアントのテキストチャットAPIを呼ぶ (分析画像を伴わない)
            # analyze_scene を画像なし（またはダミー画像）で呼ぶか、もしくはダミー画像を用意
            dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
            response = await self.vlm.analyze_scene(dummy_img, prompt)
            
            if not response:
                return {"next_step": "end", "tool_output": "Failed to get plan from LLM."}

            logger.info(f"🧠 [Agent Plan Response]: {response.strip()}")
            
            # JSONブロックの抽出
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group(0))
                if plan.get("action") == "execute":
                    return {
                        "next_step": "execute",
                        "tool_output": "",
                        "agent_goal": f"Execute {plan.get('tool_name')}",
                        "next_tool_call": plan
                    }
            
            return {"next_step": "end", "tool_output": "Plan ended or no actions requested."}

        except Exception as e:
            logger.error(f"Error during planning: {e}")
            return {"next_step": "end", "tool_output": f"Planning error: {e}"}

    def route_after_planner(self, state: AgentState) -> str:
        """条件付きエッジルーティング。"""
        return state.get("next_step", "end")

    async def node_execute_tool(self, state: AgentState) -> Dict[str, Any]:
        """ツール実行ノード: 指示されたツールを呼び出し、結果をステートに書き込む。

        V2設計 §4.1 のエポック整合性チェックを実行前およびコミット直前に実施する。
        """
        tool_call = state.get("next_tool_call", {})
        tool_name = tool_call.get("tool_name")
        args = tool_call.get("args", {})
        task_start_epoch = state.get("state_epoch", 0)

        # --- 第1の防衛線: 実行前にエポックチェック ---
        async with self.lock:
            if self.state_epoch != task_start_epoch:
                logger.warning(
                    f"🛡️ [Epoch Guard] Aborted tool execution BEFORE call. "
                    f"Start epoch {task_start_epoch} != Current epoch {self.state_epoch}"
                )
                return {"tool_output": "Aborted due to epoch mismatch."}

        output = ""
        try:
            if tool_name == "set_tracking_target":
                output = self.tools.set_tracking_target(int(args.get("track_id", 0)))
            elif tool_name == "clear_tracking_target":
                output = self.tools.clear_tracking_target()
            elif tool_name == "trigger_visual_query":
                output = await self.tools.trigger_visual_query(int(args.get("track_id", 0)), str(args.get("prompt", "")))
            elif tool_name == "store_memory":
                output = self.tools.store_memory(str(args.get("title", "")), str(args.get("content", "")), str(args.get("tags", "")))
            else:
                output = f"Unknown tool name: {tool_name}"
        except Exception as e:
            output = f"Tool execution failed: {e}"

        logger.info(f"🔧 [Tool Output]: {output}")

        # --- エポック整合性チェック (最終書き込み防衛) ---
        async with self.lock:
            if self.state_epoch != task_start_epoch:
                logger.warning(
                    f"🛡️ [Epoch Guard] Aborted tool state commit. "
                    f"Start epoch {task_start_epoch} != Current epoch {self.state_epoch}"
                )
                return {"tool_output": "Aborted due to epoch mismatch."}

            return {"tool_output": output}

    async def step(self, current_tracks: List[Dict[str, Any]], raw_frame: np.ndarray) -> Dict[str, Any]:
        """反射ループなどから受け取った最新のメタデータとフレーム情報を元に、エージェントグラフを1ステップ駆動する。"""
        # 知覚バッファ情報をツール側に共有
        self.tools.last_raw_frame = raw_frame
        self.tools.last_active_tracks = current_tracks

        initial_state: AgentState = {
            "active_tracks": current_tracks,
            "active_tracks_text": "",
            "lockon_mode": "id" if self.tools.ptz.lock_on_id is not None else "auto",
            "target_track_id": self.tools.ptz.lock_on_id,
            "agent_goal": "Identify and track targets",
            "recalled_knowledge": [],
            "conversation_history": [],
            "state_epoch": self.state_epoch,
            "next_step": "evaluate_situation",
            "tool_output": ""
        }

        # LLMプランニングは物理反射スレッドをブロックしないよう非同期タスクとして実行
        async def run_inference():
            return await self.graph.ainvoke(initial_state)

        self.current_thinking_task = asyncio.create_task(run_inference())
        try:
            result = await self.current_thinking_task
            return result
        except asyncio.CancelledError:
            logger.info(f"🛡️ [Thinking Cancelled] Thinking task at epoch {initial_state['state_epoch']} was cancelled.")
            return {"tool_output": "Cancelled"}
