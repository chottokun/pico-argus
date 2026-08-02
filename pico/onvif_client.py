import os
import threading
import queue
import time
import logging
from typing import Tuple, Optional
import cv2
from onvif import ONVIFCamera
from pico.video_reader import RTSPVideoReader

logger = logging.getLogger(__name__)

class PTZController:
    """ONVIF 規格を用いて Tapo カメラの PTZ (Pan/Tilt) 制御を非同期で安全に行うコントローラークラス。"""

    def __init__(
        self, ip: str, user: str, password: str, port: int = 2020,
        max_limit_x: float = 1.0, max_limit_y: float = 0.5,
        align_to_home: bool = False,
        video_reader: Optional[RTSPVideoReader] = None,
        invert_pan: bool = False,
        invert_tilt: bool = False,
        step_size_x: float = 0.15,
        step_size_y: float = 0.10,
        total_steps_x: int = 15,
        total_steps_y: int = 20,
        return_steps_x: Optional[int] = None,
        return_steps_y: Optional[int] = None,
        hunt_steps_x: int = 25,
        hunt_steps_y: int = 25
    ) -> None:
        self.ip: str = ip
        self.user: str = user
        self.password: str = password
        self.port: int = port
        
        # 可動限界値と反転フラグ、アライメントステップパラメータ
        self.max_limit_x: float = max_limit_x
        self.max_limit_y: float = max_limit_y
        self.invert_pan: bool = invert_pan
        self.invert_tilt: bool = invert_tilt
        self.step_size_x: float = step_size_x
        self.step_size_y: float = step_size_y
        self.total_steps_x: int = total_steps_x
        self.total_steps_y: int = total_steps_y
        self.return_steps_x: int = return_steps_x if return_steps_x is not None else (total_steps_x // 2 + 1 if total_steps_x % 2 != 0 else total_steps_x // 2)
        self.return_steps_y: int = return_steps_y if return_steps_y is not None else total_steps_y // 2
        self.hunt_steps_x: int = hunt_steps_x
        self.hunt_steps_y: int = hunt_steps_y
        
        # 現在の推測位置 (0.0, 0.0 はキャリブレーション後の真の中心原点と仮定)
        self.current_x: float = 0.0
        self.current_y: float = 0.0

        # 能動的認知制御用: VLMに指示されたロックオン対象IDと追尾ルール
        self.lock_on_id: Optional[int] = None
        self.target_rule: str = "a person wearing a hat"

        # ONVIF 接続の初期化
        try:
            self.mycam: ONVIFCamera = ONVIFCamera(self.ip, self.port, self.user, self.password)
            self.ptz = self.mycam.create_type if False else self.mycam.create_ptz_service()
            self.media = self.mycam.create_media_service()
            
            # PTZ 設定を持つ最初のプロファイルトークンを自動探索
            profiles = self.media.GetProfiles()
            self.profile_token: str = next(
                (p.token for p in profiles if hasattr(p, 'PTZConfiguration') and p.PTZConfiguration is not None),
                profiles[0].token
            )
            logger.info("ONVIF camera initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ONVIF connection to {self.ip}:{self.port} - {e}")
            raise

        # 起動時に真の中心（原点）に強制復帰するアライメント処理
        if align_to_home:
            self._align_to_home_position(video_reader)

        # スレッド安全なコマンド送信用キュー (最大サイズ1で最新の命令のみ保持)
        self.move_queue: queue.Queue = queue.Queue(maxsize=1)
        self.running: bool = True
        
        self.worker_thread: threading.Thread = threading.Thread(target=self._ptz_worker, daemon=True)
        self.worker_thread.start()

    def _align_to_home_position(self, video_reader: Optional[RTSPVideoReader] = None) -> None:
        """起動時のアライメント処理。calibrate_home を直接呼び出して100%同一のアライメントを実行する。"""
        self.calibrate_home(video_reader=video_reader)

    def calibrate_home(self, video_reader: Optional[RTSPVideoReader] = None) -> Tuple[float, float]:
        """物理限界へのブラインド突き当てを行い、カメラの真の中心原点(0.0, 0.0)に再校正・復帰するアライメント。"""
        logger.info("Starting Physical Home Alignment (calibrate_home)...")
        try:
            step_size_x = self.step_size_x
            step_size_y = self.step_size_y
            interrupted = False

            steps_to_center_x = getattr(self, "return_steps_x", 7)
            steps_to_center_y = getattr(self, "return_steps_y", 11)

            # ----------------------------------------------------
            # 🎯 ビジュアルモーションモニタリング付き 4方向アライメント (ゆったり1.2秒待機)
            # ----------------------------------------------------
            def move_and_monitor(cmd_x: float, cmd_y: float, steps: int, phase_name: str) -> None:
                nonlocal interrupted
                def extract_frame():
                    if video_reader is None:
                        return None
                    if hasattr(video_reader, "get_latest_frame"):
                        return video_reader.get_latest_frame()
                    if hasattr(video_reader, "read"):
                        return video_reader.read()[1]
                    return None

                for i in range(steps):
                    if interrupted:
                        break
                    prev_frame = extract_frame()
                    
                    req = self.ptz.create_type('RelativeMove')
                    req.ProfileToken = self.profile_token
                    req.Translation = {'PanTilt': {'x': cmd_x, 'y': cmd_y}}
                    self.ptz.RelativeMove(req)
                    time.sleep(1.2)  # RTSP駆動およびSOAPバッファ消化のゆったり確実な待機 (1.2秒)
                    
                    curr_frame = extract_frame()
                    if prev_frame is not None and curr_frame is not None:
                        try:
                            g1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                            g2 = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                            diff = cv2.absdiff(g1, g2)
                            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                            motion_ratio = (cv2.countNonZero(thresh) / (thresh.shape[0] * thresh.shape[1])) * 100.0
                            status = "MOVED ✅" if motion_ratio >= 0.8 else "WALL/STOP 🛑"
                            logger.info(f"[{phase_name}] Step {i+1}/{steps} -> {status} (Visual Motion: {motion_ratio:.2f}%)")
                        except Exception:
                            logger.info(f"[{phase_name}] Step {i+1}/{steps}")
                    else:
                        logger.info(f"[{phase_name}] Step {i+1}/{steps}")

            # 1. 右の物理限界壁へ移動 (右端突き当て: ONVIF +0.15)
            logger.info("PHASE 1-A: Resetting to RIGHT physical edge...")
            move_and_monitor(+step_size_x, 0.0, 15, "PHASE 1-A (Right)")

            # 2. 下の物理限界壁へ移動 (下端突き当て: ONVIF -0.10)
            if not interrupted:
                logger.info("PHASE 1-B: Resetting to BOTTOM physical edge...")
                move_and_monitor(0.0, -step_size_y, 15, "PHASE 1-B (Bottom)")

            # 3. 左の物理限界壁へ移動 (左端突き当て: ONVIF -0.15)
            if not interrupted:
                logger.info("PHASE 2-A: Resetting to LEFT physical edge...")
                move_and_monitor(-step_size_x, 0.0, 15, "PHASE 2-A (Left)")

            # 4. 上の物理限界壁へ移動 (上端突き当て: ONVIF +0.10)
            if not interrupted:
                logger.info("PHASE 2-B: Resetting to TOP physical edge...")
                move_and_monitor(0.0, +step_size_y, 15, "PHASE 2-B (Top)")

            # 5. 左端から【右(RIGHT)】へ 7歩 移動 (ONVIF +0.15 で右へ)
            if not interrupted:
                logger.info(f"PHASE 3-A: Returning RIGHT to Center X ({steps_to_center_x} steps)...")
                move_and_monitor(+step_size_x, 0.0, steps_to_center_x, "PHASE 3-A (Return Right)")

            # 6. 上端から【下(DOWN)】へ重力を利用して 10歩 降りて精密中央着地 (ONVIF -0.10 で下へ)
            if not interrupted:
                logger.info(f"PHASE 3-B: Descending DOWN to Center Y ({steps_to_center_y} steps)...")
                move_and_monitor(0.0, -step_size_y, steps_to_center_y, "PHASE 3-B (Descend Down)")

            self.current_x = 0.0
            self.current_y = 0.0
            if interrupted:
                logger.warning("Home Alignment was manually bypassed/interrupted. Current position established as (0.0, 0.0).")
            else:
                logger.info("Home Alignment completed successfully. Exact origin established at (0.0, 0.0).")
            # ウィンドウ破棄は行わず、MonitorWindowスレッドと同一ウィンドウを単一維持する
        except Exception as e:
            logger.error(f"Failed to perform Home Alignment: {e}")

    def _ptz_worker(self) -> None:
        """キューから移動コマンドを受け取り、カメラを駆動するバックグラウンドスレッド。"""
        while self.running:
            try:
                command = self.move_queue.get(timeout=0.1)
                if command is None:
                    break
                
                x, y = command
                try:
                    request = self.ptz.create_type('RelativeMove')
                    request.ProfileToken = self.profile_token
                    request.Translation = {'PanTilt': {'x': x, 'y': y}}
                    self.ptz.RelativeMove(request)
                    
                    # 物理駆動のラグを考慮して少し待機
                    time.sleep(0.15)
                except Exception as e:
                    logger.error(f"Error during physical camera movement: {e}")
                finally:
                    self.move_queue.task_done()
            except queue.Empty:
                continue

    def relative_move(self, x: float, y: float) -> None:
        """非ブロッキングで相対移動コマンドをキューへ追加する。"""
        if not self.running:
            return
            
        if self.move_queue.full():
            try:
                self.move_queue.get_nowait()
            except queue.Empty:
                pass
        self.move_queue.put((x, y))

    def safe_move(self, requested_x: float, requested_y: float) -> Tuple[float, float]:
        """物理PTZ制御のためのスルー透過＆ガイド型クランプ。
        ソフトウェアの過剰クランプによる移動命令のフリーズ・殺しを排出し、
        物理カメラへ素直に相対移動コマンドを発行する。
        """
        actual_move_x, actual_move_y = requested_x, requested_y

        if abs(actual_move_x) > 0.0005 or abs(actual_move_y) > 0.0005:
            # 物理カメラに命令を送信するタイミングで反転を適用（X/Y対称設計）
            cmd_x = -actual_move_x if self.invert_pan else actual_move_x
            cmd_y = -actual_move_y if self.invert_tilt else actual_move_y
            self.relative_move(cmd_x, cmd_y)
            
            # 推測位置ガイドの更新（ソフトガイドとして安全可動域内に収める）
            self.current_x = max(-self.max_limit_x, min(self.max_limit_x, self.current_x + actual_move_x))
            self.current_y = max(-self.max_limit_y, min(self.max_limit_y, self.current_y + actual_move_y))
            
            logger.info(
                f"PTZ Move: x={actual_move_x:+.3f}, y={actual_move_y:+.3f} (cmd_x={cmd_x:+.3f}, cmd_y={cmd_y:+.3f}) | "
                f"Pos Guide: X={self.current_x:+.2f}, Y={self.current_y:+.2f}"
            )
            return actual_move_x, actual_move_y
            
        return 0.0, 0.0

    def move_to_center(self) -> Tuple[float, float]:
        """現在の推測位置から原点(0,0)へ復帰移動する。"""
        res = self.safe_move(-self.current_x, -self.current_y)
        self.current_x = 0.0
        self.current_y = 0.0
        return res

    def shutdown(self) -> None:
        """非同期スレッドを停止し、リソースをクリーンアップする。"""
        self.running = False
        self.move_queue.put(None)
        self.worker_thread.join(timeout=0.5)
        logger.info("PTZController shut down successfully.")
