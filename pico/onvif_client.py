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
        self.return_steps_x: int = return_steps_x if return_steps_x is not None else int(round(max_limit_x / step_size_x))
        self.return_steps_y: int = return_steps_y if return_steps_y is not None else int(round(max_limit_y / step_size_y))
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
        """物理限界へのブラインド突き当てを行い、camera_config.jsonの限界ステップ数に基づき中心へ復帰する高速・ロバストなアライメント。"""
        logger.info("Starting startup Blind Physical Home Alignment...")
        try:
            step_size_x = self.step_size_x
            step_size_y = self.step_size_y
            interrupted = False

            max_steps_x = self.return_steps_x
            max_steps_y = self.return_steps_y
            hunt_steps_x = self.hunt_steps_x
            hunt_steps_y = self.hunt_steps_y

            logger.info(f"Configuration limits: self.max_limit_x={self.max_limit_x}, self.max_limit_y={self.max_limit_y}")
            logger.info(f"Return steps: Center-to-Edge X={max_steps_x}, Y={max_steps_y} (step_size: X={step_size_x}, Y={step_size_y})")
            logger.info(f"Blind hunting steps to corner: X={hunt_steps_x} (LEFT), Y={hunt_steps_y} (BOTTOM)")

            # ----------------------------------------------------
            # 補助関数：指定歩数だけカメラを駆動させ、プレビューとキー中断を処理する
            # ----------------------------------------------------
            def execute_blind_move(dx: float, dy: float, total_steps: int, phase_name: str) -> None:
                nonlocal interrupted
                for i in range(total_steps):
                    if interrupted:
                        break

                    # 物理カメラの運動極性(invert_pan / invert_tilt)を反映してコマンド生成
                    # Raw ONVIF仕様 (invert_pan: True): dx < 0 (物理左) -> Raw ONVIF +0.15, dx > 0 (物理右) -> Raw ONVIF -0.15
                    cmd_x = -dx if self.invert_pan else dx
                    cmd_y = -dy if self.invert_tilt else dy

                    # コマンド送信
                    request = self.ptz.create_type('RelativeMove')
                    request.ProfileToken = self.profile_token
                    request.Translation = {'PanTilt': {'x': cmd_x, 'y': cmd_y}}
                    self.ptz.RelativeMove(request)

                    # 物理駆動ラグ待機 (物理モーターが確実に1ステップ回転する0.22秒に設定)
                    time.sleep(0.22)

                    # モニター表示 & キーキャンセル監視
                    if video_reader is not None:
                        ret, frame = video_reader.read()
                        if ret and frame is not None:
                            disp = frame.copy()
                            cv2.putText(disp, f"ALIGN: {phase_name} ({i+1}/{total_steps})", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            cv2.putText(disp, "Press [c] or [ESC] to Skip", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            cv2.imshow("Cognitive Surveillance Monitor", disp)
                            
                            key = cv2.waitKey(1) & 0xFF
                            if key in (ord('c'), 27):
                                logger.warning(f"⚠️ [{phase_name}] Cancelled by user key event.")
                                interrupted = True
                                break
                    else:
                        logger.info(f"Alignment [{phase_name}]: {i+1}/{total_steps}")

            # ----------------------------------------------------
            # 左右・上下の物理限界（両壁）へちゃんと振る「フルスキャン・バックラッシュ相殺アライメント」
            # ----------------------------------------------------
            # 1. 物理左端へ完全突き当て
            logger.info(f"PHASE 1: Sweeping to LEFT physical edge ({hunt_steps_x} steps)...")
            execute_blind_move(-step_size_x, 0.0, hunt_steps_x, "Sweeping LEFT edge")
            time.sleep(0.4)

            # 2. 物理右端へ完全突き当て（全幅確認）
            if not interrupted:
                logger.info(f"PHASE 2: Sweeping to RIGHT physical edge ({hunt_steps_x} steps)...")
                execute_blind_move(step_size_x, 0.0, hunt_steps_x, "Sweeping RIGHT edge")
                time.sleep(0.4)

            # 3. 物理下端へ完全突き当て
            if not interrupted:
                logger.info(f"PHASE 3: Sweeping to BOTTOM physical edge ({hunt_steps_y} steps)...")
                execute_blind_move(0.0, -step_size_y, hunt_steps_y, "Sweeping BOTTOM edge")
                time.sleep(0.4)

            # 4. 物理上端へ完全突き当て（全高確認）
            if not interrupted:
                logger.info(f"PHASE 4: Sweeping to TOP physical edge ({hunt_steps_y} steps)...")
                execute_blind_move(0.0, step_size_y, hunt_steps_y, "Sweeping TOP edge")
                time.sleep(0.4)

            # 5. バックラッシュ(ギア遊び)を完全相殺するため、左下物理基準点へ突き当てて一方向運動に揃える
            if not interrupted:
                logger.info(f"PHASE 5: Resetting to LEFT-BOTTOM physical corner for zero-backlash...")
                execute_blind_move(-step_size_x, 0.0, hunt_steps_x, "Resetting LEFT for Backlash cancel")
                time.sleep(0.4)
                execute_blind_move(0.0, -step_size_y, hunt_steps_y, "Resetting BOTTOM for Backlash cancel")
                time.sleep(0.5)

            # 6. 左下物理基準点から、正確に実測総幅の半分 (右へ7歩 / 上へ10歩) だけ移動して【真の中心原点】へ着地
            if not interrupted:
                logger.info(f"PHASE 6: Returning to EXACT CENTER from corner in {max_steps_x} steps RIGHT, {max_steps_y} steps UP...")
                execute_blind_move(step_size_x, 0.0, max_steps_x, "Returning RIGHT to Center X")
                time.sleep(0.3)
                execute_blind_move(0.0, step_size_y, max_steps_y, "Returning UP to Center Y")
                time.sleep(0.4)

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
        """現在の推測位置と限界値を考慮し、安全な範囲にクランプして移動コマンドを送信する。

        Returns:
            Tuple[float, float]: 実際に送信されたクランプ後の移動量 (x, y)。
        """
        next_x = self.current_x + requested_x
        next_y = self.current_y + requested_y
        actual_move_x, actual_move_y = requested_x, requested_y

        # X軸（左右）のクランプ
        if next_x > self.max_limit_x:
            actual_move_x = self.max_limit_x - self.current_x
        elif next_x < -self.max_limit_x:
            actual_move_x = -self.max_limit_x - self.current_x

        # Y軸（上下）のクランプ
        if next_y > self.max_limit_y:
            actual_move_y = self.max_limit_y - self.current_y
        elif next_y < -self.max_limit_y:
            actual_move_y = -self.max_limit_y - self.current_y

        if abs(actual_move_x) > 0.001 or abs(actual_move_y) > 0.001:
            # 物理カメラに命令を送信するタイミングで反転を適用
            cmd_x = -actual_move_x if self.invert_pan else actual_move_x
            cmd_y = -actual_move_y if self.invert_tilt else actual_move_y
            self.relative_move(cmd_x, cmd_y)
            self.current_x += actual_move_x
            self.current_y += actual_move_y
            logger.info(
                f"PTZ Safe Move: x={actual_move_x:+.3f}, y={actual_move_y:+.3f} (cmd_x={cmd_x:+.3f}, cmd_y={cmd_y:+.3f}) | "
                f"Estimated Pos: X={self.current_x:+.2f}, Y={self.current_y:+.2f}"
            )
            return actual_move_x, actual_move_y
            
        return 0.0, 0.0

    def move_to_center(self) -> Tuple[float, float]:
        """現在の推測位置から原点(0,0)へ復帰移動する。"""
        return self.safe_move(-self.current_x, -self.current_y)

    def shutdown(self) -> None:
        """非同期スレッドを停止し、リソースをクリーンアップする。"""
        self.running = False
        self.move_queue.put(None)
        self.worker_thread.join(timeout=0.5)
        logger.info("PTZController shut down successfully.")
