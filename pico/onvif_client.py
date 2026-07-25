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
        invert_tilt: bool = False
    ) -> None:
        self.ip: str = ip
        self.user: str = user
        self.password: str = password
        self.port: int = port
        
        # 可動限界値と反転フラグ
        self.max_limit_x: float = max_limit_x
        self.max_limit_y: float = max_limit_y
        self.invert_pan: bool = invert_pan
        self.invert_tilt: bool = invert_tilt
        
        # 現在の推測位置 (0.0, 0.0 はキャリブレーション後の真の中心原点と仮定)
        self.current_x: float = 0.0
        self.current_y: float = 0.0

        # 能動的認知制御用: VLMに指示されたロックオン対象IDと追尾ルール
        self.lock_on_id: Optional[int] = None
        self.target_rule: str = "a person wearing a hat"

        # ONVIF 接続の初期化
        try:
            self.mycam: ONVIFCamera = ONVIFCamera(self.ip, self.port, self.user, self.password)
            self.ptz = self.mycam.create_ptz_service()
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
        """物理限界へのブラインド突き当てを行い、tapo_config.jsonの限界値に基づき中心へ復帰する高速・ロバストなアライメント。"""
        logger.info("Starting startup Blind Physical Home Alignment...")
        try:
            step_size_x = 0.15
            step_size_y = 0.10
            interrupted = False

            # 設定値(tapo_config.json)から中心から端までの最大ステップ数を逆算
            max_steps_x = int(round(self.max_limit_x / step_size_x))
            max_steps_y = int(round(self.max_limit_y / step_size_y))

            # バリデーションとフォールバック
            if max_steps_x <= 0 or max_steps_x > 20:
                max_steps_x = 8
            if max_steps_y <= 0 or max_steps_y > 20:
                max_steps_y = 10

            # どのような初期位置からでも確実に突き当たるように、全幅分(片側最大幅の2倍) + マージン2歩を算出
            hunt_steps_x = (max_steps_x * 2) + 2
            hunt_steps_y = (max_steps_y * 2) + 2

            logger.info(f"Configuration limits: self.max_limit_x={self.max_limit_x}, self.max_limit_y={self.max_limit_y}")
            logger.info(f"Target steps: Center-to-Edge X={max_steps_x}, Y={max_steps_y}")
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
                    cmd_x = -dx if self.invert_pan else dx
                    cmd_y = -dy if self.invert_tilt else dy

                    # コマンド送信
                    request = self.ptz.create_type('RelativeMove')
                    request.ProfileToken = self.profile_token
                    request.Translation = {'PanTilt': {'x': cmd_x, 'y': cmd_y}}
                    self.ptz.RelativeMove(request)

                    # 物理駆動ラグ待機 (安定動作速度 0.18秒)
                    time.sleep(0.18)

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

            # 1. 左端への追い込み（ブラインド突き当て）
            logger.info("PHASE 1: Hunting LEFT physical edge...")
            execute_blind_move(-step_size_x, 0.0, hunt_steps_x, "Hunting LEFT edge")
            time.sleep(0.5)

            # 2. 下端への追い込み（ブラインド突き当て）
            if not interrupted:
                logger.info("PHASE 2: Hunting BOTTOM physical edge...")
                execute_blind_move(0.0, -step_size_y, hunt_steps_y, "Hunting BOTTOM edge")
                time.sleep(0.5)

            # 3. 設定値に基づき「右・上」に戻して真の中心にアライメント
            if not interrupted:
                logger.info(f"PHASE 3: Returning to Center by X: {max_steps_x} steps (RIGHT)...")
                execute_blind_move(step_size_x, 0.0, max_steps_x, "Returning Center X")
                time.sleep(0.5)

            if not interrupted:
                logger.info(f"PHASE 4: Returning to Center by Y: {max_steps_y} steps (UP)...")
                execute_blind_move(0.0, step_size_y, max_steps_y, "Returning Center Y")

            time.sleep(1.0)
            self.current_x = 0.0
            self.current_y = 0.0
            if interrupted:
                logger.warning("Home Alignment was manually bypassed/interrupted. Current position established as (0.0, 0.0).")
            else:
                logger.info("Home Alignment completed successfully. Origin established.")
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

    def shutdown(self) -> None:
        """非同期スレッドを停止し、リソースをクリーンアップする。"""
        self.running = False
        self.move_queue.put(None)
        self.worker_thread.join(timeout=0.5)
        logger.info("PTZController shut down successfully.")
