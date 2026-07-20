import threading
import queue
import time
import logging
from typing import Tuple, Optional
import cv2
import numpy as np
from onvif import ONVIFCamera
from pico.video_reader import RTSPVideoReader

logger = logging.getLogger(__name__)

class PTZController:
    """ONVIF 規格を用いて Tapo カメラの PTZ (Pan/Tilt) 制御を非同期で安全に行うコントローラークラス。"""

    def __init__(
        self, ip: str, user: str, password: str, port: int = 2020,
        max_limit_x: float = 1.0, max_limit_y: float = 0.5,
        align_to_home: bool = False,
        video_reader: Optional[RTSPVideoReader] = None
    ) -> None:
        self.ip: str = ip
        self.user: str = user
        self.password: str = password
        self.port: int = port
        
        # 可動限界値
        self.max_limit_x: float = max_limit_x
        self.max_limit_y: float = max_limit_y
        
        # 現在の推測位置 (0.0, 0.0 はキャリブレーション後の真の中心原点と仮定)
        self.current_x: float = 0.0
        self.current_y: float = 0.0

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
        """カメラを左右・上下に振ることで物理的な真の中心(0.0, 0.0)に位置合わせする動的アライメント処理。"""
        logger.info("Starting startup Dynamic Home Alignment...")
        try:
            step_size_x = 0.15
            step_size_y = 0.10
            wait_time = 1.4
            max_attempts = 25

            # ----------------------------------------------------
            # 補助関数：指定方向に動きが止まるまでカメラを駆動させる
            # ----------------------------------------------------
            def move_until_limit(dx: float, dy: float, phase_name: str) -> int:
                steps = 0
                for attempt in range(max_attempts):
                    before_frame = None
                    if video_reader is not None:
                        ret, frame_data = video_reader.read()
                        if ret and frame_data is not None:
                            before_frame = cv2.cvtColor(frame_data.copy(), cv2.COLOR_BGR2GRAY)

                    request = self.ptz.create_type('RelativeMove')
                    request.ProfileToken = self.profile_token
                    request.Translation = {'PanTilt': {'x': dx, 'y': dy}}
                    self.ptz.RelativeMove(request)

                    # 待機しながらプレビュー更新
                    if video_reader is not None:
                        start_w = time.monotonic()
                        frame_drawn = False
                        while time.monotonic() - start_w < wait_time:
                            ret, frame_current = video_reader.read()
                            if ret and frame_current is not None:
                                disp = frame_current.copy()
                                cv2.putText(disp, f"ALIGN: {phase_name} ({attempt+1})", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                cv2.imshow("Tapo ONVIF YOLOv8 Experiment System", disp)
                                cv2.waitKey(1)
                                frame_drawn = True
                            time.sleep(0.05)
                        if not frame_drawn:
                            time.sleep(wait_time)
                    else:
                        time.sleep(0.18)

                    steps += 1

                    # 動き停止検知
                    if video_reader is not None and before_frame is not None:
                        ret, frame_after = video_reader.read()
                        if ret and frame_after is not None:
                            after_gray = cv2.cvtColor(frame_after.copy(), cv2.COLOR_BGR2GRAY)
                            diff = cv2.absdiff(before_frame, after_gray)
                            _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
                            moved_ratio = (np.sum(thresh == 255) / thresh.size) * 100
                            
                            logger.info(f"Alignment [{phase_name}]: Step {steps} | Motion: {moved_ratio:.2f}%")
                            
                            # 最初の4ステップ以降で動きが3%未満なら停止と判定
                            if moved_ratio < 3.0 and attempt >= 4:
                                logger.info(f"🛑 [{phase_name}] Limit detected at step {steps}.")
                                break
                    else:
                        # モニタがない（テスト等）場合は、限界追い込みは12ステップで固定
                        if attempt >= 12:
                            break
                return steps

            # ----------------------------------------------------
            # アライメント実行
            # ----------------------------------------------------
            if video_reader is not None:
                # 1. 左端への追い込み
                logger.info("PHASE 1: Hunting LEFT edge...")
                move_until_limit(-step_size_x, 0.0, "Hunting LEFT edge")
                time.sleep(1.0)

                # 2. 右端への駆動 ＆ 左右全幅計測
                logger.info("PHASE 2: Measuring RIGHT width...")
                total_steps_x = move_until_limit(step_size_x, 0.0, "Measuring RIGHT width")
                time.sleep(1.0)

                # 3. 下端への追い込み
                logger.info("PHASE 3: Hunting BOTTOM edge...")
                move_until_limit(0.0, -step_size_y, "Hunting BOTTOM edge")
                time.sleep(1.0)

                # 4. 上端への駆動 ＆ 上下全幅計測
                logger.info("PHASE 4: Measuring TOP width...")
                total_steps_y = move_until_limit(0.0, step_size_y, "Measuring TOP width")
                time.sleep(1.0)

                # 復帰数の決定
                center_steps_x = total_steps_x // 2
                center_steps_y = total_steps_y // 2
                logger.info(f"Scan complete. Width X: {total_steps_x} steps, Height Y: {total_steps_y} steps.")
                logger.info(f"Returning to Center by X: {center_steps_x} steps, Y: {center_steps_y} steps...")
            else:
                # テストなどのフォールバック処理（ブラインド固定値）
                center_steps_x = 8
                center_steps_y = 10
                logger.info("Fallback Alignment (Blind Center Return).")

            # 5. 真の原点への復帰移動
            # 現在「右端かつ上端」にいるため、左(-x)、下(-y)に戻すことで中心になります
            for i in range(center_steps_x):
                request = self.ptz.create_type('RelativeMove')
                request.ProfileToken = self.profile_token
                request.Translation = {'PanTilt': {'x': -step_size_x, 'y': 0.0}}
                self.ptz.RelativeMove(request)
                
                if video_reader is not None:
                    start_w = time.monotonic()
                    frame_drawn = False
                    while time.monotonic() - start_w < 1.0:
                        ret, frame_current = video_reader.read()
                        if ret and frame_current is not None:
                            disp = frame_current.copy()
                            cv2.putText(disp, f"ALIGN: Returning Center X ({i+1}/{center_steps_x})", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            cv2.imshow("Tapo ONVIF YOLOv8 Experiment System", disp)
                            cv2.waitKey(1)
                            frame_drawn = True
                        time.sleep(0.05)
                    if not frame_drawn:
                        time.sleep(1.0)
                else:
                    time.sleep(0.18)

            for i in range(center_steps_y):
                request = self.ptz.create_type('RelativeMove')
                request.ProfileToken = self.profile_token
                request.Translation = {'PanTilt': {'x': 0.0, 'y': -step_size_y}}
                self.ptz.RelativeMove(request)
                
                if video_reader is not None:
                    start_w = time.monotonic()
                    frame_drawn = False
                    while time.monotonic() - start_w < 1.0:
                        ret, frame_current = video_reader.read()
                        if ret and frame_current is not None:
                            disp = frame_current.copy()
                            cv2.putText(disp, f"ALIGN: Returning Center Y ({i+1}/{center_steps_y})", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            cv2.imshow("Tapo ONVIF YOLOv8 Experiment System", disp)
                            cv2.waitKey(1)
                            frame_drawn = True
                        time.sleep(0.05)
                    if not frame_drawn:
                        time.sleep(1.0)
                else:
                    time.sleep(0.18)

            time.sleep(1.0)
            self.current_x = 0.0
            self.current_y = 0.0
            logger.info("Home Alignment completed successfully. Origin established.")
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
            self.relative_move(actual_move_x, actual_move_y)
            self.current_x += actual_move_x
            self.current_y += actual_move_y
            logger.info(
                f"PTZ Safe Move: x={actual_move_x:+.3f}, y={actual_move_y:+.3f} | "
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
