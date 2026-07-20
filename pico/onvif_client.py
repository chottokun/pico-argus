import threading
import queue
import time
import logging
from typing import Tuple
from onvif import ONVIFCamera

logger = logging.getLogger(__name__)

class PTZController:
    """ONVIF 規格を用いて Tapo カメラの PTZ (Pan/Tilt) 制御を非同期で安全に行うコントローラークラス。"""

    def __init__(
        self, ip: str, user: str, password: str, port: int = 2020,
        max_limit_x: float = 1.0, max_limit_y: float = 0.5,
        align_to_home: bool = False
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
            self._align_to_home_position()

        # スレッド安全なコマンド送信用キュー (最大サイズ1で最新の命令のみ保持)
        self.move_queue: queue.Queue = queue.Queue(maxsize=1)
        self.running: bool = True
        
        self.worker_thread: threading.Thread = threading.Thread(target=self._ptz_worker, daemon=True)
        self.worker_thread.start()

    def _align_to_home_position(self) -> None:
        """カメラを最も左下（物理限界）まで駆動させ、そこから設定リミット値分戻して真の中心(0.0, 0.0)に位置合わせする。"""
        logger.info("Starting startup Home Alignment to secure origin...")
        try:
            # 1. 最も左・下へ強制追い込み (安全のため十分な幅を複数ステップに分けて送信)
            # キャリブレーションが X: 1.08, Y: 0.85 なので、1.5以上動かせば確実に端に突き当たる
            # 一括で大きな値を送るより、安定のために 2ステップに分ける
            for _ in range(2):
                request = self.ptz.create_type('RelativeMove')
                request.ProfileToken = self.profile_token
                # Xは左がマイナス、Yは下がマイナス
                request.Translation = {'PanTilt': {'x': -1.2, 'y': -1.2}}
                self.ptz.RelativeMove(request)
                time.sleep(1.2)  # カメラが動き切るまで待つ

            # 2. 端に突き当たった状態から、キャリブレーションで定義された限界幅分だけ逆方向（右・上）に動かす
            # これにより「真の中心（原点）」に正確にアライメントされる
            request = self.ptz.create_type('RelativeMove')
            request.ProfileToken = self.profile_token
            # 端から中心までの距離 ≒ リミット値をマージンでデスケールした物理ステップ
            # （キャリブ限界はマージン85%なので、0.85で割って戻すことで真の物理中心へ補正）
            target_x = self.max_limit_x / 0.85
            target_y = self.max_limit_y / 0.85
            
            request.Translation = {'PanTilt': {'x': target_x, 'y': target_y}}
            self.ptz.RelativeMove(request)
            time.sleep(1.5)

            # 現在の推測位置を 0.0 にリセット
            self.current_x = 0.0
            self.current_y = 0.0
            logger.info(f"Home Alignment completed. Position calibrated to origin (0.0, 0.0). Target values were X: {target_x:.2f}, Y: {target_y:.2f}")
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
