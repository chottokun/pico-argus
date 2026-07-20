import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class AdaptivePIDController:
    """VOR（前庭動眼反射）規範型の非線形指数適応比例ゲインモデルを搭載したPID制御器。
    
    RelativeMove（相対移動量制御）向けに再設計されています。
    """

    def __init__(
        self, kp_base: float = 0.5, ki: float = 0.05, kd: float = 0.01,
        dead_zone: float = 0.05, min_speed: float = 0.03, max_step: float = 0.12,
        integral_limit: float = 0.30, alpha: float = 2.0
    ) -> None:
        self.kp_base: float = kp_base
        self.ki: float = ki
        self.kd: float = kd
        
        self.dead_zone: float = dead_zone      # 画面中心での不感帯閾値
        self.min_speed: float = min_speed      # 静止摩擦を突破する最小移動速度ブースト
        self.max_step: float = max_step        # 1ステップあたりの最大移動ステップ制限
        self.integral_limit: float = integral_limit  # 積分ワインドアップ防止用の制限閾値
        self.alpha: float = alpha              # 指数適応ゲインの感度制御係数

        # 内部制御状態の初期化
        self.prev_error_x: float = 0.0
        self.prev_error_y: float = 0.0
        self.integral_x: float = 0.0
        self.integral_y: float = 0.0

    def calculate_step(self, target_cx: float, target_cy: float, dt: float) -> Tuple[float, float]:
        """ターゲット重心の正規化座標(0.0〜1.0)と実測時間差dtから、安全かつ最適なPTZ相対移動量を算出する。

        Args:
            target_cx (float): ターゲット中心のX座標(0.0〜1.0)。
            target_cy (float): ターゲット中心のY座標(0.0〜1.0)。
            dt (float): 前回の制御タイミングからの時間経過（秒）。

        Returns:
            Tuple[float, float]: 算出されたX軸、Y軸の相対移動量 (dx, dy)。
        """
        # 画角中心(0.5, 0.5)からの偏差
        error_x = target_cx - 0.5
        error_y = target_cy - 0.5

        # 1. 不感帯（Dead Zone）判定
        if abs(error_x) < self.dead_zone:
            error_x = 0.0
            self.integral_x = 0.0
        if abs(error_y) < self.dead_zone:
            error_y = 0.0
            self.integral_y = 0.0

        if error_x == 0.0 and error_y == 0.0:
            return 0.0, 0.0

        # ゼロ除算や不正なdtの防止
        if dt <= 0.001:
            dt = 0.2  # デフォルトのタイムステップにフォールバック

        # 2. VOR規範型の比例ゲインKp非線形指数適応
        # 偏差が小さい時はゲインを落とし、画面端に近づくほど基本感度Kp_baseまで急激に高める
        kp_x = self.kp_base * (1.0 - np.exp(-self.alpha * abs(error_x)))
        kp_y = self.kp_base * (1.0 - np.exp(-self.alpha * abs(error_y)))

        # 3. 積分項（累積誤差）の加算とアンチワインドアップ（クランプ）
        self.integral_x += error_x * dt
        self.integral_y += error_y * dt
        
        self.integral_x = max(-self.integral_limit, min(self.integral_limit, self.integral_x))
        self.integral_y = max(-self.integral_limit, min(self.integral_limit, self.integral_y))

        # 4. 微分項（偏差の変化速度）の算出
        diff_x = (error_x - self.prev_error_x) / dt
        diff_y = (error_y - self.prev_error_y) / dt

        # 5. PID出力の統合
        dx = (kp_x * error_x) + (self.ki * self.integral_x) + (self.kd * diff_x)
        dy = (kp_y * error_y) + (self.ki * self.integral_y) + (self.kd * diff_y)

        # 6. 静止摩擦突破（Minimum Speed Boost）処理
        # 出力絶対値が微小で0でない場合、モーターが動き出せる最小値(min_speed)へ押し上げる
        if 0.0 < abs(dx) < self.min_speed:
            dx = np.sign(dx) * self.min_speed
        if 0.0 < abs(dy) < self.min_speed:
            dy = np.sign(dy) * self.min_speed

        # 7. 最大ステップ幅制限（クリッピング）
        dx = max(-self.max_step, min(self.max_step, dx))
        dy = max(-self.max_step, min(self.max_step, dy))

        # 状態変数の更新
        self.prev_error_x = error_x
        self.prev_error_y = error_y

        return dx, dy

    def reset(self) -> None:
        """ターゲットをロストした際などに、積分値および過去の誤差履歴を初期化する。"""
        # すでにリセット済みならログを出さずに早期リターン
        if (self.prev_error_x == 0.0 and self.prev_error_y == 0.0 and 
                self.integral_x == 0.0 and self.integral_y == 0.0):
            return

        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0
        logger.info("AdaptivePIDController state was reset.")

