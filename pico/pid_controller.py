import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class AdaptivePIDController:
    """VOR（前庭動眼反射）規範型の非線形指数適応比例ゲインモデルを搭載したPID制御器。
    
    RelativeMove（相対移動量制御）向けに再設計され、急激な起動・急制動による
    モーター負荷を物理的に低減するためのスルーレート（加速度）制限が適用されています。
    """

    def __init__(
        self, kp_base: float = 0.35, ki: float = 0.03, kd: float = 0.01,
        dead_zone: float = 0.01, min_speed: float = 0.04, max_step: float = 0.06,
        integral_limit: float = 0.15, alpha: float = 2.0, max_acceleration: float = 0.04
    ) -> None:
        self.kp_base: float = kp_base
        self.ki: float = ki
        self.kd: float = kd
        
        self.dead_zone: float = dead_zone      # 画面中心での不感帯閾値 (0.03に絞って感度アップ)
        self.min_speed: float = min_speed      # 物理摩擦突破用ブースト (0.04)
        self.max_step: float = max_step        # 1ステップあたりの最大移動ステップ制限
        self.integral_limit: float = integral_limit  # 積分ワインドアップ防止用の制限閾値
        self.alpha: float = alpha              # 指数適応ゲインの感度制御係数
        self.max_acceleration: float = max_acceleration  # 1ステップあたりの最大加速度制限

        # 内部制御状態の初期化
        self.prev_error_x: float = 0.0
        self.prev_error_y: float = 0.0
        self.integral_x: float = 0.0
        self.integral_y: float = 0.0

        # スルーレート用：前回の出力速度値
        self.prev_output_x: float = 0.0
        self.prev_output_y: float = 0.0

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
        dead_zone_y = min(self.dead_zone, 0.01)

        if abs(error_x) < self.dead_zone:
            error_x = 0.0
            self.integral_x = 0.0
        if abs(error_y) < dead_zone_y:
            error_y = 0.0
            self.integral_y = 0.0

        if error_x == 0.0 and error_y == 0.0:
            # 停止する際も、急激な停止ではなくスルーレート（加速度制限）に従ってなめらかに減速させる
            dx, dy = 0.0, 0.0
        else:
            # ゼロ除算や不正なdtの防止
            if dt <= 0.001:
                dt = 0.2  # デフォルトのタイムステップにフォールバック

            # 2. VOR規範型の比例ゲインKp非線形指数適応 (チルト軸Yは重力抵抗克服のため2.5倍ブースト)
            kp_x = self.kp_base * (1.0 - np.exp(-self.alpha * abs(error_x)))
            kp_y = self.kp_base * 2.5 * (1.0 - np.exp(-self.alpha * abs(error_y)))

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

            # 6. 静止摩擦・物理重力突破（Minimum Speed Boost）処理
            # ターゲットが静止・中心付近(error < 0.08)に居る場合は強制ブーストを行わず、カメラを静止安定保持させる
            min_speed_y = max(0.06, self.min_speed * 1.5)
            if abs(error_x) > 0.08 and 0.0 < abs(dx) < self.min_speed:
                dx = np.sign(dx) * self.min_speed
            if abs(error_y) > 0.08 and 0.0 < abs(dy) < min_speed_y:
                dy = np.sign(dy) * min_speed_y

            # 7. 最大ステップ幅制限（クリッピング: 上下Y軸は過剰な高低振れを防ぐため 0.07 にコンパクト制限）
            max_step_y = 0.07
            dx = max(-self.max_step, min(self.max_step, dx))
            dy = max(-max_step_y, min(max_step_y, dy))

            # 状態変数の更新
            self.prev_error_x = error_x
            self.prev_error_y = error_y

        # 8. スルーレート（加速度）制限の適用
        # チルト軸(Y)は重力抵抗に打ち勝つため加速度制限を 0.10 に解放
        if self.max_acceleration > 0.0:
            diff_dx = dx - self.prev_output_x
            diff_dy = dy - self.prev_output_y

            max_acc_y = max(0.10, self.max_acceleration * 2.5)
            clamped_diff_dx = max(-self.max_acceleration, min(self.max_acceleration, diff_dx))
            clamped_diff_dy = max(-max_acc_y, min(max_acc_y, diff_dy))

            dx = self.prev_output_x + clamped_diff_dx
            dy = self.prev_output_y + clamped_diff_dy

        # 次回ループのために出力を保存
        self.prev_output_x = dx
        self.prev_output_y = dy

        return dx, dy

    def reset(self) -> None:
        """ターゲットをロストした際などに、積分値および過去の誤差履歴、スルーレート前出力を初期化する。"""
        if (self.prev_error_x == 0.0 and self.prev_error_y == 0.0 and 
                self.integral_x == 0.0 and self.integral_y == 0.0 and
                self.prev_output_x == 0.0 and self.prev_output_y == 0.0):
            return

        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_output_x = 0.0
        self.prev_output_y = 0.0
        logger.info("AdaptivePIDController state was reset.")
