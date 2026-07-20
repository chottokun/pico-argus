import pytest
from pico.pid_controller import AdaptivePIDController

def test_pid_dead_zone() -> None:
    # 不感帯 0.05
    controller = AdaptivePIDController(
        kp_base=0.5, ki=0.05, kd=0.01,
        dead_zone=0.05, min_speed=0.03, max_step=0.12
    )
    # 偏差が不感帯(0.05)未満の場合、出力は 0.0 になること
    dx, dy = controller.calculate_step(target_cx=0.52, target_cy=0.49, dt=0.2)
    assert dx == 0.0
    assert dy == 0.0

def test_pid_nonlinear_gain_and_output_limit() -> None:
    controller = AdaptivePIDController(
        kp_base=0.5, ki=0.0, kd=0.0,
        dead_zone=0.01, min_speed=0.0, max_step=0.10
    )
    # 偏差大 (cx=1.0 -> error=0.5)
    # P制御出力は kp_nonlinear * error
    # kp_nonlinear = kp_base * (1.0 - exp(-2.0 * |error|))
    # |error| = 0.5 のとき 1.0 - exp(-1.0) ≒ 0.632
    # kp_nonlinear = 0.5 * 0.632 = 0.316
    # 出力は 0.316 * 0.5 = 0.158 -> max_step(0.10) でクランプされるはず
    dx, dy = controller.calculate_step(target_cx=1.0, target_cy=0.5, dt=0.2)
    assert pytest.approx(dx, 0.01) == 0.10
    assert dy == 0.0

def test_pid_anti_windup() -> None:
    controller = AdaptivePIDController(
        kp_base=0.0, ki=0.5, kd=0.0,
        dead_zone=0.01, min_speed=0.0, max_step=0.15,
        integral_limit=0.10  # 積分制限
    )
    
    # 偏差を同じ方向に与え続け、積分値を蓄積させる
    # dt=0.2, error=0.1 のとき, 累積積分値 = 0.1 * 0.2 = 0.02
    # 何回も繰り返して制限(0.10)に達するかテスト
    for _ in range(10):
        dx, _ = controller.calculate_step(target_cx=0.6, target_cy=0.5, dt=0.2)
    
    # ki * integral_x = 0.5 * 0.10 = 0.05 (制限によりこれ以上増えない)
    assert pytest.approx(dx, 0.01) == 0.05

def test_pid_reset() -> None:
    controller = AdaptivePIDController(
        kp_base=0.0, ki=0.5, kd=0.0,
        dead_zone=0.01, min_speed=0.0, max_step=0.15
    )
    
    # 積分値を蓄積
    controller.calculate_step(target_cx=0.6, target_cy=0.5, dt=0.2)
    assert controller.integral_x > 0.0
    
    # リセット
    controller.reset()
    assert controller.integral_x == 0.0
    assert controller.prev_error_x == 0.0
