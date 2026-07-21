import pytest
from pico.pid_controller import AdaptivePIDController

def test_pid_dead_zone() -> None:
    # 不感帯 0.05, 制限オフ
    controller = AdaptivePIDController(
        kp_base=0.5, ki=0.05, kd=0.01,
        dead_zone=0.05, min_speed=0.03, max_step=0.12,
        max_acceleration=0.0
    )
    # 偏差が不感帯(0.05)未満の場合、出力は 0.0 になること
    dx, dy = controller.calculate_step(target_cx=0.52, target_cy=0.49, dt=0.2)
    assert dx == 0.0
    assert dy == 0.0

def test_pid_nonlinear_gain_and_output_limit() -> None:
    # 制限オフ
    controller = AdaptivePIDController(
        kp_base=0.5, ki=0.0, kd=0.0,
        dead_zone=0.01, min_speed=0.0, max_step=0.10,
        max_acceleration=0.0
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
    # 制限オフ
    controller = AdaptivePIDController(
        kp_base=0.0, ki=0.5, kd=0.0,
        dead_zone=0.01, min_speed=0.0, max_step=0.15,
        integral_limit=0.10, max_acceleration=0.0
    )
    
    # 偏差を同じ方向に与え続け、積分値を蓄積させる
    # dt=0.2, error=0.1 のとき, 累積積分値 = 0.1 * 0.2 = 0.02
    # 何回も繰り返して制限(0.10)に達するかテスト
    for _ in range(10):
        dx, _ = controller.calculate_step(target_cx=0.6, target_cy=0.5, dt=0.2)
    
    # ki * integral_x = 0.5 * 0.10 = 0.05 (制限によりこれ以上増えない)
    assert pytest.approx(dx, 0.01) == 0.05

def test_pid_reset() -> None:
    # 制限オフ
    controller = AdaptivePIDController(
        kp_base=0.0, ki=0.5, kd=0.0,
        dead_zone=0.01, min_speed=0.0, max_step=0.15,
        max_acceleration=0.0
    )
    
    # 積分値を蓄積
    controller.calculate_step(target_cx=0.6, target_cy=0.5, dt=0.2)
    assert controller.integral_x > 0.0
    
    # リセット
    controller.reset()
    assert controller.integral_x == 0.0
    assert controller.prev_error_x == 0.0

def test_pid_slew_rate_limit() -> None:
    # 大きなゲインをセットし、max_acceleration = 0.03 に設定
    controller = AdaptivePIDController(
        kp_base=10.0, ki=0.0, kd=0.0,
        dead_zone=0.001, min_speed=0.0, max_step=0.15,
        max_acceleration=0.03
    )

    # 1ステップ目: スルーレート制限により、0.0 から 0.03 に段階的に加速するはず
    dx1, dy1 = controller.calculate_step(target_cx=1.0, target_cy=0.5, dt=0.2)
    assert pytest.approx(dx1, 0.001) == 0.03
    assert dy1 == 0.0

    # 2ステップ目: 前回の 0.03 からさらに 0.03 加速して 0.06 になるはず
    dx2, dy2 = controller.calculate_step(target_cx=1.0, target_cy=0.5, dt=0.2)
    assert pytest.approx(dx2, 0.001) == 0.06
    assert dy2 == 0.0

    # リセット: 前回の出力履歴もリセットされること
    controller.reset()
    assert controller.prev_output_x == 0.0
    assert controller.prev_output_y == 0.0

    # 3ステップ目: リセット後のため、再度 0.03 から加速がスタートすること
    dx3, dy3 = controller.calculate_step(target_cx=1.0, target_cy=0.5, dt=0.2)
    assert pytest.approx(dx3, 0.001) == 0.03
