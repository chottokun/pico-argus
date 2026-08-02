---
type: Specification
title: Tapo C210 ONVIF Hardware Polarity Rules
---

# Tapo C210 ONVIF / PTZ 運動極性仕様書 (Hardware Polarity Spec)

本書は Tapo C210 カメラにおける ONVIF コマンド符号、設定フラグ、物理動作の対応関係を記録した絶対仕様書です。
今後の開発において本仕様を変更・混同してはなりません。

## 1. 設定フラグ (`camera_config.json` / `tapo_config.json`)

```json
{
    "INVERT_PAN": true,
    "INVERT_TILT": false
}
```

- **`INVERT_PAN = true`**: 画面右にターゲットが現れた際、カメラを右へ向かせてターゲットを画面中央に収束させるネガティブフィードバック追尾に不可欠。
- **`INVERT_TILT = false`**: 画面上にターゲットが現れた際、カメラを上へ向かせる標準垂直制御に不可欠。

---

## 2. ONVIF Raw RelativeMove コマンド符号と物理回転方向

| ONVIF コマンドパラメータ | 符号 | 物理カメラの回転方向 |
| :--- | :--- | :--- |
| **Pan (水平 `x`)** | `x < 0` (マイナス) | 物理的に **【右 (RIGHT)】** へ回転 |
| **Pan (水平 `x`)** | `x > 0` (プラス) | 物理的に **【左 (LEFT)】** へ回転 |
| **Tilt (垂直 `y`)** | `y < 0` (マイナス) | 物理的に **【下 (DOWN)】** へ回転 |
| **Tilt (垂直 `y`)** | `y > 0` (プラス) | 物理的に **【上 (UP)】** へ回転 |

---

## 3. `PTZController.safe_move` 変換方程式 (`onvif_client.py`)

```python
# 物理カメラに命令を送信するタイミングでの極性適用規則
cmd_x = -actual_move_x if self.invert_pan else actual_move_x
cmd_y = -actual_move_y if self.invert_tilt else actual_move_y
```

- **水平 (`x`)**: `requested_x > 0` (右移動要求) ➔ `cmd_x = -requested_x < 0` ➔ 物理右へ正しく移動。
- **垂直 (`y`)**: `requested_y > 0` (上移動要求) ➔ `cmd_y = requested_y > 0` ➔ 物理上へ正しく移動。

## 3.5. 自動追尾 (PID Loop) 入力座標仕様 (`pid_controller.py`)

- **偏差計算公式**:
  ```python
  error_x = 0.5 - target_cx
  error_y = 0.5 - target_cy
  ```
  `target - 0.5` を使用してはならない（反転入力は画面上のターゲットを逃がす正フィードバック暴走を引き起こす）。

---

## 4. 4方向アライメント（`calibrate_home`）シーケンス

1. **右端突き当て**: ONVIF `x = -0.15` (または `cmd_x = -step_size_x if invert_pan else step_size_x`)
2. **下端突き当て**: ONVIF `y = -0.10`
3. **左端突き当て**: ONVIF `x = +0.15`
4. **上端突き当て**: ONVIF `y = +0.10`
5. **水平中央復帰 (7歩)**: ONVIF `x = -0.15` で左端から右へ移動
6. **垂直重力降下復帰 (11歩)**: ONVIF `y = -0.10` で上端から下へ降下して正面中央でピタ止め
