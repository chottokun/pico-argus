import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# COCOクラス名の定義（インデックスに対応）
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

class PerceptionBuffer:
    """YOLO/Tracker の検出結果をテキストメタデータとして蓄積・提供する知覚バッファクラス。

    V2 設計 §1 の「常時稼働テキスト知覚バッファ」の実体。
    """

    def __init__(self, warning_zone_x: Tuple[float, float] = (0.1, 0.9), warning_zone_y: Tuple[float, float] = (0.4, 0.9)) -> None:
        self.active_tracks: List[Dict[str, Any]] = []
        self.warning_zone_x = warning_zone_x
        self.warning_zone_y = warning_zone_y

    def update(self, tracked_objects: List[Any], frame_shape: Tuple[int, int, int]) -> None:
        """トラッカーのアクティブトラック一覧を受け取り、内部のメタデータを更新する。

        Args:
            tracked_objects: TrackedObject のリスト
            frame_shape: 画像フレームのシェイプ (height, width, channels)
        """
        height, width = frame_shape[0], frame_shape[1]
        new_tracks = []

        for obj in tracked_objects:
            # bbox: [x, y, w, h] (ピクセル座標)
            x, y, w, h = obj.bbox
            
            # 正規化中心座標
            cx = (x + w / 2) / width
            cy = (y + h / 2) / height
            
            # 正規化面積
            area = (w * h) / (width * height)

            # クラス名取得
            class_name = (
                COCO_CLASSES[obj.class_id]
                if obj.class_id < len(COCO_CLASSES)
                else f"unknown_{obj.class_id}"
            )

            # 画面内の大まかな相対位置表現
            h_pos = "left" if cx < 0.33 else ("right" if cx > 0.66 else "center")
            v_pos = "top" if cy < 0.33 else ("bottom" if cy > 0.66 else "middle")
            position_label = f"{v_pos}-{h_pos}"

            # 警戒ゾーン判定
            is_in_warning = (
                self.warning_zone_x[0] <= cx <= self.warning_zone_x[1] and
                self.warning_zone_y[0] <= cy <= self.warning_zone_y[1]
            )

            new_tracks.append({
                "track_id": obj.track_id,
                "class_id": obj.class_id,
                "class_name": class_name,
                "confidence": float(obj.confidence),
                "bbox": [int(x), int(y), int(w), int(h)],
                "normalized_center": [round(cx, 3), round(cy, 3)],
                "normalized_area": round(area, 4),
                "position_label": position_label,
                "warning_zone_triggered": is_in_warning
            })

        self.active_tracks = new_tracks
        logger.debug(f"PerceptionBuffer updated: {len(self.active_tracks)} tracks active.")

    def get_active_tracks_json(self) -> List[Dict[str, Any]]:
        """AgentState に格納するための構造化データを取得する。"""
        return self.active_tracks

    def get_active_tracks_text(self) -> str:
        """LLM のコンテキストに注入するための人間可読なテキスト表現を生成する。"""
        if not self.active_tracks:
            return "No active tracks detected in the current frame."

        lines = ["Active tracks detected in the scene:"]
        for t in self.active_tracks:
            warning_tag = " [⚠️WARNING ZONE DETECTED]" if t["warning_zone_triggered"] else ""
            desc = (
                f"- ID: {t['track_id']} | "
                f"Class: {t['class_name']} (conf: {t['confidence']:.2f}) | "
                f"Position: {t['position_label']} (cx: {t['normalized_center'][0]}, cy: {t['normalized_center'][1]}){warning_tag} | "
                f"Area: {t['normalized_area']:.4%}"
            )
            lines.append(desc)
        return "\n".join(lines)
