import logging
from typing import List, Tuple, Dict
from pico.detector import Detection

logger = logging.getLogger(__name__)

class TrackedObject:
    """現在追跡中のオブジェクト状態を保持するクラス。"""
    def __init__(self, track_id: int, class_id: int, bbox: Tuple[int, int, int, int], confidence: float) -> None:
        self.track_id: int = track_id
        self.class_id: int = class_id
        self.bbox: Tuple[int, int, int, int] = bbox
        self.confidence: float = confidence
        self.lost_frames: int = 0

class SimpleIoUTracker:
    """バウンディングボックスの交差比 (IoU) に基づいて、フレーム間で同一物体を追跡する簡易トラッカー。"""

    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 3) -> None:
        self.iou_threshold: float = iou_threshold
        self.max_lost_frames: int = max_lost_frames
        self.next_id: int = 1
        
        # 追跡中のオブジェクト管理 (key: track_id)
        self.tracked_objects: Dict[int, TrackedObject] = {}

    @staticmethod
    def _calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """2つのバウンディングボックス [x, y, w, h] の IoU を算出する。"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # 矩形の右下座標を算出
        x1_end, y1_end = x1 + w1, y1 + h1
        x2_end, y2_end = x2 + w2, y2 + h2

        # 重なり部分の座標を算出
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1_end, x2_end)
        inter_y2 = min(y1_end, y2_end)

        # 重なり領域の面積
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        intersection_area = inter_w * inter_h

        if intersection_area == 0:
            return 0.0

        # それぞれの矩形面積とUnion面積
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """新規の検出リストを受け取り、追跡状態を更新して、現在アクティブな（ロストしていない）オブジェクトのリストを返す。"""
        
        # 1. 追跡中オブジェクトのロストカウントをインクリメント
        for obj in self.tracked_objects.values():
            obj.lost_frames += 1

        matched_detections: Dict[int, Detection] = {}  # key: track_id -> matched detection

        # 2. 既存のオブジェクトと新規検出のマッチング (IoUの高い組み合わせを優先)
        # 簡易的に、各検出に対して最も IoU が高い既存オブジェクトを貪欲法でマッチングする
        for det in detections:
            best_iou = 0.0
            best_track_id = -1

            for track_id, obj in self.tracked_objects.items():
                # クラスIDが一致するものだけを追跡対比にする
                if obj.class_id != det.class_id:
                    continue
                
                iou = self._calculate_iou(obj.bbox, det.bbox)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id != -1 and best_track_id not in matched_detections:
                # マッチした既存オブジェクトの状態を更新
                obj = self.tracked_objects[best_track_id]
                obj.bbox = det.bbox
                obj.confidence = det.confidence
                obj.lost_frames = 0  # 検知されたためロストカウンターリセット
                matched_detections[best_track_id] = det
            else:
                # マッチしなかった検出は新規追跡対象とする
                new_obj = TrackedObject(
                    track_id=self.next_id,
                    class_id=det.class_id,
                    bbox=det.bbox,
                    confidence=det.confidence
                )
                self.tracked_objects[self.next_id] = new_obj
                self.next_id += 1

        # 3. max_lost_frames を超えて失われたオブジェクトを削除
        lost_ids = [
            track_id for track_id, obj in self.tracked_objects.items()
            if obj.lost_frames > self.max_lost_frames
        ]
        for track_id in lost_ids:
            del self.tracked_objects[track_id]
            logger.info(f"Object {track_id} removed from tracker (exceeded max lost frames).")

        # 4. 現在のフレームでアクティブ（このフレームで検出されたもののみ）な追跡オブジェクトを返却
        # （一時的にロストしているものは内部だけでキープし、描画や制御のターゲットからは除外する）
        active_tracks = [
            obj for obj in self.tracked_objects.values()
            if obj.lost_frames == 0
        ]

        return active_tracks
