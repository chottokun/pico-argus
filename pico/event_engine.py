import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any

logger = logging.getLogger(__name__)

class PerceptionEventType(Enum):
    NEW_OBJECT = "NEW_OBJECT"
    ZONE_ENTRY = "ZONE_ENTRY"
    TARGET_LOST = "TARGET_LOST"
    CLASS_MATCH = "CLASS_MATCH"

@dataclass
class PerceptionEvent:
    event_type: PerceptionEventType
    track_id: int
    class_name: str
    confidence: float
    bbox: List[int]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 2),
            "bbox": self.bbox,
            "timestamp": time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        }

class PerceptionEventEngine:
    """知覚フレーム毎のデルタ判定・デバウンス・クールダウン・フィルタリングを備えた能動的イベントエンジン"""

    def __init__(self, min_stable_frames: int = 3, cooldown_sec: float = 5.0):
        self.min_stable_frames = min_stable_frames
        self.cooldown_sec = cooldown_sec
        self.allowed_classes: Optional[Set[str]] = None
        self.enabled_event_types: Set[PerceptionEventType] = set(PerceptionEventType)

        # 内部ステート管理
        self._frame_counters: Dict[int, int] = {}  # track_id -> 連続検知数
        self._last_event_time: Dict[str, float] = {}  # event_key -> 最終発火タイムスタンプ
        self._active_track_ids: Set[int] = set()
        self._recent_events: List[PerceptionEvent] = []
        self._max_history: int = 50

    def set_allowed_classes(self, classes: Optional[List[str]]) -> None:
        """監視対象とするクラス名を指定（Noneの場合は全クラスが対象）"""
        if classes is None:
            self.allowed_classes = None
        else:
            self.allowed_classes = set(classes)

    def set_event_enabled(self, event_type: PerceptionEventType, enabled: bool) -> None:
        """特定のイベントタイプの有効・無効を切替"""
        if enabled:
            self.enabled_event_types.add(event_type)
        else:
            self.enabled_event_types.discard(event_type)

    def process_frame(self, tracked_objects: List[Any]) -> List[PerceptionEvent]:
        """フレームごとの追跡対象オブジェクトを受け取り、抑制・制御されたイベントリストを返却"""
        current_time = time.time()
        emitted_events: List[PerceptionEvent] = []
        current_ids: Set[int] = set()

        for obj in tracked_objects:
            track_id = getattr(obj, "track_id", None)
            if track_id is None:
                continue

            current_ids.add(track_id)
            class_name = getattr(obj, "class_name", getattr(obj, "class", "unknown"))
            confidence = getattr(obj, "confidence", 0.0)
            bbox = list(getattr(obj, "bbox", [0, 0, 0, 0]))

            # クラスフィルターのチェック
            if self.allowed_classes is not None and class_name not in self.allowed_classes:
                continue

            # デバウンス: 連続安定フレーム数のカウントアップ
            count = self._frame_counters.get(track_id, 0) + 1
            self._frame_counters[track_id] = count

            # NEW_OBJECT イベントの評価
            if PerceptionEventType.NEW_OBJECT in self.enabled_event_types:
                if count == self.min_stable_frames:
                    event_key = f"{PerceptionEventType.NEW_OBJECT.value}_{track_id}"
                    last_time = self._last_event_time.get(event_key, 0.0)

                    # クールダウンチェック
                    if (current_time - last_time) >= self.cooldown_sec:
                        event = PerceptionEvent(
                            event_type=PerceptionEventType.NEW_OBJECT,
                            track_id=track_id,
                            class_name=class_name,
                            confidence=confidence,
                            bbox=bbox,
                            timestamp=current_time
                        )
                        emitted_events.append(event)
                        self._last_event_time[event_key] = current_time
                        self._record_recent_event(event)

        # 消失（消失カウンターのクリーンアップ）
        vanished_ids = self._active_track_ids - current_ids
        for tid in vanished_ids:
            self._frame_counters.pop(tid, None)

        self._active_track_ids = current_ids
        return emitted_events

    def _record_recent_event(self, event: PerceptionEvent) -> None:
        self._recent_events.append(event)
        if len(self._recent_events) > self._max_history:
            self._recent_events.pop(0)

    def get_status_summary(self) -> Dict[str, Any]:
        """現在のエンジン設定と最近の発火履歴サマリーを取得"""
        return {
            "min_stable_frames": self.min_stable_frames,
            "cooldown_sec": self.cooldown_sec,
            "allowed_classes": list(self.allowed_classes) if self.allowed_classes else None,
            "enabled_events": [e.value for e in self.enabled_event_types],
            "active_track_count": len(self._active_track_ids),
            "recent_events": [e.to_dict() for e in reversed(self._recent_events[-10:])]
        }
