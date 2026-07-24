import time
from pico.event_engine import PerceptionEventEngine, PerceptionEventType

class DummyTrackedObject:
    def __init__(self, track_id: int, class_name: str, confidence: float = 0.8, bbox=None):
        self.track_id = track_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox or [100, 100, 200, 200]

def test_debounce_requires_n_frames():
    """Nフレーム連続して安定検知されるまでイベントが発火しないことを検証"""
    engine = PerceptionEventEngine(min_stable_frames=3, cooldown_sec=5.0)
    
    obj1 = DummyTrackedObject(track_id=1, class_name="person", confidence=0.8)
    
    # 1フレーム目: まだ発火しない
    events = engine.process_frame([obj1])
    assert len(events) == 0

    # 2フレーム目: まだ発火しない
    events = engine.process_frame([obj1])
    assert len(events) == 0

    # 3フレーム目: 安定検知に達したため EVENT_NEW_OBJECT が1回だけ発火
    events = engine.process_frame([obj1])
    assert len(events) == 1
    assert events[0].event_type == PerceptionEventType.NEW_OBJECT
    assert events[0].track_id == 1
    assert events[0].class_name == "person"

def test_cooldown_prevents_frequent_events():
    """発火後、クールダウン時間内は同一オブジェクトのイベントが抑制されることを検証"""
    engine = PerceptionEventEngine(min_stable_frames=1, cooldown_sec=2.0)
    
    obj1 = DummyTrackedObject(track_id=1, class_name="person", confidence=0.8)
    
    # 1回目: 即時発火
    events1 = engine.process_frame([obj1])
    assert len(events1) == 1

    # 2回目 (直後): クールダウン中のため発火しない
    events2 = engine.process_frame([obj1])
    assert len(events2) == 0

    # クールダウン時間経過をシミュレート
    time.sleep(2.1)
    
    # フレームから一旦消えて再登場した場合にクールダウン明けで発火
    engine.process_frame([]) # 一時ロスト
    events3 = engine.process_frame([obj1])
    assert len(events3) == 1

def test_class_filter():
    """対象クラスフィルターが機能し、指定外クラスが発火しないことを検証"""
    engine = PerceptionEventEngine(min_stable_frames=1, cooldown_sec=1.0)
    engine.set_allowed_classes(["person"])
    
    obj_chair = DummyTrackedObject(track_id=1, class_name="chair", confidence=0.8)
    obj_person = DummyTrackedObject(track_id=2, class_name="person", confidence=0.8)

    events = engine.process_frame([obj_chair, obj_person])
    assert len(events) == 1
    assert events[0].track_id == 2
    assert events[0].class_name == "person"

def test_event_toggle():
    """特定のイベントタイプを無効化できることを検証"""
    engine = PerceptionEventEngine(min_stable_frames=1, cooldown_sec=1.0)
    engine.set_event_enabled(PerceptionEventType.NEW_OBJECT, False)

    obj1 = DummyTrackedObject(track_id=1, class_name="person", confidence=0.8)
    events = engine.process_frame([obj1])
    assert len(events) == 0  # NEW_OBJECT が無効なので発火しない

def test_get_status_summary():
    """設定や直近イベント履歴のサマリー照会が取得できることを検証"""
    engine = PerceptionEventEngine(min_stable_frames=1, cooldown_sec=5.0)
    obj1 = DummyTrackedObject(track_id=1, class_name="person", confidence=0.8)
    engine.process_frame([obj1])

    summary = engine.get_status_summary()
    assert summary["cooldown_sec"] == 5.0
    assert summary["min_stable_frames"] == 1
    assert len(summary["recent_events"]) == 1
    assert summary["recent_events"][0]["track_id"] == 1
