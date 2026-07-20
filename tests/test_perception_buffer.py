from pico.tracker import TrackedObject
from pico.perception_buffer import PerceptionBuffer

def test_perception_buffer_update() -> None:
    # ダミーフレームの形状 (height, width, channels) = (480, 640, 3)
    frame_shape = (480, 640, 3)
    
    # 追跡対象オブジェクトのモック
    # bbox: [x, y, w, h]
    obj1 = TrackedObject(track_id=1, class_id=0, bbox=(10, 10, 100, 200), confidence=0.95) # 左上寄り
    obj2 = TrackedObject(track_id=2, class_id=15, bbox=(400, 300, 150, 100), confidence=0.85) # 右下寄り
    
    tracked_objects = [obj1, obj2]
    
    buffer = PerceptionBuffer()
    buffer.update(tracked_objects, frame_shape)
    
    # JSON構造の確認
    json_data = buffer.get_active_tracks_json()
    assert len(json_data) == 2
    
    # オブジェクト1のメタデータ確認
    t1 = json_data[0]
    assert t1["track_id"] == 1
    assert t1["class_name"] == "person"
    assert t1["confidence"] == 0.95
    assert t1["position_label"] == "top-left"
    # 中心座標 cx = (10 + 50)/640 = 60/640 = 0.09375 => 0.094
    assert t1["normalized_center"][0] == 0.094
    # cy = (10 + 100)/480 = 110/480 = 0.22916 => 0.229
    assert t1["normalized_center"][1] == 0.229
    assert t1["warning_zone_triggered"] is False
    
    # オブジェクト2のメタデータ確認
    t2 = json_data[1]
    assert t2["track_id"] == 2
    assert t2["class_name"] == "cat"
    assert t2["confidence"] == 0.85
    assert t2["position_label"] == "bottom-right"
    assert t2["warning_zone_triggered"] is True
    
    # テキスト出力の確認
    text_data = buffer.get_active_tracks_text()
    assert "Active tracks detected in the scene:" in text_data
    assert "ID: 1 | Class: person" in text_data
    assert "ID: 2 | Class: cat" in text_data
    assert "WARNING ZONE DETECTED" in text_data
    assert "Position: top-left" in text_data
    assert "Position: bottom-right" in text_data


def test_perception_buffer_empty() -> None:
    buffer = PerceptionBuffer()
    buffer.update([], (480, 640, 3))
    
    assert len(buffer.get_active_tracks_json()) == 0
    assert buffer.get_active_tracks_text() == "No active tracks detected in the current frame."
