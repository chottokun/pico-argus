from pico.tracker import SimpleIoUTracker
from pico.detector import Detection

def test_iou_tracker_assign_id() -> None:
    tracker = SimpleIoUTracker(iou_threshold=0.3, max_lost_frames=2)
    
    # フレーム1: 1つ検出
    det_f1 = [
        Detection(class_id=0, confidence=0.8, bbox=(100, 100, 50, 50))
    ]
    tracks_f1 = tracker.update(det_f1)
    
    assert len(tracks_f1) == 1
    assert tracks_f1[0].track_id == 1
    assert tracks_f1[0].bbox == (100, 100, 50, 50)

def test_iou_tracker_match_and_lost() -> None:
    tracker = SimpleIoUTracker(iou_threshold=0.3, max_lost_frames=2)
    
    # フレーム1: 検出
    tracker.update([Detection(class_id=0, confidence=0.8, bbox=(100, 100, 50, 50))])
    
    # フレーム2: わずかに動いたターゲット（同一IDを期待）
    tracks_f2 = tracker.update([Detection(class_id=0, confidence=0.85, bbox=(105, 105, 50, 50))])
    assert len(tracks_f2) == 1
    assert tracks_f2[0].track_id == 1
    assert tracks_f2[0].bbox == (105, 105, 50, 50)
    
    # フレーム3: 検出なし (一時ロスト)
    tracks_f3 = tracker.update([])
    assert len(tracks_f3) == 0  # 画面には出力されないが、内部では保持されているはず
    
    # フレーム4: 再び同じ位置に検出 (同じIDを期待)
    tracks_f4 = tracker.update([Detection(class_id=0, confidence=0.75, bbox=(108, 108, 50, 50))])
    assert len(tracks_f4) == 1
    assert tracks_f4[0].track_id == 1

def test_iou_tracker_new_id_for_different_position() -> None:
    tracker = SimpleIoUTracker(iou_threshold=0.3, max_lost_frames=2)
    
    # ターゲット1
    tracker.update([Detection(class_id=0, confidence=0.8, bbox=(100, 100, 50, 50))])
    
    # 遠く離れたターゲット2 (新しいIDを期待)
    tracks = tracker.update([Detection(class_id=0, confidence=0.8, bbox=(400, 400, 50, 50))])
    assert len(tracks) == 1
    assert tracks[0].track_id == 2
