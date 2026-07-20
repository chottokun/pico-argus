import pytest
import numpy as np
import os
from pico.detector import YoloDetector, Detection

def test_yolo_detector_initialization() -> None:
    # モデルファイルが存在するか確認
    model_path = "yolov8s.onnx"
    if not os.path.exists(model_path):
        pytest.skip("yolov8s.onnx is missing, skipping detector initialization test")

    detector = YoloDetector(model_path=model_path, conf_threshold=0.45)
    assert detector.model_path == model_path
    assert detector.conf_threshold == 0.45

def test_yolo_detector_inference() -> None:
    model_path = "yolov8s.onnx"
    if not os.path.exists(model_path):
        pytest.skip("yolov8s.onnx is missing, skipping inference test")

    detector = YoloDetector(model_path=model_path, conf_threshold=0.45)
    
    # ダミー画像の作成（黒い画像）
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 推論実行（黒画像なので検出数は通常 0 だが、空のリストが正しく返るか検証）
    detections = detector.detect(dummy_image)
    assert isinstance(detections, list)
    for det in detections:
        assert isinstance(det, Detection)
        assert isinstance(det.class_id, int)
        assert isinstance(det.confidence, float)
        assert len(det.bbox) == 4  # [x, y, w, h]
