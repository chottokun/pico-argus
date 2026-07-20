import os
import cv2
import numpy as np
import onnxruntime as ort
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class Detection:
    """物体検出結果を格納するデータクラス。"""
    def __init__(self, class_id: int, confidence: float, bbox: Tuple[int, int, int, int]) -> None:
        self.class_id: int = class_id
        self.confidence: float = confidence
        self.bbox: Tuple[int, int, int, int] = bbox  # [x, y, w, h] (ピクセル座標)

class YoloDetector:
    """ONNX Runtime を用いて YOLOv8 モデルによる推論を行う高速推論クラス。"""

    def __init__(self, model_path: str, conf_threshold: float = 0.45, iou_threshold: float = 0.4) -> None:
        self.model_path: str = model_path
        self.conf_threshold: float = conf_threshold
        self.iou_threshold: float = iou_threshold

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO ONNX model not found: {model_path}")

        # 最適化プロバイダの設定 (CPUを実行優先とする)
        self.providers: List[str] = ['CPUExecutionProvider']
        
        self.session_options = ort.SessionOptions()
        self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # ONNX Runtime セッションの開始
        self.session = ort.InferenceSession(model_path, self.session_options, providers=self.providers)
        
        # 入出力レイヤー名の取得
        self.input_name: str = self.session.get_inputs()[0].name
        self.output_name: str = self.session.get_outputs()[0].name
        
        # モデルの想定入力サイズを取得 (YOLOv8は通常 640x640)
        self.input_shape: Tuple[int, ...] = self.session.get_inputs()[0].shape
        self.input_width: int = self.input_shape[3]
        self.input_height: int = self.input_shape[2]

        logger.info(
            f"YoloDetector initialized with {model_path}. "
            f"Input size: {self.input_width}x{self.input_height}."
        )

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """画像の前処理（640x640へのリサイズと正規化、およびアスペクト比情報の保持）。"""
        h, w = frame.shape[:2]
        
        # 縦横比を維持したリサイズとパディング (Letterbox)
        # 簡略化のため、ここでは OpenCV の DNN 同様に単純なリサイズを使用し、係数を計算する
        # （精度向上のために必要ならレターボックスに置き換え可能）
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        
        # HWC -> CHW, 0-1正規化, バッチ次元追加
        blob = resized.transpose(2, 0, 1)  # (3, 640, 640)
        blob = blob.astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)  # (1, 3, 640, 640)
        
        x_factor = w / self.input_width
        y_factor = h / self.input_height
        
        return blob, x_factor, y_factor

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """フレーム画像から物体検知を実行し、結果のリストを返す。"""
        # 前処理
        blob, x_factor, y_factor = self._preprocess(frame)
        
        # 推論実行
        outputs = self.session.run([self.output_name], {self.input_name: blob})
        output = outputs[0]  # (1, 84, 8400)
        
        # (84, 8400) に変形して転置 -> (8400, 84)
        predictions = output[0].T
        
        boxes: List[List[int]] = []
        confidences: List[float] = []
        class_ids: List[int] = []
        
        # 各予測結果を解析
        # 84 要素のうち、最初の4つは [cx, cy, w, h], 残り80個がクラススコア
        for pred in predictions:
            # クラススコア領域の最大値とそのインデックスを取得
            class_scores = pred[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            
            if confidence > self.conf_threshold:
                cx, cy, w, h = pred[0:4]
                
                # 元画像の解像度へスケーリング
                x = int((cx - w / 2) * x_factor)
                y = int((cy - h / 2) * y_factor)
                box_w = int(w * x_factor)
                box_h = int(h * y_factor)
                
                boxes.append([x, y, box_w, box_h])
                confidences.append(confidence)
                class_ids.append(class_id)
        
        # OpenCVのNMSBoxesによる非最大値抑制の適用
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.iou_threshold)
        
        detections: List[Detection] = []
        if len(indices) > 0:
            for idx in np.array(indices).flatten():
                detections.append(
                    Detection(
                        class_id=class_ids[idx],
                        confidence=confidences[idx],
                        bbox=tuple(boxes[idx])  # type: ignore
                    )
                )
                
        return detections
