#@title #yolo11 preprocess + postprocess (execute this)
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
from typing import Tuple, Union

from ultralytics.engine.results import Results
from ultralytics.utils import ops

import cv2
import numpy as np
import yaml

from ai_edge_litert.interpreter import Interpreter

class YOLOv8TFLite:
    """
    A class for performing object detection using the YOLOv8 model with TensorFlow Lite.

    This class handles model loading, preprocessing, inference, and visualization of detection results.

    Attributes:
        model (Interpreter): TensorFlow Lite interpreter for the YOLOv8 model.
        conf (float): Confidence threshold for filtering detections.
        iou (float): Intersection over Union threshold for non-maximum suppression.
        classes (Dict[int, str]): Dictionary mapping class IDs to class names.
        color_palette (np.ndarray): Random color palette for visualization with shape (num_classes, 3).
        in_width (int): Input width required by the model.
        in_height (int): Input height required by the model.
        in_index (int): Input tensor index in the model.
        in_scale (float): Input quantization scale factor.
        in_zero_point (int): Input quantization zero point.
        int8 (bool): Whether the model uses int8 quantization.
        out_index (int): Output tensor index in the model.
        out_scale (float): Output quantization scale factor.
        out_zero_point (int): Output quantization zero point.

    Methods:
        letterbox: Resizes and pads image while maintaining aspect ratio.
        draw_detections: Draws bounding boxes and labels on the input image.
        preprocess: Preprocesses the input image before inference.
        postprocess: Processes model outputs to extract and visualize detections.
        detect: Performs object detection on an input image.
    """

    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.45, metadata: Union[str, None] = None):
        """
        Initialize an instance of the YOLOv8TFLite class.

        Args:
            model_path (str): Path to the TFLite model file.
            conf (float): Confidence threshold for filtering detections.
            iou (float): IoU threshold for non-maximum suppression.
            metadata (str | None): Path to the metadata file containing class names.
        """
        self.conf = conf
        self.iou = iou
        if metadata is None:
            self.classes = {i: i for i in range(1000)}
        else:
            with open(metadata) as f:
                self.classes = yaml.safe_load(f)["names"]
        np.random.seed(42)  # Set seed for reproducible colors
        self.color_palette = np.random.uniform(128, 255, size=(len(self.classes), 3))

        # Initialize the TFLite interpreter
        self.model = Interpreter(model_path=model_path)
        self.model.allocate_tensors()

        # Get input details
        input_details = self.model.get_input_details()[0]
        print("input_details\n", input_details, "\n")
        self.in_width, self.in_height = input_details["shape"][1:3]
        self.in_index = input_details["index"]
        self.in_scale, self.in_zero_point = input_details["quantization"]
        self.int8 = input_details["dtype"] == np.int8

        if self.int8:
            # Get output details
            output_details_proto = self.model.get_output_details()[0]
            print("output_details_proto\n", output_details_proto, "\n")
            self.out_index_proto = output_details_proto["index"]
            self.out_scale_proto, self.out_zero_point_proto= output_details_proto["quantization"]

            # Get output details
            output_details = self.model.get_output_details()[1]
            print("output_details_detection\n", output_details, "\n")
            self.out_index = output_details["index"]
            self.out_scale, self.out_zero_point = output_details["quantization"]
        else:
            # Get output details
            output_details_proto = self.model.get_output_details()[1]
            print("output_details_proto\n", output_details_proto, "\n")
            self.out_index_proto = output_details_proto["index"]
            self.out_scale_proto, self.out_zero_point_proto= output_details_proto["quantization"]

            # Get output details
            output_details = self.model.get_output_details()[0]
            print("output_details_detection\n", output_details, "\n")
            self.out_index = output_details["index"]
            self.out_scale, self.out_zero_point = output_details["quantization"]

    def letterbox(
        self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Resize and pad image while maintaining aspect ratio.

        Args:
            img (np.ndarray): Input image with shape (H, W, C).
            new_shape (Tuple[int, int]): Target shape (height, width).

        Returns:
            (np.ndarray): Resized and padded image.
            (Tuple[float, float]): Padding ratios (top/height, left/width) for coordinate adjustment.
        """
        shape = img.shape[:2]  # Current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # Resize if needed
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img, (top / img.shape[0], left / img.shape[1])

    def draw_detections(self, img: np.ndarray, box: np.ndarray, score: np.float32, class_id: int) -> None:
        """
        Draw bounding boxes and labels on the input image based on the detected objects.

        Args:
            img (np.ndarray): The input image to draw detections on.
            box (np.ndarray): Detected bounding box in the format [x1, y1, width, height].
            score (np.float32): Confidence score of the detection.
            class_id (int): Class ID for the detected object.
        """
        x1, y1, w, h = box
        color = self.color_palette[class_id]

        # Draw bounding box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create label with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Get text size for background rectangle
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Position label above or below box depending on space
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw label background
        cv2.rectangle(
            img,
            (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y + label_height)),
            color,
            cv2.FILLED,
        )

        # Draw text
        cv2.putText(img, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Preprocess the input image before performing inference.

        Args:
            img (np.ndarray): The input image to be preprocessed with shape (H, W, C).

        Returns:
            (np.ndarray): Preprocessed image ready for model input.
            (Tuple[float, float]): Padding ratios for coordinate adjustment.
        """
        img, pad = self.letterbox(img, (self.in_width, self.in_height))
        img = img[..., ::-1][None]  # BGR to RGB and add batch dimension (N, H, W, C) for TFLite
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        return img / 255, pad  # Normalize to [0, 1]

    def bounding_box(self, img: np.ndarray, outputs: np.ndarray, pad: Tuple[float, float]) -> np.ndarray:
        """
        Process model outputs to extract and visualize detections.

        Args:
            img (np.ndarray): The original input image.
            outputs (np.ndarray): Raw model outputs.
            pad (Tuple[float, float]): Padding ratios from preprocessing.

        Returns:
            (np.ndarray): The input image with detections drawn on it.
        """
        # Adjust coordinates based on padding and scale to original image size
        # print("outputs_shape: ", outputs.shape, "\n")
        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]
        outputs[:, :4] *= max(img.shape)

        # Transform outputs to [x, y, w, h] format
        outputs = outputs.transpose(0, 2, 1)
        outputs[..., 0] -= outputs[..., 2] / 2  # x center to top-left x
        outputs[..., 1] -= outputs[..., 3] / 2  # y center to top-left y

        for out in outputs:
            # Get scores and apply confidence threshold
            scores = out[:, 4:].max(-1)
            keep = scores > self.conf
            boxes = out[keep, :4]
            scores = scores[keep]
            class_ids = out[keep, 4:].argmax(-1)

            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.iou).flatten()

            # Draw detections that survived NMS
            [self.draw_detections(img, boxes[i], scores[i], class_ids[i]) for i in indices]

        return img

    def postprocess(self, img, prep_img, outs):
        """
        Post-process model predictions to extract meaningful results.

        Args:
            img (np.ndarray): The original input image.
            prep_img (np.ndarray): The preprocessed image used for inference.
            outs (list): Model outputs containing predictions and prototype masks.

        Returns:
            (List[Results]): Processed detection results containing bounding boxes and segmentation masks.
        """
        preds, protos = [torch.from_numpy(p) for p in outs]

        # print("preds", type(preds), preds.shape)
        # print("protos", type(protos), protos.shape)

        preds_filtered, keep_index_batch = ops.non_max_suppression(
            preds, self.conf, self.iou,
            nc=len(self.classes),
            return_idxs = True
            )

        results = []
        for i, pred in enumerate(preds_filtered):
            # print("pred", type(pred), pred.shape)

            keep_index = keep_index_batch[i]
            # print("keep_index", type(keep_index), keep_index.shape)

            boxes = pred[:, :4]
            boxes[..., [0, 2]] *= prep_img.shape[2]
            boxes[..., [1, 3]] *= prep_img.shape[1]
            box_conf = pred[:, 4]
            class_score = pred[:, 5]
            mask_coefs = pred[:, 6:]

            # print("boxes_scaled", type(boxes), boxes.shape)
            # print("class_score", type(class_score), class_score.shape)
            # print("mask_coefs", type(mask_coefs), mask_coefs.shape)

            # print("boxes", pred[:, :4])
            # print(img.shape)
            # print(prep_img.shape[1:3])
            pred[:, :4] = ops.scale_boxes(prep_img.shape[1:3], pred[:, :4], img.shape)
            # print("boxes", pred[:, :4])

            # masks = self.process_mask(protos[i].permute(2, 0, 1), mask_coefs, pred[:, :4], img.shape[:2])
            masks = self.process_mask(protos[i].permute(2, 0, 1), pred[:, 6:], pred[:, :4], img.shape[:2])
            results.append(Results(img, path="", names=self.classes, boxes=pred[:, :6], masks=masks))

            # results.append((boxes_scaled, box_conf, class_score, masks))

        return results

    def process_mask(self, protos, masks_in, bboxes, shape):
        """
        Process prototype masks with predicted mask coefficients to generate instance segmentation masks.

        Args:
            protos (torch.Tensor): Prototype masks with shape (mask_dim, mask_h, mask_w).
            masks_in (torch.Tensor): Predicted mask coefficients with shape (n, mask_dim), where n is number of detections.
            bboxes (torch.Tensor): Bounding boxes with shape (n, 4), where n is number of detections.
            shape (Tuple[int, int]): The size of the input image as (height, width).

        Returns:
            (torch.Tensor): Binary segmentation masks with shape (n, height, width).
        """
        c, mh, mw = protos.shape  # CHW
        masks = (masks_in @ protos.view(c, -1)).view(-1, mh, mw)  # Matrix multiplication
        masks = ops.scale_masks(masks[None], shape)[0]  # Scale masks to original image size
        masks = ops.crop_mask(masks, bboxes)  # Crop masks to bounding boxes
        return masks.gt_(0.0)  # Convert to binary masks

    def detect(self, img_path: str) -> np.ndarray:
        """
        Perform object detection on an input image.

        Args:
            img_path (str): Path to the input image file.

        Returns:
            (np.ndarray): The output image with drawn detections.
        """
        # Load and preprocess image
        img = cv2.imread(img_path)
        x, pad = self.preprocess(img)

        # Apply quantization if model is int8
        if self.int8:
            x = (x / self.in_scale + self.in_zero_point).astype(np.int8)

        # print("preprocessed_image: ", x.shape, "\n")

        # Set input tensor and run inference
        self.model.set_tensor(self.in_index, x)
        self.model.invoke()

        # Get output and dequantize if necessary
        head_predict  = self.model.get_tensor(self.out_index)
        proto_predict = self.model.get_tensor(self.out_index_proto)

        if self.int8:
            head_predict = (head_predict.astype("float32") - self.out_zero_point) * self.out_scale
            proto_predict = (proto_predict.astype("float32") - self.out_zero_point_proto) * self.out_scale_proto

        # print("head_predict_shape: ", type(head_predict), head_predict.shape) # (1, 116, 8400)
        # 8400 = anchors
        # 116  = 4 box_coord + 80 class_score + 32 prototype_mask
        # print("proto_predict_shape: ", type(proto_predict), proto_predict.shape)

        # Process detections and return result
        # return self.bounding_box(img, box_predict, pad)
        # print()
        return self.postprocess(img, x, [head_predict, proto_predict])