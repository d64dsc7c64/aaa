from pathlib import Path
import sys

file_path = Path(__file__).resolve()

root_path = file_path.parent

if root_path not in sys.path:
    sys.path.append(str(root_path))

ROOT = root_path.relative_to(Path.cwd())

SOURCES_LIST = ["Image", "Video", "Webcam"]

DETECTION_MODEL_DIR = ROOT / 'weights' / 'detection'
SEGMENT_MODEL_DIR = ROOT / 'weights' / 'segment'
POSE_MODEL_DIR = ROOT / 'weights' / 'pose'

YOLOv8n = DETECTION_MODEL_DIR / "番茄.pt"
YOLOv8s = DETECTION_MODEL_DIR / "草莓.pt"
YOLOv8m = DETECTION_MODEL_DIR / "苹果.pt"
YOLOv8l = DETECTION_MODEL_DIR / "香蕉.pt"
YOLOv8x = DETECTION_MODEL_DIR / "橘子.pt"

DETECTION_MODEL_LIST = [
    "番茄.pt",
    "草莓.pt",
    "苹果.pt",
    "香蕉.pt",
    "橘子.pt"]

SEGMENT_MODEL_LIST = [
    "yolov8n-seg.pt",
    "yolov8s-seg.pt",
    "yolov8m-seg.pt",
    "yolov8l-seg.pt",
    "yolov8x-seg.pt"]

POSE_MODEL_LIST = [
    "yolov8n-pose.pt",
    "yolov8s-pose.pt",
    "yolov8m-pose.pt",
    "yolov8l-pose.pt",
    "yolov8x-pose.pt"]
