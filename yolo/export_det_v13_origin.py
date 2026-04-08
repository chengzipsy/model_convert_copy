from ultralytics import YOLO
import argparse
from onnx import shape_inference

try:
    import onnxsim
except ImportError:
    onnxsim = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w", "--weights", type=str, required=True, help="PyTorch yolo weights"
    )
    args = parser.parse_args()
    return args

def main(args):
  model = YOLO(args.weights)
  model.export(format="onnx")

if __name__ == "__main__":
  main(parse_args())
