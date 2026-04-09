import argparse
from ultralytics import YOLO
from ultralytics.nn.modules.head import Segment

def custom_segment_forward(self, x):
    p = self.proto(x[0])  
    res = []
    
    for i in range(self.nl):
        res.append(self.cv2[i](x[i]))  
        res.append(self.cv3[i](x[i]))  
        res.append(self.cv4[i](x[i]))  
    
    res.append(p)
    return tuple(res)

def main(weights, opset, sim):

    Segment.forward = custom_segment_forward
    
    model = YOLO(weights)
    
    model.export(format="onnx", opset=opset, simplify=sim)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export YOLO11-Seg to RKNN compatible ONNX")
    parser.add_argument('-w', '--weights', type=str, required=True, help='PyTorch model weights (.pt)')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset version (RKNN usually prefers 12)')
    parser.add_argument('--sim', action='store_true', default=False, help='Simplify ONNX model')
    args = parser.parse_args()
    
    main(args.weights, args.opset, args.sim)