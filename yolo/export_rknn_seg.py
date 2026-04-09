import argparse
from ultralytics import YOLO
from ultralytics.nn.modules.head import Segment

def custom_segment_forward(self, x):
    """
    專為 RK3588 (RKNN) 定製的分割頭 forward 函數
    徹底剝離 NMS 和複雜張量拼接，直接輸出原始特徵圖
    """
    # p 是多尺度特徵圖經過 proto 模塊生成的 Prototype Mask
    p = self.proto(x[0])  
    res = []
    
    for i in range(self.nl):
        # 分別導出 3 個尺度的 Bbox(邊界框)、Class(類別機率) 和 Mask(掩膜係數)
        res.append(self.cv2[i](x[i]))  
        res.append(self.cv3[i](x[i]))  
        res.append(self.cv4[i](x[i]))  
    
    # 最後附上原型掩膜
    res.append(p)
    return tuple(res)

def main(weights, opset, sim):
    print(f"正在加載模型: {weights}")
    
    # 【核心魔法】：動態替換 (Monkey Patch) 最新版 Ultralytics 的分割頭邏輯
    Segment.forward = custom_segment_forward
    
    # 實例化模型
    model = YOLO(weights)
    
    # 導出為 ONNX 格式
    print("開始導出 ONNX (適配 RKNN 格式)...")
    model.export(format="onnx", opset=opset, simplify=sim)
    print("導出完成！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export YOLO11-Seg to RKNN compatible ONNX")
    parser.add_argument('-w', '--weights', type=str, required=True, help='PyTorch model weights (.pt)')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset version (RKNN usually prefers 12)')
    parser.add_argument('--sim', action='store_true', default=False, help='Simplify ONNX model')
    args = parser.parse_args()
    
    main(args.weights, args.opset, args.sim)