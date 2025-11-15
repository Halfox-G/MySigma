import sys
import torch
sys.path.insert(0, ".")
from configs.config_nyu import config
from models.builder import EncoderDecoder as segmodel

def main():
    device = torch.device('cpu')
    # 构建模型（传入 criterion=None 以跳过权重初始化中可能的 I/O）
    model = segmodel(cfg=config, criterion=None, norm_layer=torch.nn.BatchNorm2d)
    model.eval()
    model.to(device)

    # 构造假的输入
    rgb = torch.randn(1, 3, 480, 640, device=device)
    depth = torch.randn(1, 3, 480, 640, device=device)

    try:
        out = model(rgb, depth)
        print('SUCCESS: output type:', type(out), 'shape:', out.shape if hasattr(out, 'shape') else 'N/A')
    except Exception as e:
        import traceback
        print('ERROR during forward:')
        traceback.print_exc()

if __name__ == '__main__':
    main()
