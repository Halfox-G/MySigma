import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# --- 核心辅助函数导入（来自自定义文件）---
# 必须确保 utils/, engine/ 目录下存在这些文件
from utils.init_func import init_weight
from utils.load_utils import load_pretrain
from engine.logger import get_logger

# --- 模型组件导入（来自 models 内部文件）---
from .net_utils import FeatureRectifyModule, FeatureFusionModule 
from .encoders.dual_vmamba import vssm_small as VSSM_small_Backbone
from .decoders.MLPDecoder import DecoderHead
from .decoders.fcnhead import FCNHead

# --- 外部库导入（用于 FLOPs 计算和权重初始化）---
from functools import partial
from fvcore.nn import FlopCountAnalysis, flop_count, parameter_count
from timm.models.layers import trunc_normal_ 


logger = get_logger()

# ----------------------------------------------------------------------
# 占位函数：FLOPs计算所需，如果不需要FLOPs计算，可以删除整个 flops() 方法
# ----------------------------------------------------------------------
def selective_scan_flop_jit(inputs, outputs):
    # 此函数在外部定义，为避免 NameError，我们定义一个占位符。
    return 0 


class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255), norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        
        # 默认通道数，对应 sigma_small
        self.channels = [96, 192, 384, 768]
        self.norm_layer = norm_layer
        self.aux_head = None
        self.deep_supervision = False # 简化后移除深度监督
        
        # --- 编码器 (Backbone) ---
        # 强制使用 sigma_small (V-MAMBA)
        logger.info('Using primary backbone: V-MAMBA (sigma_small)')
        self.backbone = VSSM_small_Backbone()

        # --- 解码器 (Decoder) ---
        # 优先使用 MLPDecoder，否则使用 FCNHead (作为默认/简单解码器)
        if cfg.decoder == 'MLPDecoder':
            logger.info('Using MLP Decoder')
            self.decode_head = DecoderHead(
                in_channels=self.channels, 
                num_classes=cfg.num_classes, 
                norm_layer=norm_layer, 
                embed_dim=cfg.decoder_embed_dim
            )
        else:
            logger.info('Using Default Decoder (FCN-32s)')
            self.decode_head = FCNHead(
                in_channels=self.channels[-1], 
                kernel_size=3, 
                num_classes=cfg.num_classes, 
                norm_layer=norm_layer
            )
        
        # --- 融合模块 (通常在编码器内部使用，但在这里实例化作为示例) ---
        # 确保 net_utils 被导入
        # self.fusion_module = FeatureFusionModule(dim=self.channels[0]) 

        self.criterion = criterion
        if self.criterion:
            # 依赖 utils/load_utils.py 和 utils/init_func.py
            self.init_weights(cfg, pretrained=cfg.pretrained_model)
    
    def init_weights(self, cfg, pretrained=None):
        """模型权重初始化和预训练权重加载"""
        if pretrained:
            logger.info('Loading pretrained model: {}'.format(pretrained))
            # 假设 VSSM_small_Backbone 有 init_weights 方法
            self.backbone.init_weights(pretrained=pretrained)
            
        logger.info('Initing weights ...')
        # 依赖 utils/init_func.py
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb, modal_x):
        """编码器-解码器 前向传播"""
        orisize = rgb.shape
        # 编码：RGB 和 深度图 (modal_x) 一起进入双流主干网络
        x = self.backbone(rgb, modal_x) 
        
        # 解码：特征进入分割头
        out = self.decode_head.forward(x) 
        
        # 插值回原始输入尺寸
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
        return out

    def forward(self, rgb, modal_x, label=None):
        """前向传播入口"""
        out = self.encode_decode(rgb, modal_x)
        
        if label is not None:
            # 训练模式：计算损失
            loss = self.criterion(out, label.long())
            return loss
            
        # 推理模式：返回分割输出
        return out
        
    def flops(self, shape=(3, 480, 640)):
        """模型FLOPs计算方法 (依赖 fvcore)"""
        # 这是一个计算方法示例，用于确认依赖关系，如果不需要可以删除
        supported_ops={
            "aten::silu": None, "aten::neg": None, "aten::exp": None, "aten::flip": None, 
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit, 
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanNRow": selective_scan_flop_jit,
        }

        model = copy.deepcopy(self)
        input = (torch.randn((1, *shape), device='cpu'), torch.randn((1, *shape), device='cpu'))
        
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=input, supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
    
    # ⚠️ 依赖的辅助函数（如果保留 flops() 方法则必须保留）
    def flops_selective_scan_fn(self, B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
        assert not with_complex 
        flops = 9 * B * L * D * N
        if with_D:
            flops += B * D * L
        if with_Z:
            flops += B * D * L    
        return flops