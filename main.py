# 依赖库 fastapi uvicorn python-multipart Pillow torch torchvision numpy transformers

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import torch
import traceback
import time
import base64
import os
from contextlib import asynccontextmanager

# 导入你的模型
from models.builder import EncoderDecoder as segmodel
from configs.config_nyu import config
from utils.pyt_utils import load_model

# 全局变量
model = None
device = None
depth_estimator = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 类别名称映射 - 使用NYU数据集的40个类别
CLASS_NAMES = {
    0: "墙壁",
    1: "地板",
    2: "橱柜",
    3: "床",
    4: "椅子",
    5: "沙发",
    6: "桌子",
    7: "门",
    8: "窗户",
    9: "书架",
    10: "图画",
    11: "柜台",
    12: "百叶窗",
    13: "书桌",
    14: "架子",
    15: "窗帘",
    16: "梳妆台",
    17: "枕头",
    18: "镜子",
    19: "地垫",
    20: "衣服",
    21: "天花板",
    22: "书籍",
    23: "冰箱",
    24: "电视",
    25: "纸张",
    26: "毛巾",
    27: "浴帘",
    28: "盒子",
    29: "白板",
    30: "人",
    31: "床头柜",
    32: "马桶",
    33: "水槽",
    34: "台灯",
    35: "浴缸",
    36: "包",
    37: "其他结构",
    38: "其他家具",
    39: "其他物品"
}


def _generate_pascal_colormap(N=256):
    """生成 Pascal VOC 风格的 colormap，用于固定且可区分的颜色"""
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        cid = i
        for j in range(8):
            r |= ((cid >> 0) & 1) << (7 - j)
            g |= ((cid >> 1) & 1) << (7 - j)
            b |= ((cid >> 2) & 1) << (7 - j)
            cid = cid >> 3
        cmap[i] = np.array([r, g, b], dtype=np.uint8)
    return cmap


# 为我们的 40 个类创建固定颜色表（如果类数大于256，会循环使用）
NUM_CLASSES = len(CLASS_NAMES)
CLASS_COLORS = _generate_pascal_colormap(256)[0:NUM_CLASSES]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期管理 - 替代过时的 on_event"""
    # 启动时执行
    print("🚀 学生社区家具资产动态感知与配置优化系统启动中...")
    print(f"📊 类别数量: {len(CLASS_NAMES)}")
    print(f"📋 类别映射: {CLASS_NAMES}")
    success = load_segmentation_model()
    if success:
        print("✅ 服务准备就绪，可访问 http://localhost:8000/docs")
        print("📁 数据模态: RGB + 自动估计深度图")
        print("🔧 支持接口: POST /segment (多模态), POST /segment_with_depth (手动输入深度图)")
    else:
        print("❌ 服务启动失败，请检查模型配置")

    yield  # 服务运行期间

    # 关闭时执行（如果需要清理资源）
    print("🛑 服务关闭中...")


app = FastAPI(
    title="学生社区家具资产感知配置API",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_depth_estimation_model():
    """
    加载深度估计模型，支持固定缓存路径，避免重复下载
    """
    global BASE_DIR

    # 固定缓存路径（可根据你的服务器环境修改）
    cache_dir = os.path.join(BASE_DIR, "huggingface_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # 先尝试离线加载
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_CACHE'] = cache_dir

    try:
        print(f"尝试从本地缓存加载深度估计模型（{cache_dir}）...")
        from transformers import pipeline
        depth_pipeline = pipeline(
            "depth-estimation",
            model="Intel/dpt-hybrid-midas",
            feature_extractor="Intel/dpt-hybrid-midas",
            cache_dir=cache_dir
        )
        print("✅ 本地缓存加载成功")
        return depth_pipeline

    except Exception as e:
        print(f"本地缓存加载失败: {e}")
        print("尝试在线下载并缓存...")

        # 切换在线模式
        os.environ['TRANSFORMERS_OFFLINE'] = '0'
        os.environ['HF_DATASETS_OFFLINE'] = '0'

        try:
            from transformers import pipeline
            depth_pipeline = pipeline(
                "depth-estimation",
                model="Intel/dpt-hybrid-midas",
                feature_extractor="Intel/dpt-hybrid-midas",
                cache_dir=cache_dir
            )
            print(f"✅ 在线下载成功，已缓存到 {cache_dir}")
            return depth_pipeline
        except Exception as e2:
            print(f"❌ 模型加载失败: {e2}")
            return None


def load_segmentation_model():
    """加载多模态分割模型"""
    global model, device, depth_estimator

    try:
        print("开始加载多模态分割模型...")
        print(f"使用数据集: NYU (40个类别)")

        # 1. 确定设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")

        # 2. 构建模型结构
        try:
            model = segmodel(cfg=config, criterion=None, norm_layer=torch.nn.BatchNorm2d)
        except TypeError as e:
            print(f"方式1失败: {e}，尝试方式2...")
            try:
                model = segmodel(cfg=config, norm_layer=torch.nn.BatchNorm2d)
            except TypeError as e2:
                print(f"方式2失败: {e2}，尝试方式3...")
                try:
                    model = segmodel(config)
                except Exception as e3:
                    print(f"所有方式都失败: {e3}")
                    return False

        print("模型结构构建成功")

        model_path = os.path.join(BASE_DIR, "checkpoints/nyudepthv2-sigma-small-epoch-390.pth")
        print(f"尝试从 {model_path} 加载权重...")

        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            print(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'not_dict'}")

            # 根据checkpoint的结构加载权重
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
                print("从checkpoint['model']加载权重成功")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print("从checkpoint['state_dict']加载权重成功")
            elif 'net' in checkpoint:
                model.load_state_dict(checkpoint['net'])
                print("从checkpoint['net']加载权重成功")
            else:
                model.load_state_dict(checkpoint)
                print("直接从checkpoint加载权重成功")

        except Exception as e:
            print(f"权重加载失败: {e}")
            traceback.print_exc()
            return False

        # 4. 设置为评估模式并移动到设备
        model.eval()
        model.to(device)

        # 5. 加载深度估计模型
        print("加载深度估计模型...")
        depth_estimator = load_depth_estimation_model()

        if depth_estimator is not None:
            print("✅ 深度估计模型加载完成!")
        else:
            print("⚠️  深度估计模型加载失败，将使用备用方案")

        print("✅ 多模态分割模型加载完成!")
        return True

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        traceback.print_exc()
        return False


def estimate_depth_from_rgb(rgb_image):
    """从RGB图像估计深度图 - 支持备用方案"""
    global depth_estimator

    try:
        print("开始估计深度图...")

        if depth_estimator is not None:
            # 使用深度估计模型
            depth_result = depth_estimator(rgb_image)
            depth_map = depth_result["depth"]

            # 转换为numpy数组并归一化
            depth_array = np.array(depth_map)
            depth_normalized = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min()) * 255
            depth_image = Image.fromarray(depth_normalized.astype(np.uint8))

            print("✅ 使用深度估计模型生成深度图")
        else:
            # 备用方案：使用简单的灰度图作为深度图
            print("使用备用深度估计方案...")
            gray = rgb_image.convert('L')
            depth_array = np.array(gray)
            # 添加一些简单的深度效果（中心区域更近）
            h, w = depth_array.shape
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h / 2, w / 2
            mask = 1 - np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / np.sqrt(center_x ** 2 + center_y ** 2)
            depth_array = (depth_array * mask).astype(np.uint8)
            depth_image = Image.fromarray(depth_array)
            print("✅ 使用备用方案生成深度图")

        return depth_image

    except Exception as e:
        print(f"❌ 深度图估计失败: {e}")
        # 返回一个默认的深度图（全128）
        return Image.new('L', rgb_image.size, 128)


def preprocess_image(image: Image, mode: str = 'RGB'):
    """预处理图片 - 修复版本：确保都是3通道"""
    # 根据config获取目标尺寸，如果没有则使用默认尺寸
    if hasattr(config, 'input_size'):
        target_size = config.input_size
    else:
        target_size = (512, 512)  # 默认尺寸

    # 转换颜色模式 - 确保都是3通道
    if mode == 'RGB':
        image = image.convert('RGB')
    else:
        # 对于其他模态，将单通道图像转换为3通道
        image = image.convert('L')  # 先转为灰度
        # 创建一个3通道的图像，所有通道值相同
        rgb_image = Image.new('RGB', image.size)
        # 将灰度值复制到所有三个通道
        rgb_image.paste(image, (0, 0, image.width, image.height))
        image = rgb_image

    # 调整尺寸
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(image)


def image_to_tensor(image_np):
    """将numpy数组转换为模型需要的tensor格式 - 修复版本"""
    # 现在所有图像都是3通道的: (H, W, 3) -> (3, H, W) -> (1, 3, H, W)
    tensor = torch.from_numpy(image_np).float()
    tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
    return tensor.to(device)


def process_segmentation(rgb_tensor, depth_tensor):
    """执行分割推理的通用函数"""
    with torch.no_grad():
        output = model(rgb_tensor, depth_tensor)
        print(f"模型输出尺寸: {output.shape}")
        print(f"模型输出值范围: [{output.min():.3f}, {output.max():.3f}]")

        # 获取分割结果
        mask = output.argmax(dim=1).squeeze().cpu().numpy()
        confidence = output.softmax(dim=1).max(dim=1)[0].squeeze().cpu().numpy()

        print(f"分割掩码值统计: 最小值={mask.min()}, 最大值={mask.max()}, 唯一值={np.unique(mask)}")
        print(
            f"置信度统计: 最小值={confidence.min():.3f}, 最大值={confidence.max():.3f}, 平均值={confidence.mean():.3f}")

        return mask, confidence


def create_segmentation_mask(mask):
    """创建分割掩码的Base64图像"""
    try:
        # 检查mask是否有有效值
        mask_max = mask.max()
        mask_min = mask.min()
        print(f"分割掩码值范围: [{mask_min}, {mask_max}]")
        # 生成彩色掩码：为每个类别分配固定颜色
        h, w = mask.shape
        if mask_max == mask_min:
            # 所有值相同时直接填充该类别颜色或黑色
            color = CLASS_COLORS[int(mask_min)] if 0 <= int(mask_min) < NUM_CLASSES else np.array([0, 0, 0], dtype=np.uint8)
            colored = np.zeros((h, w, 3), dtype=np.uint8)
            colored[:, :] = color
            print("警告: 分割结果所有像素值相同，已用单色填充")
        else:
            # 确保mask为整数索引且在[0, NUM_CLASSES-1]范围内
            mask_idx = mask.astype(np.int32)
            mask_idx = np.clip(mask_idx, 0, NUM_CLASSES - 1)
            # 使用向量化索引将类ID映射到颜色
            colored = CLASS_COLORS[mask_idx]

        mask_img = Image.fromarray(colored)
        mask_buffer = io.BytesIO()
        mask_img.save(mask_buffer, format="PNG")
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
        return mask_base64

    except Exception as e:
        print(f"掩码后处理失败: {e}")
        # 创建一个默认的掩码
        mask_img = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
        mask_buffer = io.BytesIO()
        mask_img.save(mask_buffer, format="PNG")
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
        return mask_base64


def analyze_segmentation_results(mask, confidence, processing_time):
    """分析分割结果并返回统计信息（过滤掉占比小于0.5%的类别）"""
    unique_labels, counts = np.unique(mask, return_counts=True)

    # 添加详细的调试信息
    print(f"=== 详细统计信息 ===")
    print(f"unique_labels: {unique_labels}")
    print(f"counts: {counts}")
    print(f"mask总像素数: {mask.size}")
    print(f"mask形状: {mask.shape}")

    # 构建详细的类别信息 - 严格过滤掉占比小于0.5%的类别
    class_details = []
    main_objects = []
    filtered_detected_classes = []

    for label, count in zip(unique_labels, counts):
        class_id = int(label)
        percentage = (count / mask.size) * 100

        # 详细的调试信息
        print(f"类别 {class_id} ({CLASS_NAMES.get(class_id, '未知')}): count={count}, percentage={percentage:.10f}%")

        # 严格过滤：只保留占比大于0.5%的类别
        if percentage > 0.5:
            rounded_percentage = round(percentage, 2)
            class_name = CLASS_NAMES.get(class_id, f"未知类别{class_id}")
            # 获取颜色并格式化为十六进制字符串和 RGB 列表
            try:
                col = CLASS_COLORS[class_id]
                r, g, b = int(col[0]), int(col[1]), int(col[2])
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                rgb_color = [r, g, b]
            except Exception:
                hex_color = "#000000"
                rgb_color = [0, 0, 0]

            class_info = {
                "class_id": class_id,
                "class_name": class_name,
                "pixel_count": int(count),
                "percentage": rounded_percentage,
                "color": hex_color,
                "color_rgb": rgb_color
            }
            class_details.append(class_info)
            filtered_detected_classes.append(class_id)

            # 如果是非未标记类别且占比大于1%，认为是主要物体
            if class_id != 0 and percentage > 1.0:
                main_objects.append(class_name)
        else:
            print(f"⚠️  过滤掉类别 {class_id}，因为占比为 {percentage:.6f}%")

    # 按占比从大到小排序类别详情
    class_details = sorted(class_details, key=lambda x: x["percentage"], reverse=True)

    print(f"=== 过滤后结果 ===")
    print(f"分割完成! 检测到 {len(filtered_detected_classes)} 个类别 (过滤后), 耗时: {processing_time}s")
    print("类别详情:")
    for class_info in class_details:
        print(f"  - {class_info['class_name']} (ID: {class_info['class_id']}): {class_info['percentage']}%")

    return {
        "detected_classes": filtered_detected_classes,
        "class_details": class_details,
        "main_objects": main_objects,
        "average_confidence": float(confidence.mean())
    }


@app.post("/segment")
async def segment_multimodal(
        rgb_image: UploadFile = File(..., description="RGB彩色图像"),
        use_estimated_depth: bool = Form(True, description="是否使用估计的深度图")
):
    """
    家具资产感知分割接口（自动生成深度图）
    - rgb_image: RGB彩色图像
    - use_estimated_depth: 是否使用估计的深度图（True:自动生成, False:需要上传深度图）
    """
    start_time = time.time()

    try:
        # 1. 验证模型是否加载
        if model is None:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "模型未加载成功，请检查服务状态"
                }
            )

        # 2. 验证文件类型
        if not rgb_image.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "RGB图像必须是图片格式",
                    "file_type": rgb_image.content_type
                }
            )

        print(f"开始处理多模态分割请求...")
        print(f"使用估计深度图: {use_estimated_depth}")

        # 3. 读取并预处理RGB图像
        rgb_bytes = await rgb_image.read()
        rgb_img = Image.open(io.BytesIO(rgb_bytes))
        rgb_np = preprocess_image(rgb_img, 'RGB')

        print(f"RGB图像尺寸: {rgb_np.shape}")

        # 4. 生成或处理深度图像
        if use_estimated_depth:
            # 自动生成深度图
            print("自动生成深度图中...")
            depth_img = estimate_depth_from_rgb(rgb_img)
            depth_np = preprocess_image(depth_img, 'modal')
            depth_source = "estimated"
        else:
            # 这里可以扩展为接收上传的深度图
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "当前版本仅支持自动生成深度图"
                }
            )

        print(f"深度图像尺寸: {depth_np.shape}")
        print(f"深度图来源: {depth_source}")

        # 5. 转换为tensor
        rgb_tensor = image_to_tensor(rgb_np)
        depth_tensor = image_to_tensor(depth_np)

        print(f"RGB tensor尺寸: {rgb_tensor.shape}")
        print(f"深度 tensor尺寸: {depth_tensor.shape}")

        # 6. 多模态模型推理
        mask, confidence = process_segmentation(rgb_tensor, depth_tensor)

        # 7. 后处理：将分割掩码转换为Base64便于前端显示
        mask_base64 = create_segmentation_mask(mask)

        # 8. 统计分割结果和类别信息
        processing_time = round(time.time() - start_time, 3)
        analysis_results = analyze_segmentation_results(mask, confidence, processing_time)

        # 9. 返回详细结果
        return {
            "status": "success",
            "processing_time": processing_time,
            "depth_source": depth_source,
            "depth_model_status": "loaded" if depth_estimator is not None else "fallback",
            "segmentation_map": mask_base64,
            "mask_shape": mask.shape,
            "detected_classes": analysis_results["detected_classes"],
            "class_details": analysis_results["class_details"],
            "main_objects": analysis_results["main_objects"],
            "class_names_mapping": CLASS_NAMES,
            "average_confidence": analysis_results["average_confidence"],
            "message": f"分割完成，检测到 {len(analysis_results['main_objects'])} 个主要物体" if analysis_results[
                'main_objects'] else "主要检测到未标记区域",
            "note": "已过滤占比小于0.01%的类别"
        }

    except Exception as e:
        error_msg = f"分割失败: {str(e)}"
        print(error_msg)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": error_msg,
                "processing_time": round(time.time() - start_time, 3)
            }
        )


@app.post("/segment_with_depth")
async def segment_with_depth(
        rgb_image: UploadFile = File(..., description="RGB彩色图像"),
        depth_image: UploadFile = File(..., description="深度图图像")
):
    """
    家具资产感知分割接口（手动输入深度图）
    - rgb_image: RGB彩色图像
    - depth_image: 深度图图像
    """
    start_time = time.time()

    try:
        # 1. 验证模型是否加载
        if model is None:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "模型未加载成功，请检查服务状态"
                }
            )

        # 2. 验证文件类型
        if not rgb_image.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "RGB图像必须是图片格式",
                    "file_type": rgb_image.content_type
                }
            )

        if not depth_image.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "深度图必须是图片格式",
                    "file_type": depth_image.content_type
                }
            )

        print(f"开始处理手动深度图分割请求...")

        # 3. 读取并预处理RGB图像和深度图
        rgb_bytes = await rgb_image.read()
        depth_bytes = await depth_image.read()

        rgb_img = Image.open(io.BytesIO(rgb_bytes))
        depth_img = Image.open(io.BytesIO(depth_bytes))

        rgb_np = preprocess_image(rgb_img, 'RGB')
        depth_np = preprocess_image(depth_img, 'modal')

        print(f"RGB图像尺寸: {rgb_np.shape}")
        print(f"深度图像尺寸: {depth_np.shape}")
        print(f"深度图来源: manual")

        # 4. 转换为tensor
        rgb_tensor = image_to_tensor(rgb_np)
        depth_tensor = image_to_tensor(depth_np)

        print(f"RGB tensor尺寸: {rgb_tensor.shape}")
        print(f"深度 tensor尺寸: {depth_tensor.shape}")

        # 5. 多模态模型推理
        mask, confidence = process_segmentation(rgb_tensor, depth_tensor)

        # 6. 后处理：将分割掩码转换为Base64便于前端显示
        mask_base64 = create_segmentation_mask(mask)

        # 7. 统计分割结果和类别信息
        processing_time = round(time.time() - start_time, 3)
        analysis_results = analyze_segmentation_results(mask, confidence, processing_time)

        # 8. 返回详细结果
        return {
            "status": "success",
            "processing_time": processing_time,
            "depth_source": "manual",
            "depth_model_status": "manual_input",
            "segmentation_map": mask_base64,
            "mask_shape": mask.shape,
            "detected_classes": analysis_results["detected_classes"],
            "class_details": analysis_results["class_details"],
            "main_objects": analysis_results["main_objects"],
            "class_names_mapping": CLASS_NAMES,
            "average_confidence": analysis_results["average_confidence"],
            "message": f"分割完成，检测到 {len(analysis_results['main_objects'])} 个主要物体" if analysis_results[
                'main_objects'] else "主要检测到未标记区域",
            "note": "已过滤占比小于0.01%的类别，使用手动输入的深度图"
        }

    except Exception as e:
        error_msg = f"分割失败: {str(e)}"
        print(error_msg)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": error_msg,
                "processing_time": round(time.time() - start_time, 3)
            }
        )


@app.get("/", response_class=HTMLResponse)
async def root():
    """尝试返回项目根目录下的 index.html；如果不存在则返回一个简单的 JSON 信息"""
    try:
        index_path = Path("index.html")
        if index_path.exists():
            content = index_path.read_text(encoding="utf-8")
            return HTMLResponse(content=content, status_code=200)
        else:
            # 回退到原始的 JSON 信息
            return JSONResponse(
                status_code=200,
                content={
                    "message": "学生社区家具资产动态感知与配置优化服务运行中（自动深度估计 + 手动深度图输入）",
                    "model_loaded": model is not None,
                    "depth_estimator_loaded": depth_estimator is not None,
                    "device": str(device) if device else "unknown",
                    "data_modalities": ["RGB", "自动估计深度图", "手动输入深度图"],
                    "class_names": CLASS_NAMES,
                    "endpoints": {
                        "multimodal_segmentation": "POST /segment",
                        "multimodal_segmentation_with_depth": "POST /segment_with_depth",
                        "health_check": "GET /health",
                        "model_info": "GET /model-info",
                        "api_docs": "GET /docs"
                    }
                }
            )
    except Exception as e:
        # 任何读取错误都回退为 JSON
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"无法读取 index.html: {e}"
            }
        )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "depth_estimator_loaded": depth_estimator is not None,
        "depth_model_status": "loaded" if depth_estimator is not None else "fallback",
        "device": str(device) if device else "unknown",
        "service": "multimodal_segmentation",
        "timestamp": time.time()
    }


@app.get("/model-info")
async def model_info():
    if model is None:
        return {"status": "model_not_loaded"}

    return {
        "model_type": "Multimodal_EncoderDecoder",
        "modality": "RGB + 深度图",
        "input_modalities": ["RGB", "深度图"],
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "device": str(device),
        "depth_estimation": "available" if depth_estimator is not None else "fallback_mode",
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "status": "loaded"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
