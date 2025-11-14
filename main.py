from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
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
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 导入你的模型
# 确保 models/builder.py 和 configs/config_nyu.py 存在于 PYTHONPATH 中
try:
    from models.builder import EncoderDecoder as segmodel
    from configs.config_nyu import config # config 变量虽然未直接使用，但用于模型初始化
    # 【已移除】from utils.pyt_utils import load_model
except ImportError as e:
    logging.error(f"导入自定义模块失败: {e}. 请确保项目结构正确并设置了 PYTHONPATH。")

# 全局变量
model = None
device = None
depth_estimator = None

# 类别名称映射 (NYU数据集)
CLASS_NAMES = {
    0: "墙壁", 1: "地板", 2: "橱柜", 3: "床", 4: "椅子", 5: "沙发", 6: "桌子", 7: "门", 8: "窗户",
    9: "书架", 10: "图画", 11: "柜台", 12: "百叶窗", 13: "书桌", 14: "架子", 15: "窗帘", 16: "梳妆台",
    17: "枕头", 18: "镜子", 19: "地垫", 20: "衣服", 21: "天花板", 22: "书籍", 23: "冰箱", 24: "电视",
    25: "纸张", 26: "毛巾", 27: "浴帘", 28: "盒子", 29: "白板", 30: "人", 31: "床头柜", 32: "马桶",
    33: "水槽", 34: "台灯", 35: "浴缸", 36: "包", 37: "其他结构", 38: "其他家具", 39: "其他物品"
}

# 固定的类别颜色 (HSL 格式)
CLASS_COLORS = {
    0: "hsl(20, 70%, 50%)", 1: "hsl(40, 70%, 50%)", 2: "hsl(60, 70%, 50%)", 3: "hsl(80, 70%, 50%)",
    4: "hsl(100, 70%, 50%)", 5: "hsl(120, 70%, 50%)", 6: "hsl(140, 70%, 50%)", 7: "hsl(160, 70%, 50%)",
    8: "hsl(180, 70%, 50%)", 9: "hsl(200, 70%, 50%)", 10: "hsl(220, 70%, 50%)", 11: "hsl(240, 70%, 50%)",
    12: "hsl(260, 70%, 50%)", 13: "hsl(280, 70%, 50%)", 14: "hsl(300, 70%, 50%)", 15: "hsl(320, 70%, 50%)",
    16: "hsl(340, 70%, 50%)", 17: "hsl(10, 50%, 60%)", 18: "hsl(30, 50%, 60%)", 19: "hsl(50, 50%, 60%)",
    20: "hsl(70, 50%, 60%)", 21: "hsl(90, 50%, 60%)", 22: "hsl(110, 50%, 60%)", 23: "hsl(130, 50%, 60%)",
    24: "hsl(150, 50%, 60%)", 25: "hsl(170, 50%, 60%)", 26: "hsl(190, 50%, 60%)", 27: "hsl(210, 50%, 60%)",
    28: "hsl(230, 50%, 60%)", 29: "hsl(250, 50%, 60%)", 30: "hsl(270, 50%, 60%)", 31: "hsl(290, 50%, 60%)",
    32: "hsl(310, 50%, 60%)", 33: "hsl(330, 50%, 60%)", 34: "hsl(350, 50%, 60%)", 35: "hsl(20, 30%, 70%)",
    36: "hsl(40, 30%, 70%)", 37: "hsl(60, 30%, 70%)", 38: "hsl(80, 30%, 70%)", 39: "hsl(100, 30%, 70%)"
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期管理 - 模型加载在服务启动时执行"""
    logging.info("🚀 多模态语义分割服务启动中...")
    logging.info(f"📊 类别数量: {len(CLASS_NAMES)}")
    success = load_segmentation_model()
    if success:
        logging.info("✅ 服务准备就绪，可访问 http://localhost:8000/docs")
    else:
        logging.error("❌ 服务启动失败，请检查模型配置")

    yield  # 服务运行期间

    logging.info("🛑 服务关闭中...")


app = FastAPI(
    title="多模态语义分割API",
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
    """加载深度估计模型 - 两级策略"""
    try:
        # 第一级：尝试离线模式
        logging.info("第一级：尝试离线模式加载...")
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'

        from transformers import pipeline
        depth_pipeline = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
        logging.info("✅ 离线模式加载成功")
        return depth_pipeline

    except Exception as e:
        logging.warning(f"离线模式失败: {e}")

        try:
            # 第二级：从网络下载
            logging.info("第二级：尝试从网络下载...")
            os.environ['TRANSFORMERS_OFFLINE'] = '0'  # 重置为在线模式
            from transformers import pipeline
            depth_pipeline = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
            logging.info("✅ 网络下载成功")
            return depth_pipeline

        except Exception as e2:
            logging.error(f"❌ 所有加载方法都失败: {e2}")
            return None


def load_segmentation_model():
    """加载多模态分割模型"""
    global model, device, depth_estimator

    try:
        logging.info("开始加载多模态分割模型...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"使用设备: {device}")

        # 1. 构建模型结构
        # 使用配置对象 config
        try:
            model = segmodel(cfg=config, criterion=None, norm_layer=torch.nn.BatchNorm2d)
        except TypeError as e:
            logging.warning(f"方式1失败: {e}，尝试方式2...")
            try:
                model = segmodel(cfg=config, norm_layer=torch.nn.BatchNorm2d)
            except TypeError as e2:
                logging.warning(f"方式2失败: {e2}，尝试方式3...")
                try:
                    model = segmodel(config)
                except Exception as e3:
                    logging.error(f"所有模型结构构建方式都失败: {e3}")
                    return False

        logging.info("模型结构构建成功")

        # 2. 加载权重
        model_path = "nyudepthv2-sigma-small-epoch-390.pth"
        logging.info(f"尝试从 {model_path} 加载权重...")

        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            # 根据checkpoint的结构加载权重
            state_dict = None
            if isinstance(checkpoint, dict):
                for key in ['model', 'state_dict', 'net']:
                    if key in checkpoint:
                        state_dict = checkpoint[key]
                        logging.info(f"从checkpoint['{key}']加载权重成功")
                        break
            if state_dict is None:
                state_dict = checkpoint
                logging.info("直接从checkpoint加载权重成功")

            # 假设 checkpoint 已经用 torch.load 读到变量 checkpoint
            if isinstance(checkpoint, dict) and any(k in checkpoint for k in ('model', 'state_dict', 'net')):
                state = None
                for k in ('model', 'state_dict', 'net'):
                    if k in checkpoint:
                        state = checkpoint[k]
                        break
            else:
                state = checkpoint

            # model_state 为当前模型的 state_dict（目标 key -> tensor）
            model_state = model.state_dict()

            # 过滤：只保留名字一致且形状一致的参数
            filtered = {}
            mismatch = []
            for k, v in state.items():
                if k in model_state:
                    if getattr(v, 'shape', None) == getattr(model_state[k], 'shape', None):
                        filtered[k] = v
                    else:
                        mismatch.append((k, getattr(v, 'shape', None), getattr(model_state[k], 'shape', None)))
            # else: key 不在模型中 -> 将被视为 unexpected

            logging.info(f"准备加载 {len(filtered)} / {len(model_state)} 个匹配的参数")
            if mismatch:
                logging.warning(f"发现形状不匹配的项（将跳过）: {mismatch[:10]} ...")

            # 加载并使用 strict=False 打印 missing/unexpected
            load_res = model.load_state_dict(filtered, strict=False)
            logging.info(f"load_state_dict 返回 missing_keys={load_res.missing_keys}, unexpected_keys={load_res.unexpected_keys}")

        except Exception as e:
            logging.error(f"权重加载失败: {e}")
            traceback.print_exc()
            return False

        # 3. 设置为评估模式并移动到设备
        model.eval()
        model.to(device)

        # 4. 加载深度估计模型
        logging.info("加载深度估计模型...")
        depth_estimator = load_depth_estimation_model()

        if depth_estimator is not None:
            logging.info("✅ 深度估计模型加载完成!")
        else:
            logging.warning("⚠️  深度估计模型加载失败，将使用备用方案")

        logging.info("✅ NYU多模态分割模型加载完成!")
        return True

    except Exception as e:
        logging.error(f"❌ 模型加载失败: {e}")
        traceback.print_exc()
        return False


def estimate_depth_from_rgb(rgb_image):
    """从RGB图像估计深度图 - 支持备用方案"""
    global depth_estimator

    try:
        logging.info("开始估计深度图...")

        if depth_estimator is not None:
            depth_result = depth_estimator(rgb_image)
            depth_map = depth_result["depth"]

            depth_array = np.array(depth_map)
            # 归一化到 0-255
            depth_normalized = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min()) * 255
            depth_image = Image.fromarray(depth_normalized.astype(np.uint8))

            logging.info("✅ 使用深度估计模型生成深度图")
        else:
            # 备用方案：使用简单的灰度图作为深度图
            logging.warning("使用备用深度估计方案...")
            gray = rgb_image.convert('L')
            depth_array = np.array(gray)
            # 简单的深度效果（中心区域更近）
            h, w = depth_array.shape
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h / 2, w / 2
            mask_distance = 1 - np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / np.sqrt(center_x ** 2 + center_y ** 2)
            depth_array = (depth_array * mask_distance).astype(np.uint8)
            depth_image = Image.fromarray(depth_array)
            logging.info("✅ 使用备用方案生成深度图")

        return depth_image

    except Exception as e:
        logging.error(f"❌ 深度图估计失败: {e}")
        return Image.new('L', rgb_image.size, 128) # 返回一个默认的深度图


def preprocess_image(image: Image, mode: str = 'RGB'):
    """预处理图片：颜色模式转换和尺寸调整"""
    # 尝试从配置中获取目标尺寸
    target_size = getattr(config, 'input_size', (512, 512))

    # 转换颜色模式
    if mode == 'RGB':
        image = image.convert('RGB')
    else:
        # 深度图（或其他模态）通常是灰度图，但模型输入可能要求3通道
        image = image.convert('L') # 先转为灰度
        # 创建一个新的RGB图像并粘贴灰度图，使其具有3个相同的通道
        rgb_image = Image.new('RGB', image.size)
        rgb_image.paste(image, (0, 0, image.width, image.height))
        image = rgb_image

    # 调整尺寸
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(image)


def image_to_tensor(image_np):
    """将numpy数组转换为模型需要的tensor格式"""
    # (H, W, 3) -> (3, H, W) -> (1, 3, H, W)
    tensor = torch.from_numpy(image_np).float()
    tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
    return tensor.to(device)


def process_segmentation(rgb_tensor, depth_tensor):
    """执行分割推理的通用函数"""
    with torch.no_grad():
        output = model(rgb_tensor, depth_tensor)
        
        # 调整尺寸以匹配目标尺寸
        if output.shape[-2:] != rgb_tensor.shape[-2:]:
            output = torch.nn.functional.interpolate(
                output, 
                size=rgb_tensor.shape[-2:], 
                mode='bilinear', 
                align_corners=True
            )

        # 获取分割结果
        mask = output.argmax(dim=1).squeeze().cpu().numpy()
        confidence = output.softmax(dim=1).max(dim=1)[0].squeeze().cpu().numpy()

        return mask, confidence


def create_segmentation_mask(mask):
    """创建分割掩码的Base64图像"""
    try:
        mask_max = mask.max()
        mask_min = mask.min()

        if mask_max == mask_min:
            mask_normalized = np.zeros_like(mask, dtype=np.uint8)
        else:
            # 正常归一化到0-255
            mask_normalized = ((mask - mask_min) * 255 / (mask_max - mask_min)).astype(np.uint8)

        mask_img = Image.fromarray(mask_normalized)
        mask_buffer = io.BytesIO()
        mask_img.save(mask_buffer, format="PNG")
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
        return mask_base64

    except Exception as e:
        logging.error(f"掩码后处理失败: {e}")
        # 创建一个默认的掩码
        mask_img = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
        mask_buffer = io.BytesIO()
        mask_img.save(mask_buffer, format="PNG")
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
        return mask_base64


def analyze_segmentation_results(mask, confidence, processing_time):
    """分析分割结果并返回统计信息"""
    unique_labels, counts = np.unique(mask, return_counts=True)
    
    mask_size = mask.size
    class_details = []
    main_objects = []
    filtered_detected_classes = []

    for label, count in zip(unique_labels, counts):
        class_id = int(label)
        percentage = (count / mask_size) * 100

        # 过滤：只保留占比大于0.001%的类别
        if percentage > 0.001:
            rounded_percentage = round(percentage, 2)
            class_name = CLASS_NAMES.get(class_id, f"未知类别{class_id}")
            
            class_info = {
                "class_id": class_id,
                "class_name": class_name,
                "pixel_count": int(count),
                "percentage": rounded_percentage,
                "color": CLASS_COLORS.get(class_id, "hsl(0, 0%, 50%)")
            }
            class_details.append(class_info)
            filtered_detected_classes.append(class_id)

            # 如果是非未标记类别（通常为0）且占比大于1%，认为是主要物体
            if class_id != 0 and percentage > 1.0:
                main_objects.append(class_name)

    logging.info(f"分割完成! 检测到 {len(filtered_detected_classes)} 个类别 (过滤后), 耗时: {processing_time}s")

    return {
        "detected_classes": filtered_detected_classes,
        "class_details": class_details,
        "main_objects": main_objects,
        "average_confidence": float(confidence.mean())
    }


@app.post("/segment")
async def segment_multimodal(
        rgb_image: UploadFile = File(..., description="RGB彩色图像"),
        # 移除 use_estimated_depth 的 Form 定义，因为该接口默认且只支持自动估计
):
    """
    多模态语义分割接口（自动生成深度图）
    - rgb_image: RGB彩色图像
    """
    start_time = time.time()

    try:
        # 1. 验证模型是否加载
        if model is None:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "模型未加载成功，请检查服务状态"}
            )

        # 2. 验证文件类型
        if not rgb_image.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "RGB图像必须是图片格式"}
            )

        logging.info("开始处理多模态分割请求（自动深度图）...")

        # 3. 读取并预处理RGB图像
        rgb_bytes = await rgb_image.read()
        rgb_img = Image.open(io.BytesIO(rgb_bytes))
        rgb_np = preprocess_image(rgb_img, 'RGB')
        
        # 4. 生成深度图像
        depth_img = estimate_depth_from_rgb(rgb_img)
        depth_np = preprocess_image(depth_img, 'modal')
        depth_source = "estimated"

        # 5. 转换为tensor
        rgb_tensor = image_to_tensor(rgb_np)
        depth_tensor = image_to_tensor(depth_np)

        # 6. 多模态模型推理
        mask, confidence = process_segmentation(rgb_tensor, depth_tensor)

        # 7. 后处理：将分割掩码转换为Base64
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
            "message": f"多模态分割完成，检测到 {len(analysis_results['main_objects'])} 个主要物体",
            "note": "已过滤占比小于0.01%的类别"
        }

    except Exception as e:
        error_msg = f"多模态分割失败: {str(e)}"
        logging.error(error_msg)
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
    多模态语义分割接口（手动输入深度图）
    - rgb_image: RGB彩色图像
    - depth_image: 深度图图像
    """
    start_time = time.time()

    try:
        # 1. 验证模型是否加载
        if model is None:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "模型未加载成功，请检查服务状态"}
            )

        # 2. 验证文件类型
        if not rgb_image.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "RGB图像必须是图片格式"}
            )

        if not depth_image.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "深度图必须是图片格式"}
            )

        logging.info("开始处理手动深度图分割请求...")

        # 3. 读取并预处理RGB图像和深度图
        rgb_bytes = await rgb_image.read()
        depth_bytes = await depth_image.read()

        rgb_img = Image.open(io.BytesIO(rgb_bytes))
        depth_img = Image.open(io.BytesIO(depth_bytes))

        rgb_np = preprocess_image(rgb_img, 'RGB')
        depth_np = preprocess_image(depth_img, 'modal')
        depth_source = "manual"

        # 4. 转换为tensor
        rgb_tensor = image_to_tensor(rgb_np)
        depth_tensor = image_to_tensor(depth_np)

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
            "depth_source": depth_source,
            "depth_model_status": "manual_input",
            "segmentation_map": mask_base64,
            "mask_shape": mask.shape,
            "detected_classes": analysis_results["detected_classes"],
            "class_details": analysis_results["class_details"],
            "main_objects": analysis_results["main_objects"],
            "class_names_mapping": CLASS_NAMES,
            "average_confidence": analysis_results["average_confidence"],
            "message": f"多模态分割完成，检测到 {len(analysis_results['main_objects'])} 个主要物体",
            "note": "已过滤占比小于0.01%的类别，使用手动输入的深度图"
        }

    except Exception as e:
        error_msg = f"多模态分割失败: {str(e)}"
        logging.error(error_msg)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": error_msg,
                "processing_time": round(time.time() - start_time, 3)
            }
        )


@app.get("/")
async def root():
    return {
        "message": "多模态语义分割API服务运行中(自动深度估计 + 手动深度图输入)",
        "model_loaded": model is not None,
        "depth_estimator_loaded": depth_estimator is not None,
        "device": str(device) if device else "unknown",
        "endpoints": {
            "multimodal_segmentation": "POST /segment",
            "multimodal_segmentation_with_depth": "POST /segment_with_depth",
            "health_check": "GET /health",
            "model_info": "GET /model-info",
            "api_docs": "GET /docs",
            "frontend_page": "GET /upload"
        }
    }


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
    from fastapi.responses import HTMLResponse

    @app.get("/upload", response_class=HTMLResponse)
    async def upload_page():
        # 确保 index.html 存在于同一目录下
        try:
            with open("index.html", "r", encoding="utf-8") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        except FileNotFoundError:
             return HTMLResponse(
                content="<h1>错误：未找到 index.html 文件</h1><p>请将前端文件保存为 index.html 并放置于 main.py 同一目录下。</p>",
                status_code=404
            )
             
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")