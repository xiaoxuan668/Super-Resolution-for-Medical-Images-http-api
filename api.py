from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
from model.models import Generator
from model.model_config import DEVICE, upscale_factor, model_load_path
from scripts.model_metrics import ssim  # 假设你有SSIM计算函数（可复用项目中的逻辑）

# 初始化FastAPI应用
app = FastAPI(
    title="医学图像超分辨率API",
    description="基于Swift-SRGAN的X射线图像超分辨率服务"
)

# 允许跨域请求（前端调用需要）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境需指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局加载模型（启动时加载，避免重复加载）
model = None


def init_model():
    """初始化生成器模型并加载权重"""
    global model
    if model is None:
        model = Generator(upscale_factor=upscale_factor).to(DEVICE)
        # 加载模型权重（适配原项目的权重格式）
        state_dict = torch.load(model_load_path, map_location=torch.device(DEVICE))
        model.load_state_dict(state_dict["model"])
        model.eval()  # 推理模式
    return model


# 启动时初始化模型
init_model()


def prepare_image(image: Image.Image, is_hr_image: bool = False) -> Image.Image:
    """预处理输入图像（适配模型输入要求）"""
    # 转换为RGB通道
    img = image.convert("RGB")

    # 若为HR图像，下采样到256x256作为输入；若为LR图像，确保尺寸为256x256
    target_size = (256, 256)
    if img.size != target_size:
        img = transforms.Resize(target_size, interpolation=Image.BICUBIC)(img)
    return img


@app.get("/health", tags=["系统状态"])
def health_check():
    """检查服务状态及模型加载情况"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "model_loaded": model is not None,
        "upscale_factor": upscale_factor,
        "model_path": model_load_path
    }


@app.post("/super-resolution", tags=["超分辨率处理"])
async def super_resolution(
        file: UploadFile = File(..., description="输入低分辨率图像（建议256x256，PNG/JPG）"),
        return_metrics: bool = Query(False, description="是否返回SSIM（需提供原始高分辨率图像）"),
        original_hr_file: UploadFile = File(None, description="原始高分辨率图像（1024x1024，计算SSIM时必填）")
):
    try:
        # 读取输入图像
        input_img = Image.open(io.BytesIO(await file.read()))
        # 预处理输入图像（转为256x256 RGB）
        lr_img = prepare_image(input_img, is_hr_image=False)

        # 执行超分辨率推理
        with torch.no_grad():
            # 转换为张量并添加批次维度
            lr_tensor = to_tensor(lr_img).unsqueeze(0).to(DEVICE)
            # 模型推理
            sr_tensor = model(lr_tensor)
            # 转换为PIL图像
            sr_img = transforms.ToPILImage()(sr_tensor.squeeze(0).cpu())

        # 计算SSIM（若需要）
        ssim_score = None
        if return_metrics:
            if not original_hr_file:
                raise HTTPException(status_code=400, detail="计算SSIM需提供原始高分辨率图像")

            # 读取原始HR图像并预处理
            hr_img = Image.open(io.BytesIO(await original_hr_file.read())).convert("RGB")
            # 确保HR图像尺寸为1024x1024（与超分输出匹配）
            hr_img = transforms.Resize((1024, 1024), interpolation=Image.BICUBIC)(hr_img)
            # 转换为张量
            hr_tensor = to_tensor(hr_img).unsqueeze(0).to(DEVICE)
            sr_tensor_for_metric = to_tensor(sr_img).unsqueeze(0).to(DEVICE)
            # 计算SSIM（假设ssim函数输入为(N,C,H,W)张量）
            ssim_score = ssim(hr_tensor, sr_tensor_for_metric).item()

        # 准备响应（返回超分图像，SSIM放在响应头）
        img_buffer = io.BytesIO()
        sr_img.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        # 响应头（包含SSIM）
        headers = {}
        if ssim_score is not None:
            headers["X-SSIM"] = f"{ssim_score:.4f}"

        return StreamingResponse(
            img_buffer,
            media_type="image/png",
            headers=headers
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败：{str(e)}")