# 项目简介  
# 项目简介
本项目基于 **Swift-SRGAN** 模型提供医学图像超分辨率 API 服务，可将低分辨率胸部 X 射线图像（256×256）提升至高分辨率（1024×1024），同时支持 **结构相似性指数（SSIM）** 计算，用于评估超分效果。  
项目复用了原 Streamlit 应用的核心模型架构和权重文件，提供标准化的 HTTP 接口供外部调用。

# 项目结构
Super-Resolution-for-Medical-Images-streamlit-demo/  
├── api.py                  # FastAPI接口主文件  
├── requirements.txt        # 项目依赖列表  
├── model/                  # 模型相关模块  
│   ├── model_config.py     # 模型配置（设备、路径、超参数）  
│   ├── models.py           # Generator/Discriminator模型定义  
│   ├── netG.pth.tar        # 预训练生成器权重文件  
│   └── run_inference.py    # 原推理逻辑（API中复用核心代码）  
├── scripts/                # 辅助脚本  
│   └── model_metrics.py    # 包含SSIM计算函数  

# 环境配置
## 1. 依赖安装
创建并激活虚拟环境后，安装依赖：

### 创建 conda 环境

conda create --name medical_sr python=3.9.16
conda activate medical_sr

### 安装依赖

pip install -r requirements.txt

### 启动 API 服务

在项目根目录执行以下命令启动服务：

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

--host 0.0.0.0：允许外部设备访问（同一局域网内）
--port 8000：服务端口（可修改为其他未占用端口）

启动成功后，终端显示：
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)

## 2. 接口说明

服务启动后，可通过以下方式访问接口文档：
Swagger UI：http://localhost:8000/docs（推荐，支持在线测试）

####  1. 健康检查接口

URL：/health
方法：GET
功能：检查服务状态、模型加载情况及配置信息

响应示例

```json
{
  "status": "healthy",
  "device": "cpu",  # 或"cuda"（若支持GPU）
  "model_loaded": true,
  "upscale_factor": 4,
  "model_path": "./model/netG.pth.tar"
}


```

#### 2. 超分辨率处理接口

URL：/super-resolution
方法：POST
功能：输入低分辨率图像，返回超分辨率处理结果
请求参数：
Parms:
参数名	类型	必选	说明
return_metrics	布尔值	否	是否计算 SSIM（默认 false）
Body:
file	文件	是	低分辨率图像（建议 256x256，PNG/JPG）
original_hr_file	文件	否	原始高分辨率图像（1024x1024，计算 SSIM 时必选)

## 响应说明：
成功（200 OK）：
返回超分辨率图像（PNG 格式二进制流）
若return_metrics=true，响应头包含X-SSIM字段（如X-SSIM: 0.9234）
{
  "detail": "计算SSIM需提供原始高分辨率图像"
}
