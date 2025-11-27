import cv2
import numpy as np
import io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse

# 导入盲水印核心库
from blind_watermark.blind_watermark import WaterMark

app = FastAPI(title="Blind Watermark Service")

# 固定长度配置
MAX_WATERMARK_LENGTH = 256  # 最大支持 256 字符
FIXED_WM_BIT_LENGTH = 2047  # 固定的 wm_bit 长度（通过 calculate_fixed_length.py 计算得出）

@app.get("/", summary="Health Check")
def read_root():
    """Health check endpoint to confirm the service is running."""
    return {
        "status": "ok", 
        "message": "Blind Watermark service is running.",
        "config": {
            "max_watermark_length": MAX_WATERMARK_LENGTH,
            "fixed_wm_bit_length": FIXED_WM_BIT_LENGTH
        }
    }

@app.post("/embed", summary="Embed Watermark")
async def embed_watermark(
    password_img: int = Form(..., description="Password for embedding location (integer)"),
    password_wm: int = Form(..., description="Password for watermark encryption (integer)"),
    wm_content: str = Form(..., description="Watermark text content (max 256 characters)"),
    file: UploadFile = File(..., description="Original image file")
):
    """
    Embeds a text watermark into an image using fixed-length padding.
    Watermark is padded to MAX_WATERMARK_LENGTH, so extraction doesn't need wm_bit_length parameter.
    """
    try:
        # 1. 验证水印长度
        if len(wm_content) > MAX_WATERMARK_LENGTH:
            raise HTTPException(
                status_code=400, 
                detail=f"Watermark content exceeds maximum length of {MAX_WATERMARK_LENGTH} characters"
            )
        
        # 2. 填充水印到固定长度
        wm_content_padded = wm_content.ljust(MAX_WATERMARK_LENGTH, '\0')
        
        # 3. 读取上传的图片
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        ori_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if ori_img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        # 4. 初始化水印工具
        bwm = WaterMark(password_wm=password_wm, password_img=password_img)
        bwm.read_img(img=ori_img)
        bwm.read_wm(wm_content_padded, mode='str')
        
        # 5. 记录实际的 wm_bit_length（用于日志）
        actual_wm_bit_length = len(bwm.wm_bit)
        
        # 6. 执行嵌入
        embed_img = bwm.embed()

        # 7. 将处理后的图片编码为JPEG格式并返回
        _, encoded_img = cv2.imencode(".jpg", embed_img)
        
        # 8. 返回图片（不再需要在响应头中返回 wm_bit_length）
        return StreamingResponse(
            io.BytesIO(encoded_img.tobytes()), 
            media_type="image/jpeg",
            headers={
                "X-Original-Length": str(len(wm_content)),  # 原始长度（用于调试）
                "X-Max-Length": str(MAX_WATERMARK_LENGTH)
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during embedding: {str(e)}")


@app.post("/extract", summary="Extract Watermark")
async def extract_watermark(
    password_img: int = Form(..., description="Password for embedding location (integer)"),
    password_wm: int = Form(..., description="Password for watermark encryption (integer)"),
    file: UploadFile = File(..., description="Watermarked image file")
):
    """
    Extracts a text watermark from an image using fixed-length extraction.
    No need to provide wm_bit_length - it uses the fixed FIXED_WM_BIT_LENGTH.
    """
    try:
        # 1. 读取上传的图片
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        embed_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if embed_img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        # 2. 初始化水印工具
        bwm = WaterMark(password_wm=password_wm, password_img=password_img)
        
        # 3. 执行提取（使用固定的 FIXED_WM_BIT_LENGTH）
        wm_extract = bwm.extract(embed_img=embed_img, wm_shape=FIXED_WM_BIT_LENGTH, mode='str')
        
        # 4. 去除填充的空字符
        wm_extract = wm_extract.rstrip('\0')
        
        return {"watermark": wm_extract}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during extraction: {str(e)}")
