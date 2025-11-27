
import cv2
import numpy as np
import io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse

# 导入盲水印核心库
from blind_watermark.blind_watermark import WaterMark

app = FastAPI(title="Blind Watermark Service")

@app.get("/", summary="Health Check")
def read_root():
    """Health check endpoint to confirm the service is running."""
    return {"status": "ok", "message": "Blind Watermark service is running."}

@app.post("/embed", summary="Embed Watermark")
async def embed_watermark(
    password_img: int = Form(..., description="Password for embedding location (integer)"),
    password_wm: int = Form(..., description="Password for watermark encryption (integer)"),
    wm_content: str = Form(..., description="Watermark text content"),
    file: UploadFile = File(..., description="Original image file")
):
    """
    Embeds a text watermark into an image.
    Receives image, passwords, and watermark text, returns the watermarked image.
    """
    try:
        # 1. 读取上传的图片
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        ori_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if ori_img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        # 2. 初始化水印工具
        bwm = WaterMark(password_wm=password_wm, password_img=password_img)
        bwm.read_img(img=ori_img)
        bwm.read_wm(wm_content, mode='str')
        
        # 3. 执行嵌入
        embed_img = bwm.embed()

        # 4. 将处理后的图片编码为JPEG格式并返回
        _, encoded_img = cv2.imencode(".jpg", embed_img)
        
        return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during embedding: {str(e)}")


@app.post("/extract", summary="Extract Watermark")
async def extract_watermark(
    password_img: int = Form(..., description="Password for embedding location (integer)"),
    password_wm: int = Form(..., description="Password for watermark encryption (integer)"),
    wm_length: int = Form(..., description="Length of the original watermark text"),
    file: UploadFile = File(..., description="Watermarked image file")
):
    """
    Extracts a text watermark from an image.
    Requires the same passwords used for embedding and the original watermark length.
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
        
        # 3. 执行提取
        wm_extract = bwm.extract(embed_img=embed_img, wm_shape=wm_length, mode='str')
        
        return {"watermark": wm_extract}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during extraction: {str(e)}")
