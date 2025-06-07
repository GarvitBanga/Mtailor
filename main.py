from fastapi import FastAPI,File,UploadFile
from model import ImgPreProcessing,ONNX
from PIL import Image   
import io
import base64
from pydantic import BaseModel

class ImageRequest(BaseModel):
    image: str

app=FastAPI()

preprocessing=ImgPreProcessing()
onnx_model=ONNX("resnet.onnx")


@app.get("/")
def read_root():
    return {"message":"ImageNet Classification API"}

@app.get("/health")
def health():
    return {"status":"healthy","model":"loaded"}

@app.post("/predictlocal")
async def predictlocal(file:UploadFile=File(...)):
    try:
        image_data=await file.read()
        image=Image.open(io.BytesIO(image_data))
        processed_img=preprocessing.preprocess(image)
        prediction=onnx_model.predict(processed_img)
        return {"predicted_class":prediction}
    except Exception as e:
        return {"error":str(e)}
@app.post("/predict")
def predict(request: ImageRequest):
    try:
        image_data=base64.b64decode(request.image)
        img=Image.open(io.BytesIO(image_data))
        processed_img=preprocessing.preprocess(img)
        prediction=onnx_model.predict(processed_img)
        return {"predicted_class":prediction}
    except Exception as e:
        return {"error":str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)
