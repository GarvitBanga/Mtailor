import numpy as np
import onnxruntime
from PIL import Image


class ImgPreProcessing:
    def __init__(self):
        self.std=np.array([0.229,0.224,0.225])
        self.mean=np.array([0.485,0.456,0.406])

    def preprocess(self,img):
        if img.mode!='RGB':
            img=img.convert('RGB')
        img=img.resize((224,224))
        img=np.array(img,dtype=np.float32) / 255.0
        img=(img-self.mean)/self.std
        img=np.transpose(img,(2,0,1))
        img=np.expand_dims(img,axis=0)
        return img

class ONNX:
    def __init__(self,path):
        self.inference_session=onnxruntime.InferenceSession(path)
        self.input_name=self.inference_session.get_inputs()[0].name

    def predict(self,img):
        img=img.astype(np.float32)
        prediction=self.inference_session.run(None,{self.input_name:img})
        return int(np.argmax(prediction[0]))

if __name__ == "__main__":
    preprocessing=ImgPreProcessing()
    onnx=ONNX("resnet.onnx")
    test_images=["n01440764_tench.jpeg","n01667114_mud_turtle.JPEG"]
    for path in test_images:
        img=Image.open(path)
        img=preprocessing.preprocess(img)
        print(onnx.predict(img))