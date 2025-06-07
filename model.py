import numpy as np


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
