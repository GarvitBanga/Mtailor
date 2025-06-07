from PIL import Image
try:
    from model import ImgPreProcessing,ONNX
except Exception as e:
    print("Model loading failed",e)

if __name__ == "__main__":
    try:
        preprocessing=ImgPreProcessing()
        print("Preprocessing loaded successfully")
    except Exception as e:
        print("Preprocessing loading failed",e)

    test_images=["n01440764_tench.jpeg","n01667114_mud_turtle.JPEG"]
    labels=[0,35]

    print("Testing Preprocessing")
    for path in test_images:
        img=Image.open(path)
        img=preprocessing.preprocess(img)
        assert img.shape==(1,3,224,224),"Preprocessing failed for "+path
        print("Preprocessing passed for",path)

    print("Testing ONNX")
    try:
        onnx=ONNX("resnet.onnx")
        print("ONNX loaded successfully")
    except Exception as e:
        print("ONNX loading failed",e)

    print("Testing ONNX with sample images")
    for i,path in enumerate(test_images):
        img=Image.open(path)
        img=preprocessing.preprocess(img)
        result=onnx.predict(img)
        assert result==labels[i],"ONNX prediction failed for "+path
        print("ONNX prediction passed for",path)


