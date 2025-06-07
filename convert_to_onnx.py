from pytorch_model import Classifier, BasicBlock
import torch
import torch.onnx as onnx

def convert_to_onnx():
    resnet = Classifier(BasicBlock, [2, 2, 2, 2],num_classes=1000)
    resnet.load_state_dict(torch.load("pytorch_model_weights.pth"))
    resnet.eval()
    dummy_input = torch.randn(1,3,224,224)
    onnx.export(resnet, dummy_input, "resnet.onnx", export_params=True, opset_version=11, input_names=['input'], output_names=['output'])

if __name__ == "__main__":
    convert_to_onnx()