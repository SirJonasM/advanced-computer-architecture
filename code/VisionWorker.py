import torch
from torchvision import transforms, models
from torchvision.models.quantization import MobileNet_V2_QuantizedWeights

object_model = None
preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

def load_object_recognition_model():
    global object_model
    object_model = models.quantization.mobilenet_v2(
        weights=MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1,
        quantize=True
    )
    object_model.eval()
    print("Model initialized")

def recognition_worker(frame_rgb):
    global object_model
    print("Vision worker called", flush=True)
    with torch.no_grad():
        inp  = preprocess(frame_rgb).unsqueeze(0)
        out  = object_model(inp)[0]
        prob = out.softmax(dim=0)
        top_prob, top_idx = torch.max(prob, dim=0)
        return int(top_idx), float(top_prob)