
"""
FastAPI inference server for the AffectNet FER model.
Run with:  uvicorn app:app --host 0.0.0.0 --port 8000
Requires: fer_model.pth (or .onnx), label_map.json, facenet-pytorch, timm, torch

DISCLAIMER: this endpoint returns an EXPRESSION classification (sad/neutral/fear/disgust),
not a mental-health diagnosis. Surface it to end users accordingly.
"""
import io, json
import torch
import torch.nn.functional as F
import timm
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from facenet_pytorch import MTCNN
import torchvision.transforms as T

with open("label_map.json") as f:
    meta = json.load(f)
IMG_SIZE = meta["img_size"]
BACKBONE = meta["backbone"]
idx2label = {int(k): v for k, v in meta["idx2label"].items()}
display_name = meta["display_name"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FERModel(nn.Module):
    def __init__(self, backbone_name, num_classes):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.head(self.backbone(x))

model = FERModel(BACKBONE, num_classes=len(idx2label)).to(device)
model.load_state_dict(torch.load("fer_model.pth", map_location=device))
model.eval()

mtcnn = MTCNN(image_size=IMG_SIZE, margin=20, post_process=False, device=device)

val_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

app = FastAPI(title="FER Inference API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)  # tighten allow_origins to your Lovable domain in production

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    face = mtcnn(img)
    if face is None:
        return {"error": "No face detected in the image."}

    face_img = Image.fromarray(face.permute(1, 2, 0).byte().cpu().numpy())
    tensor = val_tf(face_img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        probs = F.softmax(out, dim=1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())

    return {
        "label": idx2label[pred_idx],
        "display_name": display_name[idx2label[pred_idx]],
        "confidence": float(probs[pred_idx]),
        "all_probs": {display_name[idx2label[i]]: float(p) for i, p in enumerate(probs)},
        "disclaimer": "Expression classification only — not a clinical diagnosis.",
    }
