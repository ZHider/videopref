import torch

from visualpref.backbone import load_backbone
from visualpref.config import DEFAULT_BACKBONE_DIR, DEFAULT_FEATURE_DIM
from visualpref.features import extract_frame_features
from visualpref.model import PreferenceModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loading backbone from", DEFAULT_BACKBONE_DIR, "on", device)
    model, processor, feature_dim = load_backbone(DEFAULT_BACKBONE_DIR, device=device)
    assert feature_dim == DEFAULT_FEATURE_DIM, (feature_dim, DEFAULT_FEATURE_DIM)
    print("feature_dim =", feature_dim, "OK")

    # 造一批随机帧 -> 提取 [CLS]
    import tempfile
    from pathlib import Path

    import numpy as np
    from PIL import Image
    tmp = Path(tempfile.mkdtemp())
    paths = []
    for i in range(4):
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        p = tmp / f"{i}.jpg"
        Image.fromarray(arr).save(p)
        paths.append(p)

    feats = extract_frame_features(model, processor, paths, device, batch_size=2)
    print("frame features shape =", tuple(feats.shape))  # expect [4, 768]
    # 池化 + 头
    vpm = PreferenceModel(feature_dim=feature_dim).to(device)
    prob = vpm(feats.to(device).unsqueeze(0), mask=None)
    print("model prob =", prob.detach().cpu().numpy().tolist())
    print("SMOKE OK")

if __name__ == "__main__":
    main()
