import argparse
import os
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms, models


def detect_images(input_dir: str, output_dir: str, threshold: float = 0.5) -> None:
    """Run object detection on images and save annotated copies."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval()
    model.to(device)
    preprocess = transforms.Compose([transforms.ToTensor()])
    label_map = {17: "cat", 18: "dog"}

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        path = os.path.join(input_dir, fname)
        image = Image.open(path).convert("RGB")
        input_tensor = preprocess(image).to(device)
        with torch.no_grad():
            outputs = model([input_tensor])[0]
        draw = ImageDraw.Draw(image)
        for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
            if score < threshold or label.item() not in label_map:
                continue
            box = box.to("cpu").tolist()
            draw.rectangle(box, outline="red", width=2)
            text = f"{label_map[label.item()]}: {score:.2f}"
            draw.text((box[0], box[1]), text, fill="red")
        out_path = os.path.join(output_dir, fname)
        image.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect cats and dogs in images")
    parser.add_argument("input", help="Folder of images to process")
    parser.add_argument("output", help="Folder to save annotated images")
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold")
    args = parser.parse_args()
    detect_images(args.input, args.output, args.threshold)


if __name__ == "__main__":
    main()
