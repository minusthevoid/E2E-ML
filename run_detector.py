import argparse
import os
from PIL import Image, ImageDraw
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights


def load_model():
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights, progress=False)
    model.eval()
    return model, weights


def detect_images(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model, weights = load_model()
    preprocess = weights.transforms()
    categories = {17: "cat", 18: "dog"}

    for fname in os.listdir(input_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            path = os.path.join(input_dir, fname)
            image = Image.open(path).convert("RGB")
            input_tensor = preprocess(image)
            with torch.no_grad():
                pred = model([input_tensor])[0]

            draw = ImageDraw.Draw(image)
            for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
                label_id = label.item()
                conf = score.item()
                if label_id in categories and conf > 0.5:
                    box = box.tolist()
                    draw.rectangle(box, outline="red", width=2)
                    text = f"{categories[label_id]}: {conf*100:.1f}%"
                    draw.text((box[0] + 5, box[1] + 5), text, fill="red")

            out_path = os.path.join(output_dir, fname)
            image.save(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect cats and dogs in images")
    parser.add_argument("--input", required=True, help="Folder of images to process")
    parser.add_argument("--output", default="detections", help="Output folder for results")
    args = parser.parse_args()

    detect_images(args.input, args.output)
