import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import os
import json
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ----------------------------
# Custom Dataset for your project
# ----------------------------
class FireDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))  # ['Fire', 'Non_Fire']
        self.images = []
        self.labels = []

        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label, img_path  # include img_path for Grad-CAM filenames

# ----------------------------
# Grad-CAM helper
# ----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        handle_f = self.target_layer.register_forward_hook(forward_hook)
        handle_b = self.target_layer.register_backward_hook(backward_hook)
        self.hook_handles.extend([handle_f, handle_b])

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        loss = output[0, class_idx]
        loss.backward(retain_graph=True)

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (x.size(2), x.size(3)))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

# ----------------------------
# Generate report
# ----------------------------
def generate_report(model_path, data_dir, out_dir, device='cpu'):
    os.makedirs(out_dir, exist_ok=True)
    gradcam_dir = os.path.join(out_dir, 'gradcam')
    os.makedirs(gradcam_dir, exist_ok=True)

    # ----------------------------
    # Data transforms & loader
    # ----------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = FireDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # ----------------------------
    # Model setup
    # ----------------------------
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(dataset.classes))

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()

    gradcam = GradCAM(model, model.layer4[-1])  # last block of layer4

    # ----------------------------
    # Evaluation
    # ----------------------------
    all_preds = []
    all_labels = []

    for idx, (images, labels, img_paths) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.append(preds.cpu().item())
        all_labels.append(labels.cpu().item())

        # Grad-CAM overlay
        cam = gradcam(images, class_idx=preds.item())
        img_np = images[0].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        img_np = np.uint8(255 * img_np)
        cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cam_heatmap, 0.5, img_np, 0.5, 0)

        orig_filename = os.path.basename(img_paths[0])
        out_path = os.path.join(gradcam_dir, f"{preds.item()}_{orig_filename}")
        cv2.imwrite(out_path, overlay)

    gradcam.remove_hooks()

    # ----------------------------
    # Metrics
    # ----------------------------
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=dataset.classes, digits=4)
    cm = confusion_matrix(all_labels, all_preds).tolist()

    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    # ----------------------------
    # Save report
    # ----------------------------
    out_file = os.path.join(out_dir, 'report.json')
    with open(out_file, 'w') as f:
        json.dump({
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm
        }, f, indent=4)

    print(f"Report and Grad-CAM images saved at: {out_dir}")

# ----------------------------
# CLI usage
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Forest Fire Report with Grad-CAM")
    parser.add_argument('--model_path', type=str, required=True, help='Path to your saved model checkpoint (.pth)')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to validation dataset')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save report and Grad-CAM')
    parser.add_argument('--device', type=str, default='cpu', help='Device: cpu or cuda')
    args = parser.parse_args()

    generate_report(args.model_path, args.data_dir, args.out_dir, args.device)
