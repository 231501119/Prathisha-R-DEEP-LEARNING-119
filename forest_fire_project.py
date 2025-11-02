#!/usr/bin/env python3
"""
Forest Fire Detection + Heatmap generator (single-file project)

This script trains a binary image classifier (fire / no_fire) using transfer learning (ResNet18)
and provides utilities to:
  - train and save a model
  - evaluate and print Accuracy / Precision / Recall / F1 and Confusion Matrix
  - generate a geographic heatmap (HTML) for a chosen bounding-box using NASA FIRMS active-fire CSV
  - create Grad-CAM visualization for model explainability

HOW TO USE: see the README section at the bottom or run `python forest_fire_project.py -h`

Author: ChatGPT
"""

import os
import argparse
import io
import math
import time
import json
import requests
from typing import Tuple, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# optional visualization libs
import matplotlib.pyplot as plt
from matplotlib import cm

# folium for heatmap (generates an HTML file you can open in your browser)
try:
    import folium
    from folium.plugins import HeatMap
except Exception:
    folium = None


# ----------------------------
# Configuration defaults
# ----------------------------
DEFAULT_IMG_SIZE = 224
DEFAULT_BATCH = 24
DEFAULT_EPOCHS = 8
DEFAULT_LR = 1e-4
MODEL_SAVE_DIR = "models"


# ----------------------------
# Utility: prepare dataloaders
# ----------------------------

def prepare_dataloaders(data_root: str,
                        img_size: int = DEFAULT_IMG_SIZE,
                        batch_size: int = DEFAULT_BATCH,
                        val_split: float = 0.2,
                        test_split: float = 0.0,
                        seed: int = 42):
    """
    Expects `data_root` to use ImageFolder layout, e.g.:
      data_root/fire/xxx.jpg
      data_root/no_fire/yyy.jpg

    If the dataset already has train/val folders, place `data_root/train/` etc and the script will detect it.
    Returns (train_loader, val_loader, test_loader, class_names)
    test_loader may be None if not created.
    """
    # transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # if dataset already split into train/val
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    test_dir = os.path.join(data_root, 'test')

    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
        val_ds = datasets.ImageFolder(val_dir, transform=eval_transform)
        test_ds = None
        if os.path.isdir(test_dir):
            test_ds = datasets.ImageFolder(test_dir, transform=eval_transform)
        class_names = train_ds.classes
    else:
        # single folder with classes as subfolders
        full_ds = datasets.ImageFolder(data_root, transform=eval_transform)
        class_names = full_ds.classes
        n = len(full_ds)
        if n < 2:
            raise RuntimeError(f"Not enough images found in {data_root}. Expect subfolders for each class.")
        # compute split sizes
        val_size = int(math.floor(val_split * n))
        test_size = int(math.floor(test_split * n))
        train_size = n - val_size - test_size
        generator = torch.Generator().manual_seed(seed)
        train_idx, val_idx = torch.utils.data.random_split(list(range(n)), [train_size, val_size + test_size], generator=generator)
        # if test split requested, split val_idx further
        if test_size > 0:
            val_idx, test_idx = torch.utils.data.random_split(val_idx, [val_size, test_size], generator=generator)
            test_ds = Subset(full_ds, test_idx)
        else:
            # val_idx is the validation set
            test_ds = None
        train_ds = Subset(full_ds, train_idx)
        val_ds = Subset(full_ds, val_idx)
        # overwrite transforms for train subset
        train_ds.dataset.transform = train_transform

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = None
    if 'test_ds' in locals() and test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, class_names


# ----------------------------
# Build model (ResNet18 transfer learning)
# ----------------------------

def build_model(num_classes: int, feature_extract: bool = False):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


# ----------------------------
# Training loop
# ----------------------------

def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          device: torch.device,
          epochs: int = DEFAULT_EPOCHS,
          lr: float = DEFAULT_LR,
          save_dir: str = MODEL_SAVE_DIR):

    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_f1 = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        t0 = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

        epoch_loss = running_loss / (len(train_loader.dataset) if hasattr(train_loader.dataset, '__len__') else len(all_labels))
        epoch_acc = accuracy_score(all_labels, all_preds)

        # validation
        model.eval()
        val_loss = 0.0
        v_preds = []
        v_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                v_preds.extend(preds.detach().cpu().numpy().tolist())
                v_labels.extend(labels.detach().cpu().numpy().tolist())

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(v_labels, v_preds)
        val_precision = precision_score(v_labels, v_preds, zero_division=0)
        val_recall = recall_score(v_labels, v_preds, zero_division=0)
        val_f1 = f1_score(v_labels, v_preds, zero_division=0)

        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)

        t1 = time.time()
        print(f"Epoch {epoch}/{epochs} — train_loss: {epoch_loss:.4f} — val_loss: {val_loss:.4f} — val_acc: {val_acc:.4f} — val_precision: {val_precision:.4f} — val_recall: {val_recall:.4f} — val_f1: {val_f1:.4f} — time: {t1-t0:.1f}s")

        # save best
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = os.path.join(save_dir, 'best_resnet18_fire.pth')
            torch.save({'model_state': model.state_dict(), 'class_names': getattr(train_loader.dataset.dataset if isinstance(train_loader.dataset, Subset) else train_loader.dataset, 'classes', None)}, save_path)
            print(f"Saved best model -> {save_path}")

    # save final history
    hist_path = os.path.join(save_dir, 'train_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. History saved to {hist_path}")


# ----------------------------
# Evaluation utility
# ----------------------------

def evaluate_model(model_path: str, data_root: str, device: torch.device, batch_size: int = DEFAULT_BATCH):
    checkpoint = torch.load(model_path, map_location=device)
    # build model from checkpoint info
    # assume ResNet18 architecture
    # attempt to recover classes if saved
    class_names = checkpoint.get('class_names', None)
    # create a model with 2 classes if unknown
    num_classes = len(class_names) if class_names else 2
    model = build_model(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()

    # prepare dataloader (assumes data_root contains validation/test as 'val' or 'test', try 'val' then 'test')
    eval_dir = None
    if os.path.isdir(os.path.join(data_root, 'val')):
        eval_dir = os.path.join(data_root, 'val')
    elif os.path.isdir(os.path.join(data_root, 'test')):
        eval_dir = os.path.join(data_root, 'test')
    else:
        # assume data_root is single folder with classes
        eval_dir = data_root

    transform = transforms.Compose([
        transforms.Resize(int(DEFAULT_IMG_SIZE * 1.14)),
        transforms.CenterCrop(DEFAULT_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    ds = datasets.ImageFolder(eval_dir, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print('\nEVALUATION METRICS')
    print('------------------')
    print(f'Accuracy: {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall: {rec:.4f}')
    print(f'F1-score: {f1:.4f}')
    print('\nConfusion Matrix:')
    print(cm)
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=ds.classes, zero_division=0))


# ----------------------------
# Grad-CAM Implementation (basic)
# ----------------------------

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: str = None):
        """
        model: a torch model
        target_layer: dot-separated attribute name of the conv layer to target (e.g. 'layer4.1.conv2')
                      If None, GradCAM will try to find the last conv layer.
        """
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.activations = None
        self.gradients = None

        if target_layer is None:
            # try to auto-detect a conv layer from ResNet-like models
            self.target_module = self._find_last_conv(self.model)
        else:
            self.target_module = self._get_module_by_name(self.model, target_layer)

        if self.target_module is None:
            raise RuntimeError('Could not find a target conv layer for GradCAM')

        # register hooks
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple
            self.gradients = grad_out[0].detach()

        self.target_module.register_forward_hook(forward_hook)
        self.target_module.register_backward_hook(backward_hook)

    def _get_module_by_name(self, model, name: str):
        parts = name.split('.')
        mod = model
        try:
            for p in parts:
                if p.isdigit():
                    mod = mod[int(p)]
                else:
                    mod = getattr(mod, p)
            return mod
        except Exception:
            return None

    def _find_last_conv(self, model):
        # depth-first search for last nn.Conv2d
        target = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                target = module
        return target

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        input_tensor: 1xCxHxW tensor on same device as model
        class_idx: target class. If None, use predicted class
        Returns heatmap (HxW numpy array normalized 0..1)
        """
        self.model.zero_grad()
        outputs = self.model(input_tensor)
        if class_idx is None:
            class_idx = outputs.argmax(dim=1).item()
        score = outputs[0, class_idx]
        score.backward(retain_graph=True)

        activations = self.activations  # shape [1, C, H, W]
        gradients = self.gradients  # shape [1, C, H, W]
        # global-average-pool gradients -> weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1,1,H,W]
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=(input_tensor.size(2), input_tensor.size(3)), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        # normalize
        cam -= cam.min()
        if cam.max() != 0:
            cam /= cam.max()
        return cam


def save_gradcam_overlay(img_path: str, cam: np.ndarray, out_path: str, alpha: float = 0.5):
    """Save a Grad-CAM overlay image. cam is HxW in 0..1"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((cam.shape[1], cam.shape[0]))
    img_arr = np.array(img).astype(float) / 255.0

    cmap = cm.get_cmap('jet')
    heatmap = cmap(cam)[:, :, :3]  # drop alpha
    overlay = (1 - alpha) * img_arr + alpha * heatmap
    overlay = (overlay * 255).astype(np.uint8)
    out_img = Image.fromarray(overlay)
    out_img.save(out_path)


# ----------------------------
# Heatmap generation from NASA FIRMS (active-fire CSV)
# ----------------------------

def download_firms_csv(source: str = 'viirs', days: int = 7) -> Tuple[bool, str]:
    """
    Attempt to download a global FIRMS CSV for VIIRS or MODIS (7d/24h variants). Returns (success, local_path)

    Note: FIRMS offers several flavours. For the prototype we use a global 7-day VIIRS product URL.
    If download fails, user is asked to manually download from the FIRMS website.
    """
    urls = {
        # known public endpoints (may change). If one fails, the script will report a helpful message.
        'viirs_7d': 'https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_Global_7d.csv',
        'viirs_24h': 'https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_Global_24h.csv',
        'modis_7d': 'https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6/global/7d.csv'
    }
    key = 'viirs_7d' if source == 'viirs' and days >= 7 else ('viirs_24h' if source == 'viirs' else 'modis_7d')
    url = urls.get(key)
    if url is None:
        return False, ''
    print(f"Downloading FIRMS active-fire CSV from: {url}")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        local = os.path.join('data', 'firms_active_fire.csv')
        os.makedirs('data', exist_ok=True)
        with open(local, 'wb') as f:
            f.write(r.content)
        return True, local
    except Exception as e:
        print('Failed to download FIRMS CSV:', e)
        return False, ''


def create_heatmap_from_csv(csv_path: str, bbox: Tuple[float, float, float, float], out_html: str = 'firms_heatmap.html', min_confidence: float = 0.0):
    """
    csv_path: path to FIRMS CSV (contains lat, lon, brightness, scan, track, acq_date, acq_time, confidence, version, type)
    bbox: (lat_min, lat_max, lon_min, lon_max)
    out_html: output folium html
    """
    if folium is None:
        raise RuntimeError('folium is not installed. Please pip install folium to enable heatmap generation')

    df = pd.read_csv(csv_path)
    # ensure column names exist
    if not set(['latitude', 'longitude']).issubset(df.columns):
        # try alternatives
        if 'lat' in df.columns and 'lon' in df.columns:
            df = df.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
        else:
            raise RuntimeError('CSV does not contain latitude/longitude columns')

    lat_min, lat_max, lon_min, lon_max = bbox
    region_df = df[(df.latitude >= lat_min) & (df.latitude <= lat_max) & (df.longitude >= lon_min) & (df.longitude <= lon_max)]
    if region_df.empty:
        print('No FIRMS active-fire points found inside the given bounding box for the selected CSV/time window.')

    # prepare points: [lat, lon, weight]; weight we set to 'brightness' if exists, otherwise 1
    weights = region_df['brightness'] if 'brightness' in region_df.columns else pd.Series(1, index=region_df.index)
    points = region_df[['latitude', 'longitude']].values.tolist()
    points_weighted = [[r[0], r[1], float(w)] for r, w in zip(points, weights)]

    # center map
    lat_center = float((lat_min + lat_max) / 2.0)
    lon_center = float((lon_min + lon_max) / 2.0)

    m = folium.Map(location=[lat_center, lon_center], zoom_start=7)
    if len(points_weighted) > 0:
        HeatMap(points_weighted, radius=12, blur=15, max_zoom=12).add_to(m)
    else:
        # add a marker indicating no points
        folium.Marker(location=[lat_center, lon_center], popup='No FIRMS points in bbox').add_to(m)
    m.save(out_html)
    print(f'Heatmap written to {out_html} — open this file in your browser to view it')


# ----------------------------
# Helper: single-image prediction and gradcam wrapper
# ----------------------------

def predict_image(model_path: str, image_path: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get('class_names', None)
    num_classes = len(class_names) if class_names else 2
    model = build_model(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(int(DEFAULT_IMG_SIZE * 1.14)),
        transforms.CenterCrop(DEFAULT_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    input_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_t)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        pred = int(torch.argmax(probs).item())
    labels = class_names if class_names else ['class_0', 'class_1']
    return labels[pred], float(probs[pred].item()), model, input_t


# ----------------------------
# CLI / main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description='Forest Fire Detection – train / eval / heatmap / gradcam')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_train = sub.add_parser('train', help='Train a model')
    p_train.add_argument('--data_dir', required=True, help='Path to image dataset root (ImageFolder layout)')
    p_train.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    p_train.add_argument('--batch', type=int, default=DEFAULT_BATCH)
    p_train.add_argument('--lr', type=float, default=DEFAULT_LR)
    p_train.add_argument('--feature_extract', action='store_true', help='Freeze backbone weights and only train head')

    p_eval = sub.add_parser('eval', help='Evaluate saved model on validation/test set')
    p_eval.add_argument('--model_path', required=True)
    p_eval.add_argument('--data_dir', required=True)
    p_eval.add_argument('--batch', type=int, default=DEFAULT_BATCH)

    p_heat = sub.add_parser('heatmap', help='Download FIRMS CSV and build a geographic heatmap for a bbox')
    p_heat.add_argument('--source', choices=['viirs', 'modis'], default='viirs')
    p_heat.add_argument('--days', type=int, default=7)
    p_heat.add_argument('--lat_min', type=float, required=True)
    p_heat.add_argument('--lat_max', type=float, required=True)
    p_heat.add_argument('--lon_min', type=float, required=True)
    p_heat.add_argument('--lon_max', type=float, required=True)
    p_heat.add_argument('--out', default='firms_heatmap.html')

    p_pred = sub.add_parser('predict', help='Predict single image and optionally save gradcam overlay')
    p_pred.add_argument('--model_path', required=True)
    p_pred.add_argument('--image', required=True)
    p_pred.add_argument('--gradcam_out', default=None)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    if args.cmd == 'train':
        print('Preparing data...')
        train_loader, val_loader, test_loader, class_names = prepare_dataloaders(args.data_dir, img_size=DEFAULT_IMG_SIZE, batch_size=args.batch)
        print('Classes:', class_names)
        model = build_model(num_classes=len(class_names), feature_extract=args.feature_extract)
        # attach class_names to saving behavior by monkey-patching dataset classes into loader dataset
        # train the model
        train(model=model, train_loader=train_loader, val_loader=val_loader, device=device, epochs=args.epochs, lr=args.lr)

    elif args.cmd == 'eval':
        evaluate_model(model_path=args.model_path, data_root=args.data_dir, device=device, batch_size=args.batch)

    elif args.cmd == 'heatmap':
        success, csv_path = download_firms_csv(source=args.source, days=args.days)
        if not success:
            print('Automatic FIRMS download failed. Please manually download an active-fire CSV from NASA FIRMS and pass its path to create_heatmap_from_csv')
            return
        bbox = (args.lat_min, args.lat_max, args.lon_min, args.lon_max)
        create_heatmap_from_csv(csv_path, bbox=bbox, out_html=args.out)

    elif args.cmd == 'predict':
        label, prob, model, input_t = predict_image(args.model_path, args.image, device)
        print(f'Prediction: {label} — probability {prob:.4f}')
        if args.gradcam_out:
            # create gradcam
            # find a good default target layer for ResNet18
            cam = GradCAM(model=model, target_layer='layer4.1.conv2')
            heat = cam.generate(input_t)
            save_gradcam_overlay(args.image, heat, args.gradcam_out)
            print('Saved Grad-CAM overlay to', args.gradcam_out)


if __name__ == '__main__':
    main()

# ----------------------------
# End of file
# ----------------------------
