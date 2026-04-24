"""
Micro-CT Bone Cell Classification System
Classifies human vs. non-human bone cells with visual explanations and detailed metrics.

Methodology:
- Specimen-level k-fold cross-validation to prevent data leakage
- Transfer learning with ResNet-18 pretrained on ImageNet
- Early stopping to prevent overfitting
- Grad-CAM visualization for model interpretability
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from PIL import Image as PILImage
import numpy as np
import os
import re
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import random

# ============= CONFIGURATION =============
class Config:
    # Paths — update to match local directory structure before running
    DATA_DIR = "/path/to/data"
    HUMAN_DIR = os.path.join(DATA_DIR, "Human Images Train VOIs")
    ANIMAL_DIR = os.path.join(DATA_DIR, "Non-Human Images Train VOIs")
    OUTPUT_DIR = "/path/to/results"
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "bone_classifier_model_fold{}.pth")

    # Image and training parameters
    IMG_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    RANDOM_SEED = 42

    # Cross-validation
    N_FOLDS = 5

    # Early stopping
    EARLY_STOPPING_PATIENCE = 10

    # Device selection
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============= DATASET CLASS =============
class BoneCellDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

# ============= MODEL DEFINITION =============
class BoneClassifier(nn.Module):
    """
    ResNet-18 adapted for binary classification (human vs. non-human bone).
    The final fully connected layer is replaced to output two class scores.
    """
    def __init__(self, num_classes=2):
        super(BoneClassifier, self).__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# ============= GRAD-CAM IMPLEMENTATION =============
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    Produces a coarse localization map highlighting regions used by the
    network when making a classification decision (Selvaraju et al., 2017).
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer = self.model.base_model.layer4[-1]
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class):
        model_output = self.model(input_image)
        self.model.zero_grad()
        class_loss = model_output[0, target_class]
        class_loss.backward()
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(pooled_gradients * self.activations, dim=1).squeeze()
        cam = torch.relu(cam)
        cam = cam / torch.max(cam)
        return cam.cpu().numpy()

# ============= SPECIMEN ID EXTRACTION =============
def get_specimen_id(folder_name):
    """
    Parses a VOI folder name to extract the base specimen identifier,
    removing scan-level suffixes (e.g., VOI1, _Rec, _top).
    """
    specimen_id = re.split(r'[_-](?:VOI|Rec|rec|voi)\d*[A-Za-z]*(?:top|bottom)?$', folder_name)[0]
    specimen_id = re.split(r'VOI\d+$', specimen_id)[0]
    return specimen_id.rstrip('_-')

# ============= DATA LOADING =============
def load_all_specimens():
    """
    Loads all specimen image paths from the human and non-human directories.
    Images are grouped by specimen ID to ensure specimen-level train/test splits
    and prevent data leakage across folds.
    """
    print("Loading samples...")

    def group_by_specimen(base_dir, label):
        specimens = {}
        if not os.path.exists(base_dir):
            print(f"WARNING: Directory not found: {base_dir}")
            return specimens

        for folder_name in sorted(os.listdir(base_dir)):
            folder_path = os.path.join(base_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            images = [os.path.join(folder_path, f)
                     for f in os.listdir(folder_path)
                     if f.lower().endswith('.bmp')]

            if not images:
                continue

            specimen_id = get_specimen_id(folder_name)
            unique_id = f"{'H' if label == 0 else 'A'}_{specimen_id}"

            if unique_id not in specimens:
                specimens[unique_id] = {'images': [], 'label': label, 'folders': []}

            specimens[unique_id]['images'].extend(images)
            specimens[unique_id]['folders'].append(folder_name)

        return specimens

    human_specimens = group_by_specimen(Config.HUMAN_DIR, 0)
    animal_specimens = group_by_specimen(Config.ANIMAL_DIR, 1)

    print(f"Found {len(human_specimens)} unique human samples")
    print(f"Found {len(animal_specimens)} unique non-human samples")

    all_specimens = {**human_specimens, **animal_specimens}
    total_images = sum(len(s['images']) for s in all_specimens.values())
    print(f"Total images: {total_images}")

    return all_specimens

# ============= DATA TRANSFORMS =============
def get_transforms():
    """
    Returns training and evaluation transforms.
    Training transforms include augmentation (flips, rotation, color jitter)
    to improve generalization. Test transforms apply only resizing and normalization.
    ImageNet mean and standard deviation values are used for normalization,
    consistent with ResNet-18 pretraining expectations.
    """
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform

# ============= EARLY STOPPING =============
class EarlyStopping:
    """
    Monitors validation loss and halts training when no improvement is
    observed for a given number of epochs (patience). The best-performing
    model state is saved to disk for later evaluation (Prechelt, 2012).
    """
    def __init__(self, patience=10, save_path='best_model.pth'):
        self.patience = patience
        self.save_path = save_path
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

# ============= TRAINING =============
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, fold):
    """
    Trains the model for one fold. Tracks training and validation loss per epoch.
    Early stopping is applied based on validation loss. The best model checkpoint
    is saved and reloaded for final evaluation.
    """
    print(f"\nStarting training for fold {fold+1}...")
    save_path = Config.MODEL_SAVE_PATH.format(fold+1)
    early_stopping = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE, save_path=save_path)
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in train_loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)

        improved = early_stopping(val_loss, model)
        status = " -> Model saved (improved)" if improved else f" (no improvement {early_stopping.counter}/{early_stopping.patience})"

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%{status}")

        if early_stopping.early_stop:
            print(f"  -> Early stopping triggered at epoch {epoch+1}")
            break

    return train_losses, val_losses

# ============= EVALUATION =============
def evaluate_model(model, test_loader):
    """
    Evaluates the model on the held-out test fold. Returns per-image
    predictions, class probabilities, true labels, and overall accuracy.
    """
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(Config.DEVICE)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_paths.extend(paths)

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    accuracy = accuracy_score(all_labels, all_predictions)
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'labels': all_labels,
        'paths': all_paths
    }

# ============= GRAD-CAM VISUALIZATION =============
def generate_gradcam_visualizations(model, test_paths, test_labels, test_transform, fold, num_per_class=10):
    """
    Generates Grad-CAM visualizations for a random sample of correctly
    processed images — 10 human and 10 non-human per fold. Each output
    includes the original image, the activation heatmap, and an overlay.
    Images are labeled with specimen ID and prediction confidence.
    """
    print(f"\nGenerating Grad-CAM visualizations for fold {fold+1}...")
    model.eval()
    gradcam = GradCAM(model)

    gradcam_dir = os.path.join(Config.OUTPUT_DIR, f'gradcam_fold{fold+1}')
    os.makedirs(gradcam_dir, exist_ok=True)

    class_names = ['Human', 'Non-Human']

    human_paths = [(p, l) for p, l in zip(test_paths, test_labels) if l == 0]
    animal_paths = [(p, l) for p, l in zip(test_paths, test_labels) if l == 1]

    random.seed(Config.RANDOM_SEED)
    human_samples = random.sample(human_paths, min(num_per_class, len(human_paths)))
    animal_samples = random.sample(animal_paths, min(num_per_class, len(animal_paths)))

    all_samples = human_samples + animal_samples

    for idx, (img_path, true_label) in enumerate(all_samples):
        pil_image = PILImage.open(img_path).convert('RGB')
        image_tensor = test_transform(pil_image).unsqueeze(0).to(Config.DEVICE)

        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()
        probabilities = torch.softmax(output, dim=1)[0]
        confidence = probabilities[predicted_class].item() * 100

        cam = gradcam.generate_cam(image_tensor, predicted_class)

        folder_name = os.path.basename(os.path.dirname(img_path))
        specimen_id = get_specimen_id(folder_name)
        image_name = os.path.basename(img_path)

        original_img = np.array(pil_image.resize((Config.IMG_SIZE, Config.IMG_SIZE)))
        cam_resized = np.array(PILImage.fromarray(np.uint8(255 * cam)).resize((Config.IMG_SIZE, Config.IMG_SIZE))) / 255.0
        heatmap = plt.cm.jet(cam_resized)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        overlay = (original_img * 0.6 + heatmap * 0.4).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        true_label_name = class_names[true_label]
        pred_label_name = class_names[predicted_class]
        correct = "✓" if true_label == predicted_class else "✗"

        fig.suptitle(f'Sample: {specimen_id} | Image: {image_name}\n'
                     f'True: {true_label_name} | Predicted: {pred_label_name} {correct} | Confidence: {confidence:.1f}%',
                     fontsize=10, y=1.02)

        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()

        category = "human" if true_label == 0 else "nonhuman"
        save_name = f'{category}_{specimen_id}_{idx+1}.png'
        save_name = re.sub(r'[^\w\-_.]', '_', save_name)
        plt.savefig(os.path.join(gradcam_dir, save_name), bbox_inches='tight', dpi=150)
        plt.close()

        print(f"  Saved: {save_name}")

    print(f"Saved {len(all_samples)} Grad-CAM images to {gradcam_dir}/")

# ============= VISUALIZATION =============
def visualize_fold_results(all_fold_results, all_train_losses, all_val_losses):
    """
    Produces summary figures across all folds:
      - Training and validation loss curves
      - Mean confusion matrix (averaged across folds)
      - Per-fold test accuracy bar chart
    All figures are saved to the output directory.
    """
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(12, 6))
    for i, (tl, vl) in enumerate(zip(all_train_losses, all_val_losses)):
        plt.plot(tl, label=f'Fold {i+1} Train', linestyle='--', alpha=0.7)
        plt.plot(vl, label=f'Fold {i+1} Val', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss — All Folds')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'training_curves_all_folds.png'))
    plt.close()

    mean_conf = np.mean([r['confusion_matrix'] for r in all_fold_results], axis=0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(mean_conf, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=['Human', 'Non-Human'], yticklabels=['Human', 'Non-Human'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Mean Confusion Matrix Across All Folds')
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'confusion_matrix_mean.png'))
    plt.close()

    accuracies = [r['accuracy'] * 100 for r in all_fold_results]
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(accuracies)+1), accuracies, color='steelblue')
    plt.axhline(y=np.mean(accuracies), color='red', linestyle='--', label=f'Mean: {np.mean(accuracies):.2f}%')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy per Fold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'accuracy_per_fold.png'))
    plt.close()

    print(f"\nVisualizations saved to {Config.OUTPUT_DIR}/")

# ============= MAIN EXECUTION =============
def main():
    print("=" * 60)
    print("BONE CELL CLASSIFICATION SYSTEM")
    print(f"K-Fold Cross Validation (k={Config.N_FOLDS})")
    print(f"Early Stopping (patience={Config.EARLY_STOPPING_PATIENCE})")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    all_specimens = load_all_specimens()
    specimen_ids = list(all_specimens.keys())

    if len(specimen_ids) == 0:
        print("ERROR: No samples found. Check that DATA_DIR paths are correctly configured.")
        return

    human_ids = [sid for sid in specimen_ids if all_specimens[sid]['label'] == 0]
    animal_ids = [sid for sid in specimen_ids if all_specimens[sid]['label'] == 1]

    print(f"\nHuman samples ({len(human_ids)}): {[sid.replace('H_','') for sid in human_ids]}")
    print(f"Non-human samples ({len(animal_ids)}): {[sid.replace('A_','') for sid in animal_ids]}")

    random.seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    torch.manual_seed(Config.RANDOM_SEED)

    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_SEED)

    human_ids = np.array(human_ids)
    animal_ids = np.array(animal_ids)

    human_folds = list(kf.split(human_ids))
    animal_folds = list(kf.split(animal_ids))

    train_transform, test_transform = get_transforms()

    all_fold_results = []
    all_train_losses = []
    all_val_losses = []

    for fold in range(Config.N_FOLDS):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{Config.N_FOLDS}")
        print(f"{'='*60}")

        h_train_idx, h_test_idx = human_folds[fold]
        a_train_idx, a_test_idx = animal_folds[fold]

        train_specimen_ids = list(human_ids[h_train_idx]) + list(animal_ids[a_train_idx])
        test_specimen_ids = list(human_ids[h_test_idx]) + list(animal_ids[a_test_idx])

        random.shuffle(train_specimen_ids)
        val_size = max(1, int(len(train_specimen_ids) * 0.1))
        val_specimen_ids = train_specimen_ids[:val_size]
        train_specimen_ids = train_specimen_ids[val_size:]

        print(f"Train samples: {len(train_specimen_ids)}")
        print(f"Val samples:   {len(val_specimen_ids)}")
        print(f"Test samples:  {len(test_specimen_ids)}")
        print(f"Test set: {[sid.replace('H_','').replace('A_','') for sid in test_specimen_ids]}")

        def specimens_to_images(sids):
            paths, labels = [], []
            for sid in sids:
                for img in all_specimens[sid]['images']:
                    paths.append(img)
                    labels.append(all_specimens[sid]['label'])
            return paths, labels

        train_paths, train_labels = specimens_to_images(train_specimen_ids)
        val_paths, val_labels = specimens_to_images(val_specimen_ids)
        test_paths, test_labels = specimens_to_images(test_specimen_ids)

        print(f"Train images: {len(train_paths)}")
        print(f"Val images:   {len(val_paths)}")
        print(f"Test images:  {len(test_paths)}")

        train_dataset = BoneCellDataset(train_paths, train_labels, train_transform)
        val_dataset = BoneCellDataset(val_paths, val_labels, test_transform)
        test_dataset = BoneCellDataset(test_paths, test_labels, test_transform)

        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)

        model = BoneClassifier().to(Config.DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

        train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer, Config.EPOCHS, fold
        )
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH.format(fold+1)))
        results = evaluate_model(model, test_loader)
        all_fold_results.append(results)

        print(f"\nFold {fold+1} Test Accuracy: {results['accuracy']*100:.2f}%")
        print(f"Confusion Matrix:")
        print(f"                  Human  Non-Human")
        print(f"Actual Human       {results['confusion_matrix'][0,0]:4d}   {results['confusion_matrix'][0,1]:4d}")
        print(f"Actual Non-Human   {results['confusion_matrix'][1,0]:4d}   {results['confusion_matrix'][1,1]:4d}")

        generate_gradcam_visualizations(model, test_paths, test_labels, test_transform, fold, num_per_class=10)

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")

    accuracies = [r['accuracy'] * 100 for r in all_fold_results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    print(f"\nPer-fold accuracies:")
    for i, acc in enumerate(accuracies):
        print(f"  Fold {i+1}: {acc:.2f}%")

    print(f"\nMean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Min Accuracy:  {min(accuracies):.2f}%")
    print(f"Max Accuracy:  {max(accuracies):.2f}%")

    all_labels_combined = np.concatenate([r['labels'] for r in all_fold_results])
    all_preds_combined = np.concatenate([r['predictions'] for r in all_fold_results])

    # Bootstrap confidence interval on combined predictions across all folds
    n_bootstrap = 1000
    bootstrap_accs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(all_labels_combined), len(all_labels_combined), replace=True)
        bootstrap_accs.append(accuracy_score(all_labels_combined[idx], all_preds_combined[idx]))

    ci_lower = np.percentile(bootstrap_accs, 2.5)
    ci_upper = np.percentile(bootstrap_accs, 97.5)

    print(f"95% CI (bootstrap): [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")

    visualize_fold_results(all_fold_results, all_train_losses, all_val_losses)

    with open(os.path.join(Config.OUTPUT_DIR, 'results.txt'), 'w') as f:
        f.write("BONE CELL CLASSIFICATION RESULTS\n")
        f.write(f"K-Fold Cross Validation (k={Config.N_FOLDS})\n")
        f.write(f"Early Stopping (patience={Config.EARLY_STOPPING_PATIENCE})\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%\n")
        f.write(f"Min Accuracy:  {min(accuracies):.2f}%\n")
        f.write(f"Max Accuracy:  {max(accuracies):.2f}%\n")
        f.write(f"95% CI (bootstrap): [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]\n\n")
        f.write("Per-fold accuracies:\n")
        for i, acc in enumerate(accuracies):
            f.write(f"  Fold {i+1}: {acc:.2f}%\n")
        f.write("\nPer-Image Results (all folds):\n")
        f.write("-" * 60 + "\n")
        class_names = ['Human', 'Non-Human']
        for fold_idx, results in enumerate(all_fold_results):
            f.write(f"\nFold {fold_idx+1}:\n")
            for i, path in enumerate(results['paths']):
                true_label = class_names[results['labels'][i]]
                pred_label = class_names[results['predictions'][i]]
                confidence = results['probabilities'][i][results['predictions'][i]] * 100
                folder_name = os.path.basename(os.path.dirname(path))
                specimen_id = get_specimen_id(folder_name)
                f.write(f"  Sample: {specimen_id} | File: {os.path.basename(path)}\n")
                f.write(f"    True: {true_label}, Predicted: {pred_label}, Confidence: {confidence:.2f}%\n")

    print(f"\nAll results saved to {Config.OUTPUT_DIR}/")
    print("\nProgram completed successfully!")

if __name__ == "__main__":
    main()
