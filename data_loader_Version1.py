import os
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from albumentations import Compose, RandomResizedCrop, HorizontalFlip, Normalize

class DocumentLayoutDataset(Dataset):
    def __init__(self, json_dir, image_dir, transform=None):
        self.json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith('.json')]
        self.image_dir = image_dir
        self.transform = transform
        self.samples = self._load_samples()
        
    def _load_samples(self):
        samples = []
        for json_file in self.json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            image_path = os.path.join(self.image_dir, data['file_name'])
            if not os.path.exists(image_path):
                continue
                
            boxes = []
            labels = []
            
            for ann in data['annotations']:
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x+w, y+h])  # Convert to [x1, y1, x2, y2]
                labels.append(ann['category_id'])
                
            if len(boxes) > 0:
                samples.append({
                    'image_path': image_path,
                    'boxes': np.array(boxes, dtype=np.float32),
                    'labels': np.array(labels, dtype=np.int64)
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes = sample['boxes'].copy()
        labels = sample['labels'].copy()
        
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                labels=labels
            )
            
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
        
        # Convert to tensors
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }
        
        return image, target

# Example usage
def get_data_loaders(json_dir, image_dir, batch_size=2):
    train_transform = Compose([
        RandomResizedCrop(height=800, width=800, scale=(0.8, 1.0)),
        HorizontalFlip(p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    
    val_transform = Compose([
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    
    train_dataset = DocumentLayoutDataset(json_dir, image_dir, transform=train_transform)
    val_dataset = DocumentLayoutDataset(json_dir, image_dir, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

# Custom collate function for batching
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)