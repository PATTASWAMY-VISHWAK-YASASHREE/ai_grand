import torch
from torch.utils.data import DataLoader
import time
import copy
import wandb

def train_model(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs=25):
    """
    Training function for document layout detection model
    """
    # Initialize wandb for experiment tracking
    wandb.init(project="ps05-document-layout")
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_map = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader
            
            running_loss = 0.0
            
            # Iterate over data
            for images, targets in data_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                # Track history if only in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    # FasterRCNN returns dict of losses during training
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        losses.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += losses.item() * len(images)
            
            epoch_loss = running_loss / len(data_loader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f}')
            
            # Log metrics to wandb
            wandb.log({f"{phase}_loss": epoch_loss})
            
            # Evaluate on validation set
            if phase == 'val':
                # Calculate mAP
                map_score = evaluate_model(model, val_loader, device)
                print(f'Validation mAP@0.5: {map_score:.4f}')
                wandb.log({"val_map": map_score})
                
                # If best model, save weights
                if map_score > best_map:
                    best_map = map_score
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), f"best_model_epoch_{epoch}.pth")
        
        # Update learning rate
        if scheduler:
            scheduler.step()
    
    print(f'Best val mAP: {best_map:4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def evaluate_model(model, data_loader, device):
    """
    Evaluate model on validation set
    """
    model.eval()
    predictions = []
    image_ids = []
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images = list(img.to(device) for img in images)
            
            outputs = model(images)
            
            # Process each output in the batch
            for j, output in enumerate(outputs):
                # Add to predictions list
                predictions.append(output)
                # Use index as image_id for simplicity
                image_ids.append(i * data_loader.batch_size + j)
    
    # Calculate mAP using evaluation function
    map_score = evaluate_map(predictions, "validation_annotations.json", image_ids)
    
    return map_score