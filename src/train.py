import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

def train_model(model, train_loader, test_loader, epochs=10, class_weights=None):
    device = torch.device("cpu")  # CPU only
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)  #
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    test_f1s = []
    
    for epoch in range(epochs):
        model.train() #Set model to training model
        running_loss = 0
        train_preds = []
        train_labels_all = []
        
        for images, labels in train_loader:
            images = images.to(device) # Move data to device(here CPU)
            labels = labels.to(device)
            
            optimizer.zero_grad() #Reset gradients
            outputs = model(images) # Forward pass
            loss = criterion(outputs, labels)
            loss.backward()  #Backpropagation
            optimizer.step()
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.numpy())
            train_labels_all.extend(labels.numpy())

        train_loss = running_loss / len(train_loader)
        train_accuracy = accuracy_score(train_labels_all, train_preds)

        train_losses.append(train_loss)
        train_accs.append(train_accuracy)

        
        # Evaluation
        model.eval()  
        running_test_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():   #Disable gradient computation
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_test_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())
        
        test_loss = running_test_loss / len(test_loader)
        test_accuracy = accuracy_score(all_labels, all_preds)
        test_f1 = f1_score(all_labels, all_preds, average="macro")

        test_losses.append(test_loss)
        test_accs.append(test_accuracy)
        test_f1s.append(test_f1)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        print(f"Train Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f}")
        print(f"Macro F1: {test_f1:.4f}")
        print("="*40)

    return model, train_losses, test_losses, train_accs, test_accs, test_f1s