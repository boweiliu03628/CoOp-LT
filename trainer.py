import torch
import os
import datasets
from torch.utils.data import DataLoader
from cooplt import CoOpLT
from clip import clip
import torch.nn as nn
import pickle



def load_clip_to_cpu(backbone_name):

    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    model = torch.jit.load(model_path, map_location="cpu").eval()
    return model

def dataset_map(dataset_name):
    dataset_mapping={
        'cifar10': 'CIFAR10_LT',
        'cifar100_if200': 'CIFAR100_LT',
        'cifar100_if100': 'CIFAR100_LT',
        'cifar100_if100_outer': 'CIFAR100_LT'
    }
    return dataset_mapping[dataset_name]

def backbone_map(backbone_name):
    backbone_mapping={
        'clip_vit_b16': 'ViT-B/16'
    }
    return backbone_mapping[backbone_name]



class Trainer:
    def __init__(self, cfg):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg

        self.build_dataloader()
        self.build_model()

        self.best_test_accuracy = 0.0
        self.best_epoch = 0
        self.output_dir = cfg.output_dir
        os.makedirs(cfg.output_dir, exist_ok=True)



    
    
    def build_dataloader(self):
        cfg = self.cfg
        mapped__dataset_name=dataset_map(cfg.dataset)
        train_dataset = getattr(datasets, mapped__dataset_name)(cfg.train_img_path)
        test_dataset = getattr(datasets, mapped__dataset_name)(cfg.test_img_path)
        self.train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, num_workers=cfg.num_workers,
                                       shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, num_workers=cfg.num_workers,
                                      shuffle=False)
        self.classnames = train_dataset.classnames

    


    def build_model(self):
        cfg = self.cfg
        classnames = self.classnames

        print("Building model")
        print(f"Loading CLIP (backbone: {cfg.backbone})")

        mapped__backbone_name=backbone_map(cfg.backbone)
        clip_model = load_clip_to_cpu(mapped__backbone_name)
        self.model = CoOpLT(cfg, classnames, clip_model)
        self.model.to(self.device)
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, cfg.num_epochs)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_accuracy = correct / total
        return epoch_loss, epoch_accuracy

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy

    def train_and_evaluate(self):
        cfg = self.cfg
        for epoch in range(cfg.num_epochs):
            train_loss, train_accuracy = self.train()
            test_accuracy = self.test()
            print(f'Epoch {epoch + 1}/{cfg.num_epochs}, '
                  f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
                  f'Test Accuracy: {test_accuracy:.4f}')
            if test_accuracy > self.best_test_accuracy:
                self.best_test_accuracy = test_accuracy
                self.best_epoch = epoch + 1
                torch.save(self.model.state_dict(), os.path.join(cfg.output_dir, 'best_model.pth'))
            self.scheduler.step()
        print(f"Best Test Accuracy: {self.best_test_accuracy:.4f} at Epoch {self.best_epoch}")
        self.test_per_class()

    def test_per_class(self):
        cfg = self.cfg
        checkpoint = torch.load(os.path.join(cfg.output_dir, 'best_model.pth'), map_location=self.device,weights_only=True)
        self.model.load_state_dict(checkpoint)
        classnames = self.classnames
        class_correct = [0] * len(classnames)
        class_total = [0] * len(classnames)
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_correct[label] += (predicted[i] == labels[i]).item()
                    class_total[label] += 1
        class_accuracies = [class_correct[i] / class_total[i] for i in range(len(classnames))]
        for i, classname in enumerate(classnames):
            print(f'Accuracy of {classname}: {class_accuracies[i]:.4f}')
        print(class_accuracies)
        with open(f'./output/class_acc/{cfg.dataset}_class_acc.pkl', 'wb') as f:
            pickle.dump(class_accuracies, f)
        accuracy = sum(class_accuracies) / len(class_accuracies)
        print(f"average accuracy is: {accuracy:.4f}")


    def test_only(self):
        self.test_per_class()



