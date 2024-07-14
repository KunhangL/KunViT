from typing import Callable
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import warmup_scheduler
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# From https://github.com/omihub777/ViT-CIFAR/blob/main/criterions.py
class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim)) 


class ViTTrainer:
    def __init__(
        self,
        n_epochs: int,
        device: torch.device,
        model: nn.Module,
        batch_size: int,
        eval_batch_size: int,
        checkpoints_dir: str,
        train_dataset: Dataset,
        dev_dataset: Dataset,
        test_dataset: Dataset = None,
        optimizer: torch.optim = Adam,
        lr=1e-3,
        min_lr=1e-5,
        beta1=0.9,
        beta2=0.999,
        weight_decay=5e-5,
        warmup_epoch=5,
        num_classes=10,
        smoothing=0.1 # For label smoothing
    ):
        self.n_epochs = n_epochs
        self.device = device
        self.model = model
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.checkpoints_dir = checkpoints_dir
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.optimizer = optimizer
        self.lr = lr
        self.min_lr = min_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.warmup_epoch = warmup_epoch
        self.criterion = LabelSmoothingCrossEntropyLoss(classes=num_classes, smoothing=smoothing)

    def train(self, checkpoint_epoch: int = 0, print_every: int = 100, save_every: int = 10):
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        self.model.to(self.device)

        if isinstance(self.optimizer, Callable):
            params = filter(lambda p: p.requires_grad, self.model.parameters())
            self.optimizer = self.optimizer(params, lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)
            self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.n_epochs, eta_min=self.min_lr)
            self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1., total_epoch=self.warmup_epoch, after_scheduler=self.base_scheduler)

        best_epoch = -1
        best_dev_acc = -1
        train_loss_curve = []
        dev_acc_curve = []
        for epoch in range(checkpoint_epoch + 1, self.n_epochs + 1):
            self.model.train()
            
            loss_sum = 0.0
            i = 0
            for img, label in train_dataloader:
                i += 1

                img = img.to(self.device)
                label = label.to(self.device)

                outputs = self.model(img)
                loss = self.criterion(outputs, label)
                loss_sum += loss.item()

                if i % print_every == 0:
                    print(
                        f'[epoch {epoch}/{self.n_epochs}] averaged training loss of batch {i}/{len(train_dataloader)} = {loss.item()}'
                    )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

            loss_sum /= len(train_dataloader)
            train_loss_curve.append(loss_sum)

            if epoch % save_every == 0:
                print(f'\n======== [epoch {epoch}/{self.n_epochs}] saving the checkpoint ========\n')
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()
                    },
                    f=os.path.join(self.checkpoints_dir, f'epoch_{epoch}.pt')
                )

            with torch.no_grad():
                print(f'\n======== [epoch {epoch}/{self.n_epochs}] dev data evaluation ========\n')
                dev_acc = self.test(self.dev_dataset, mode='dev_eval')
                dev_acc_curve.append(dev_acc.cpu())

                if dev_acc > best_dev_acc:
                    best_epoch = epoch
                    best_dev_acc = dev_acc

        self._ploter(train_loss_curve, dev_acc_curve)
        print('\n#Params: ', sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        print(f'\nBest epoch = {best_epoch} with dev_eval acc = {best_dev_acc}\n')

    def test(self, dataset: Dataset, mode: str, print_every: int = 100):
        """
        Choose one of the three as the mode:
        ['train_eval', 'dev_eval', 'test_eval']
        """
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=0
        )

        self.model.to(self.device)
        self.model.eval()

        loss_sum = 0.
        correct_cnt = 0
        total_cnt = 0

        i = 0
        for img, label in dataloader:
            i += 1
            if i % print_every == 0:
                print(f'{mode} progress: {i}/{len(dataloader)}')

            img = img.to(self.device)
            label = label.to(self.device)

            outputs = self.model(img)
            loss = self.criterion(outputs, label)

            correct_cnt += (torch.argmax(outputs, dim=1) == label).sum()
            loss_sum += loss.item()
            total_cnt += outputs.size(0)

        loss_sum /= len(dataloader)
        acc = (correct_cnt / total_cnt) * 100
        print(f'averaged {mode} loss = {loss_sum}')
        print(f'{mode} acc = {acc: .3f}\n')

        return acc

    def load_checkpoint_and_train(self, checkpoint_epoch: int):
        """
        Input (checkpoint_epoch): set the epoch from which to restart training
        """
        checkpoint = torch.load(
            os.path.join(
                self.checkpoints_dir,
                f'epoch_{checkpoint_epoch}.pt'
            ),
            map_location=self.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = self.optimizer(params, lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']

        self.train(checkpoint_epoch=epoch)

    def load_checkpoint_and_test(
        self, checkpoint_epoch: int, mode: str
    ):
        """
        Choose one of the three modes as the mode:
        ['train_eval', 'dev_eval', 'test_eval']
        """
        checkpoint = torch.load(
            os.path.join(
                self.checkpoints_dir,
                f'epoch_{checkpoint_epoch}.pt'
            ),
            map_location=self.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])

        with torch.no_grad():
            if mode == 'train_eval':
                self.test(self.train_dataset, mode='train_eval')
            elif mode == 'dev_eval':
                self.test(self.dev_dataset, mode='dev_eval')
            elif mode == 'test_eval':
                self.test(self.test_dataset, mode='test_eval')
            else:
                raise ValueError('the mode should be one of train_eval, dev_eval and test_eval')
            
    def _ploter(self, train_loss_curve, dev_acc_curve):
        epochs = range(1, self.n_epochs + 1)

        fig, ax1 = plt.subplots()

        # Plot the training loss curve with its y-axis on the left side
        ax1.plot(epochs, train_loss_curve, 'b-', label='train_loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('train_loss', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # Setting the x-axis to use only integer values
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Create a secondary y-axis on the right side for development accuracy
        ax2 = ax1.twinx()
        ax2.plot(epochs, dev_acc_curve, 'r-', label='dev_acc')
        ax2.set_ylabel('dev_acc(%)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # Add legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # Save the plot
        plt.savefig('results.png', dpi=300, bbox_inches='tight')