import argparse
import torch
import numpy as np

from dataloader import get_dataset
from trainer import ViTTrainer
from models import VisionTransformer


def main(args):
    # Load the CIFAR-10 dataset
    train_dataset, test_dataset = get_dataset()

    trainer = ViTTrainer(
        n_epochs=args.n_epochs,
        device=torch.device(args.device),
        model=VisionTransformer(
            num_classes=args.num_classes,
            patch_size=args.patch_size,
            dropout_p=args.dropout_p,
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads
        ),
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        checkpoints_dir=args.checkpoints_dir,
        train_dataset=train_dataset,
        dev_dataset=test_dataset,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
    )

    if args.mode == 'train':
        # default training from the beginning
        print('================= training =================\n')
        trainer.train()

    elif args.mode == 'train_on':
        # train from (checkpoint_epoch + 1)
        print('================= training on =================\n')
        trainer.load_checkpoint_and_train(checkpoint_epoch=args.checkpoint_epoch)
    
    elif args.mode == 'test':
        # test using the saved model from checkpoint_epoch
        print(f'================= testing: {args.test_mode} =================\n')
        trainer.load_checkpoint_and_test(
            checkpoint_epoch=args.checkpoint_epoch,
            mode=args.test_mode
        )

    else:
        raise RuntimeError('Please check the mode of the trainer!!!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vision Transformer')

    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--dropout_p', type=float, default=0.0)
    parser.add_argument('--num_layers', type=int, default=7)
    parser.add_argument('--hidden_dim', type=int, default=384)
    parser.add_argument('--mlp_dim', type=int, default=384)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'train_on', 'test'])
    parser.add_argument('--test_mode', type=str, default='dev_eval',
                        choices=['train_eval', 'dev_eval', 'test_eval'])
    parser.add_argument('--checkpoint_epoch', type=int, default=14)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)