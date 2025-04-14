import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import C3Rec
from dataset import DatasetLoader
from trainer import train_model, evaluate_fitb

class Args:
    def __init__(self):
        self.embed_size = 128
        self.num_items = 16
        self.alpha = 0.5
        self.lambda_reg = 1e-5
        self.r1 = 1e-3
        self.outdim_size = 64
        self.lr = 1e-4
        self.num_epochs = 10
        self.batch_size = 32
        self.datadir = './data'
        self.polyvore_split = 'disjoint'
        self.task = 'compatibility'  # 或 'fitb'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    args = Args()

    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = DatasetLoader(args, split='train', task=args.task, transform=transform)
    val_dataset = DatasetLoader(args, split='valid', task=args.task, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = C3Rec(args)
    model.to(args.device)

    best_val_loss = train_model(model, train_loader, val_loader, args)

    if args.task == 'fitb':
        test_dataset = DatasetLoader(args, split='test', task='fitb', transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        accuracy = evaluate_fitb(model, test_loader, args)
        print(f"FITB Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
