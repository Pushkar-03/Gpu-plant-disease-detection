import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from src.device import get_device

def main():
    parser = argparse.ArgumentParser(description="Train GPU Plant Disease Detection Model")
    parser.add_argument("--data-dir", type=str, default=str(config.RAW_DATA_DIR), help="Path to raw dataset")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS, help="Number of DataLoader workers")
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE[0], help="Image size (square)")
    parser.add_argument("--dry-run", action="store_true", help="Verify pipeline without executing training")
    
    args = parser.parse_args()
    
    print("="*50)
    print("TRAINING PIPELINE CONFIGURATION")
    print("="*50)
    
    device = get_device()
    
    try:
        import torch
        from src.dataset import get_dataloaders
        from src.model import create_model
        from src.train import train_model
    except ImportError:
        print("\n[ERROR] PyTorch is not installed.")
        print("Cannot construct DataLoaders or Models locally.")
        print("Please run this script in a PyTorch-enabled environment (e.g. Google Colab).")
        sys.exit(1)
        
    data_path = Path(args.data_dir)
    if not data_path.exists() or not any(data_path.iterdir()):
        print(f"\n[ERROR] Dataset not found or empty at {data_path}")
        print("Run 'python scripts/prepare_dataset.py' first to verify dataset.")
        sys.exit(1)
        
    print("\nLoading dataset and creating DataLoaders...")
    image_size = (args.image_size, args.image_size)
    train_loader, val_loader, classes = get_dataloaders(
        data_dir=data_path, image_size=image_size,
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    num_classes = len(classes)
    
    print("\nInitializing EfficientNet-B0 Model...")
    model = create_model(num_classes=num_classes, pretrained=True)
    
    print("\nConfiguration Summary:")
    print(f" - Device: {device}")
    print(f" - Dataset: {data_path}")
    print(f" - Classes Detected: {num_classes}")
    print(f" - Image Size: {image_size}")
    print(f" - Batch Size: {args.batch_size}")
    print(f" - Epochs: {args.epochs}")
    print(f" - Learning Rate: {args.learning_rate}")
    print(f" - Number of Workers: {args.num_workers}")
    
    if args.dry_run:
        print("\n[DRY RUN] Pipeline successfully verified. Exiting without training.")
        sys.exit(0)
        
    print("\nStarting Training...")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=args.epochs, device=device)

if __name__ == "__main__":
    main()
