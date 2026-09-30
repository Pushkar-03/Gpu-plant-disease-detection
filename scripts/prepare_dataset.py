import sys
import csv
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config

def prepare_dataset():
    raw_dir = config.RAW_DATA_DIR
    if not raw_dir.exists():
        print(f"Dataset directory not found: {raw_dir}")
        print("Please place the ImageFolder dataset under data/raw/")
        sys.exit(1)
        
    classes = [d for d in raw_dir.iterdir() if d.is_dir()]
    if not classes:
        print(f"Dataset directory not found or empty: {raw_dir}")
        print("Please place the ImageFolder dataset under data/raw/")
        sys.exit(1)
        
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    class_counts = {}
    total_images = 0
    
    for c_dir in classes:
        images = [f for f in c_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_extensions]
        count = len(images)
        if count == 0:
            print(f"[WARNING] Empty class directory detected: {c_dir.name}")
        elif count < 10:
            print(f"[WARNING] Very few images ({count}) in class directory: {c_dir.name}")
            
        class_counts[c_dir.name] = count
        total_images += count
        
    print(f"Dataset path: {raw_dir}")
    print(f"Number of classes: {len(classes)}")
    print(f"Classes: {[c.name for c in classes]}")
    print(f"Total images: {total_images}\n")
    
    print("Class distribution:")
    for c_name, count in class_counts.items():
        print(f"  {c_name}: {count} images")
        
    if total_images > 0:
        report_path = config.RESULTS_DIR / "dataset_report.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['class_name', 'image_count'])
            for c_name, count in class_counts.items():
                writer.writerow([c_name, count])
        print(f"\nDataset report saved to {report_path}")
    else:
        print("\nNo valid images found in dataset.")

if __name__ == "__main__":
    prepare_dataset()
