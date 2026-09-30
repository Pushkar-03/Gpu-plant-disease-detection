# Dataset Directory

## Expected Dataset Structure
The project utilizes `torchvision.datasets.ImageFolder`. Your dataset must be organized in the following directory structure inside `data/raw/`:

```text
data/raw/
    Class_1_Name/
        image1.jpg
        image2.png
    Class_2_Name/
        image1.jpg
        image2.jpg
```

## Supported Image Formats
Supported formats include `.jpg`, `.jpeg`, `.png`, and `.bmp`. The preprocessing pipeline automatically resizes and normalizes images for EfficientNet.

## Notes
- Class folder names are automatically mapped as class labels.
- Do NOT commit datasets to version control. The `.gitignore` prevents `data/raw/` from being pushed to GitHub.
