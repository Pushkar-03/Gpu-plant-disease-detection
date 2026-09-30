# Google Colab Setup Guide

Because deep learning models require heavy parallel processing, attempting to train this project on a local CPU-only machine is strongly discouraged. We rely on Google Colab to access free NVIDIA GPU hardware.

Follow these exact steps to execute this capstone project:

1. **Open Google Colab:** Navigate to [colab.research.google.com](https://colab.research.google.com/).
2. **Upload the Notebook:** Click `File` > `Upload notebook` and select `notebooks/gpu_training_and_benchmark.ipynb` from this repository.
3. **Change Runtime Type:** In the top menu, click `Runtime` > `Change runtime type`.
4. **Select GPU:** Under "Hardware accelerator", select `GPU` (usually a T4). Save.
5. **Clone the Repository:** Run the first few cells. The notebook will automatically clone your GitHub repository into the Colab environment.
6. **Install Dependencies:** The notebook will execute `pip install -r requirements.txt`.
7. **Place the Dataset:** You have two options:
   - Upload a `.zip` file of your dataset directly to Colab and extract it to `/content/gpu-plant-disease-detection/data/raw/`.
   - Mount your Google Drive (as shown in the notebook) and copy the dataset over.
8. **Run the Notebook:** Execute the remaining cells in order. The notebook will handle dataset verification, model training, and the final GPU vs. CPU benchmarking.
9. **Collect Generated Results:** Look in the file explorer on the left sidebar under `gpu-plant-disease-detection/results/`.
10. **Copy Artifacts Locally:** Download the generated lightweight artifacts (CSVs, PNG plots, and the `.txt` environment report) to your local machine and commit them to your repository to finalize your capstone submission. Do **NOT** attempt to push the heavy `.pth` model back to GitHub!
