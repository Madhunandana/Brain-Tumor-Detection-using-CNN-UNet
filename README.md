# Brain Tumor Detection using CNN & UNet

## Project Overview
This project implements a deep learning model for brain tumor detection and segmentation using MRI images. It utilizes a U-Net architecture, a popular convolutional neural network designed for biomedical image segmentation. The model is trained to identify and segment tumors from brain MRI scans.

## Dataset
The project uses the [LGG MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) from Kaggle.
- The dataset contains brain MRI images together with their corresponding manual flair abnormality segmentation masks.
- It includes 110 patients with lower-grade gliomas (LGG).

## Model Architecture
The core of this project is the U-Net model, which consists of:
- **Encoder (Contracting Path):** Captures context via convolutional layers and max-pooling.
- **Decoder (Expansive Path):** Enables precise localization using transposed convolutions and skip connections from the encoder.

## Installation
To run this project, you need to install the required dependencies. You can install them using `pip`:

```bash
pip install -r requirements.txt
```

**Requirements:**
- tensorflow
- keras
- numpy
- pandas
- matplotlib
- opencv-python
- scikit-learn
- tqdm
- kagglehub

## Usage
1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Notebook:**
    Open `Brain_Tumor_Detection.ipynb` in Jupyter Notebook or Google Colab and execute the cells sequentially.
    - The notebook handles dataset downloading (via `kagglehub`), preprocessing, training, and evaluation.

## Results
The model is evaluated using metrics such as:
- **Dice Coefficient:** Measures the overlap between the predicted segmentation and the ground truth.
- **IoU (Intersection over Union):** Also known as the Jaccard index, measuring the accuracy of the segmentation.
- **Accuracy & Loss:** Standard training metrics.

(Note: Specific result values can be observed in the notebook output after training.)

## Credits
- Dataset: [Mateusz Buda - LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
- Original Notebook Logic: Based on standard U-Net implementations for medical imaging.
