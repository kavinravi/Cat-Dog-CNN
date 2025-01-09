# CNN for Image Classification

## Overview
This project implements a Convolutional Neural Network (CNN) for classifying images of dogs and cats. The repository includes:
- Code for building and training the CNN model.
- Code for predicting images using the trained CNN model.
- A dataset hosted on Google Drive containing inputs and labels for training and testing.

## Files Included
### 1. **CNN.ipynb**
- Contains the code for building, training, and exporting the CNN model.
- Loads training data from the dataset (see link below).

### 2. **Predict_Images.ipynb**
- Script to predict image labels using the trained CNN model.
- Input: Images or data in CSV format.
- Output: Predicted labels saved to a file or displayed on the console.

### 3. **cat_dog_classifier.keras**
- The trained CNN model exported in HDF5 format for future predictions.

---

## Dataset
The dataset is hosted on Google Drive and includes:
- `input.csv` and `labels.csv`: Used for training the CNN model.
- `input_test.csv` and `labels_test.csv`: Used for evaluating the model.

### Download Dataset
Download the dataset from Google Drive:
[**[Dataset Download Link](https://drive.google.com/file/d/1mkB3Dp6U9BSu8oOrEFP3sRVRoBCs30hm/view?usp=drive_link)**](#)

*Replace `#` with the link to the dataset on Google Drive.*

After downloading, extract the dataset:
```bash
unzip dataset.zip -d data/
```

Ensure the extracted files are located in the `data/` directory.

---

## Getting Started
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- TensorFlow/Keras
- NumPy
- pandas
- Matplotlib (optional, for visualizations)

Install dependencies using:
```bash
pip install tensorflow numpy pandas matplotlib
```

### Setup
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Download and extract the dataset (see instructions above).

3. Verify the project structure matches the provided layout.

---

## How to Use

### Train the Model
To train the CNN model:
```bash
python src/train_cnn.py
```
- Adjust hyperparameters in the script as needed.
- The trained model will be saved in the `model/` directory.

### Make Predictions
To use the trained model for predictions:
```bash
python src/predict_cnn.py --input <path_to_input_data>
```
- Replace `<path_to_input_data>` with the file path of the test data (e.g., `data/input_test.csv`).
- Predictions will be saved or displayed as specified in the script.

---

## Results and Visualizations
The trained CNN achieves the following metrics:
- **Accuracy**: xx% on the test set.
- **Loss**: yy% on the test set.

Visualization of training accuracy and loss is available in `train_cnn.py`. Modify the script to save plots.
