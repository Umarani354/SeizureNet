
# 🧠 Epileptic Seizure Detection using CNN Architectures  
*Comparing LeNet, AlexNet, GoogLeNet, and ResNet for EEG-based seizure detection*

<p align="center">
  <img src="https://github.com/Umarani354/SeizureNet/blob/main/poster.jpg"  width="400">
</p>

## 📌 Overview
Epilepsy is a neurological disorder characterized by recurrent seizures, and early detection plays a crucial role in timely treatment.  
This project leverages **deep learning** to classify EEG signals as seizure or non-seizure using four different **Convolutional Neural Network (CNN)** architectures:

- **LeNet**
- **AlexNet**
- **GoogLeNet**
- **ResNet**

The goal is to **compare the performance** of these architectures and identify the most effective model for epileptic seizure detection.
## 📂 Dataset
We used the **Bonn University EEG dataset**, containing recordings from **500 subjects**:

| Set | Condition | Count |
|-----|-----------|-------|
| A | Healthy (Eyes Open) | 100 |
| B | Healthy (Eyes Closed) | 100 |
| C | Epileptic (Non-seizure, Healthy Brain Regions) | 100 |
| D | Epileptic (Non-seizure, Tumor-affected Brain Regions) | 100 |
| E | Epileptic (During Seizure) | 100 |

- **Sampling Rate:** 173.6 Hz  
- **Duration per recording:** 23.6 seconds  
- **Data points per recording:** 4097  
## 🛠 Methodology
1. **Preprocessing**
   - Z-score normalization
   - Outlier clipping (±3 threshold)
   - Reshaping to `(samples, timesteps, features)` for Conv1D
   
2. **Model Training**
   - **LeNet**: Simple Conv1D layers, dropout, class weights
   - **AlexNet**: 5 Conv1D layers, batch normalization, L2 regularization
   - **ResNet**: Residual blocks, LeakyReLU, skip connections
   - **GoogLeNet**: Inception modules for multi-scale feature extraction
3. **Evaluation**
   - Accuracy, Sensitivity, Specificity, ROC-AUC, Confusion Matrix
   - Precision-Recall curves
## 📊 Results


| Model      | Accuracy (%) | Sensitivity (%) | Specificity (%) |
|------------|-------------:|----------------:|----------------:|
| LeNet      | 95.00        | 95.71           | 94.13           |
| AlexNet    | 97.52        | 97.17           | 97.61           |
| ResNet     | 97.00        | 94.00           | 96.00           |
| **GoogLeNet** | **98.96**  | **97.61**       | **99.08**       |

**Classification Report Highlights (Epileptic class)**  
<p align="center">
  <img src="https://github.com/Umarani354/SeizureNet/blob/main/Model%20Classification%20report.png"  width="700">
</p>
  

> 📌 **Winner:**  🏆GoogLeNet — highest accuracy & specificity, excellent generalization.
## 🚀 How to Run (Google Colab)

1. **Open in Google Colab**  
   Click the badge below to launch the notebook in Colab: 
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Umarani354/SeizureNet/blob/main/epileptic_seziure.ipynb)

2. **Enable GPU Runtime**  
   - Go to `Runtime` → `Change runtime type` → set **Hardware accelerator** to **GPU**.

3. **Download & Place Dataset**  
   - Download from the repo.  
   - In Colab, upload the dataset to the file path specified in the code (check the cell that loads the dataset and replace with your uploaded file’s path).  

4. **Run All Cells**  
   - From the menu, select `Runtime` → `Run all`.  
   - The notebook will:
     - Load the EEG dataset from the provided path
     - Preprocess the data (Z-score normalization & clipping)
     - Train **LeNet, AlexNet, ResNet, GoogLeNet**
     - Display comparison metrics & visualizations

5. **View Results**  
   - Accuracy & loss plots  
   - ROC curves  
   - Classification reports
## 🛠 Tech Stack  
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white) ![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
