# Deepfake Speech Detection

This project is a self-initiated study on deepfake speech detection, focusing on comparing the performance of two classification approaches: feature-based classification and image-based classification.

For the feature-based classification, this project utilizes Random Forest and Support Vector Machine (SVM). Meanwhile, for the image-based classification, it employs Convolutional Neural Networks (CNN) and a hybrid CNN-LSTM model.

The dataset used in this project is ASVSpoof2019 Logical Access, which can be accessed at [this link](https://www.asvspoof.org/index2019.html).

## Key Features
* Feature-based classification with Random Forest and SVM
* Image-based classification with CNN and CNN-LSTM
* Performance comparison of different classification approaches

## Steps Overview

### 1. Dataset Preparation
* Download the ASVSpoof2019 Logical Access dataset from [this link](https://www.asvspoof.org/index2019.html).
* Perform audio preprocessing, including:
  * Undersampling to balance class distribution (only applied to training and validation data).

### 2. Feature Extraction
* Extract **feature-based** features:
  * Chromagram (1-12), MFCC (1-20), spectral centroid, spectral spread, spectral rolloff, zero crossing rate, and root mean square.
* Extract **image-based** features:
  * Convert audio into **mel-spectrograms**.
* The feature extraction process can be found [here](https://github.com/GalihFt/Deepfake-Speech-Project/tree/f871b327b7dfe48e737af6f0a008fe4575da5ebb/Data%20Extract).

### 3. Data Preprocessing
* Normalize numerical variables using **min-max normalization**.
* Resize images to **128 × 128 pixels** to standardize mel-spectrogram inputs.

### 4. Feature-Based Classification (Machine Learning)
* Train **Support Vector Machine (SVM)** and **Random Forest** models:
  * Initialize models and parameters.
  * Train models and measure training time.
  * Evaluate performance using the validation set.
* The machine learning model training process can be found [here](https://github.com/GalihFt/Deepfake-Speech-Project/blob/f871b327b7dfe48e737af6f0a008fe4575da5ebb/Model/Klasifikasi_Tabular.ipynb).

### 5. Image-Based Classification (Deep Learning)
* Train **CNN** and **CNN-LSTM** models:
  * Build architectures and initialize parameters.
  * Train models and monitor validation performance.
  * Measure training time and evaluate model performance.
* The deep learning model training process can be found [here](https://github.com/GalihFt/Deepfake-Speech-Project/blob/f871b327b7dfe48e737af6f0a008fe4575da5ebb/Model/Klasifikasi_Image.ipynb).

### 6. Hyperparameter Tuning
* Optimize all models based on training and validation performance.

### 7. Model Evaluation
* Compare the performance of each model to determine the best-performing classifier.

## Conclusion
The following table summarizes the accuracy of the different classification approaches used in this project:

| Model          | Accuracy (%) |
|--------------|--------------|
| CNN          | 96.55%       |
| CNN-LSTM     | 95.02%       |
| Random Forest (RF) | 83.44% |
| Support Vector Machine (SVM) | 82.85% |

Based on the results, **image-based classifiers (CNN and CNN-LSTM) significantly outperform feature-based classifiers (Random Forest and SVM)** in detecting deepfake speech. CNN achieved the highest accuracy at **96.55%**, demonstrating the effectiveness of convolutional neural networks in extracting patterns from mel-spectrograms.

Meanwhile, the feature-based classifiers (RF and SVM) achieved lower accuracy, suggesting that handcrafted features alone may not be sufficient for deepfake speech detection. However, these methods still provide useful insights and can be computationally more efficient compared to deep learning models.

Although **image-based classifiers provide higher accuracy, they also require significantly longer training and inference time compared to feature-based classifiers**. This trade-off between accuracy and computational efficiency should be considered when choosing an approach for real-time or resource-constrained applications.

Overall, this study highlights the superiority of deep learning-based approaches in the field of deepfake speech detection, reinforcing the importance of leveraging spectrogram-based representations for classification tasks. **This research will continue to be developed and improved over time to enhance performance, explore additional techniques, and adapt to new challenges in deepfake detection.**

