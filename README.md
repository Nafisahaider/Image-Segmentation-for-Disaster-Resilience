#  Image Segmentation for Disaster Resilience

##  Problem Statement

The objective of this project is to develop a **deep learning-based image segmentation model** that can identify and classify disaster-affected regions (e.g., flooded areas) from satellite or drone imagery. This aids in **faster emergency response and damage assessment** following natural disasters such as floods, earthquakes, or wildfires.



##  Dataset: FloodNet Track 1

- **Dataset Used**: [FloodNet Track 1 (12GB)](https://github.com/hasibzunair/floodnet-dataset)
- **Description**: A large-scale dataset of post-flood urban satellite imagery with pixel-wise segmentation annotations.
- **Classes**: Background, Flooded Building, Non-Flooded Building, Water, Road, Vegetation, etc.
- **Input**: RGB satellite images
- **Annotations**: Segmentation masks



##  Week-wise Project Plan

###  Week 1: Dataset Preparation & EDA

-  Downloaded and extracted the **FloodNet Track 1** dataset (12GB)
-  Preprocessed the dataset:
  - Resized images
  - Verified annotation integrity
  - Mapped label colors to class IDs
-  Performed EDA:
  - Image dimension and mask shape analysis
  - Class distribution visualization
  - Sample image + mask visualization


###  Week 2: Baseline Modeling

-  Trained a **basic CNN** and a **U-Net model** on preprocessed data
-  Applied **data augmentation**:
  - Horizontal/vertical flips
  - Random rotation
  - Brightness and contrast adjustments
-  Evaluation Metrics:
  - **IoU (Intersection over Union)**
  - **Dice Coefficient**
-  Visualized segmentation results (predicted vs. ground truth)

####  Mid-Project Review
- Basic U-Net trained
- Segmentation masks generated
- Evaluation metrics and visualization completed

---

###  Week 3: Model Improvement

-  Fine-tuned with **pretrained encoders**:
  - ResNet-50 + U-Net
  - VGG-16 + U-Net
-  Tackled **class imbalance** using:
  - Focal Loss
  - Class-weighted cross-entropy
-  Performed **hyperparameter tuning**:
  - Learning rate, batch size, optimizer (Adam, SGD)
  - Epoch count, dropout rate, weight decay

---

###  Week 4: Final Evaluation & Report

-  Selected best-performing model: **U-Net with ResNet-50 encoder**
-  Evaluated on unseen test set (holdout split)
-  Documented:
  - Final IoU, Dice scores
  - Confusion matrix (optional)
-  Side-by-side comparison of:
  - Input image
  - Ground truth mask
  - Predicted segmentation



##  Evaluation Metrics

| Metric           | Description                                               |
|------------------|-----------------------------------------------------------|
| IoU              | Measures the overlap between predicted and true segments |
| Dice Coefficient | Harmonic mean of precision and recall for segmentation   |
| Accuracy         | Pixel-wise classification accuracy (optional)            |



##  Technologies & Libraries

- Python
- TensorFlow / PyTorch
- OpenCV, PIL
- Albumentations (for augmentation)
- Matplotlib, Seaborn (visualization)
- NumPy, Pandas



##  References

- [FloodNet Dataset GitHub](https://github.com/hasibzunair/floodnet-dataset)
- Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation (2015)
- Lin et al., Focal Loss for Dense Object Detection

---

##  Author

- Nafisa Haider

---

##  License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
