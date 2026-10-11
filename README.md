# RetinGuard: AI-Powered Diabetic Retinopathy Detection System
**RetinGuard** is an AI-powered early detection and severity classification system for Diabetic Retinopathy (DR). Built to assist medical professionals, this system analyzes human fundus images using Deep Learning to automatically identify pathological characteristics and classify the severity of the disease.

---

## About The Project
Diabetic Mellitus often leads to severe microvascular complications, with Diabetic Retinopathy being a leading cause of preventable blindness. Conventional screening heavily relies on manual visual analysis by ophthalmologists, which is time-consuming and subjective.

RetinGuard addresses this by leveraging a Convolutional Neural Network (CNN) to automate the screening process. By classifying fundus images into five standardized medical severity levels, the system acts as a reliable, objective, and efficient initial triage tool.

## 5 Levels of Severity Classification:
- 0 (No_DR): No abnormalities or lesions detected.
- 1 (Mild): Early-stage microaneurysms present.
- 2 (Moderate): Blockage in some retinal blood vessels.
- 3 (Severe): Widespread blood vessel blockage causing restricted blood supply.
- 4 (Proliferative): Advanced stage with fragile new blood vessels prone to bleeding.

## Model Architecture & Dataset
- **Dataset:** Kaggle Diabetic Retinopathy (Resized) comprising 35,108 high-resolution fundus images.
- **Class Balancing:** Oversampling techniques were applied to handle class imbalance, ensuring fair feature learning across all severity levels.
- **Base Model:** EfficientNetB3 (Transfer Learning via ImageNet weights).
- **Optimization:** Mixed precision (float16) to speed up training, coupled with GlobalAveragePooling2D and GlobalMaxPooling2D concatenation for maximum feature extraction.
- **Performance:** Achieved a validation accuracy of 67.78% and an AUC of 0.8927, demonstrating strong generalization capabilities for medical screening.
