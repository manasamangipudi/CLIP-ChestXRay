# Disease Detection in Chest X-rays using CLIP Features with Interpretable ROI Analysis

This project implements a multi-modal deep learning approach for classifying chest X-rays as normal or diseased by leveraging both radiographic images and associated textual reports. The system uses CLIP (Contrastive Language-Image Pre-training) to learn joint visual-textual representations, followed by a classifier for predictions.

## Key Features

- Multi-modal learning combining chest X-rays and radiological reports
- CLIP-based architecture with Vision Transformer (ViT) and BERT
- Interpretable results using Grad-CAM visualizations
- High classification accuracy (83.17% on test dataset)
- ROC-AUC score of 0.8093

## Architecture

The pipeline consists of:
1. Vision Transformer (ViT) for image encoding
2. BERT for text encoding 
3. CLIP model for learning joint representations
4. Classifier head for final predictions
5. Grad-CAM for visualization of important regions

## Dataset

Uses the IU Chest X-Ray dataset:
- 3,955 radiology reports
- 7,470 chest X-rays
- 3,955 unique patients
- Split: Train (2,725), Validation (584), Test (585)

## Performance

| Model               | ROC-AUC |
|--------------------|---------|
| Our Single-Modal   | 0.7094  |
| Our Multi-Modal    | 0.8093  |
| ResNet-18          | 0.8340  |
| DenseNet-121       | 0.8420  |

## Implementation Details

- ViT: patch-16-224 pre-trained weights
- BERT: uncased weights
- AdamW optimizer
- Learning rate: 1e-5
- Batch size: 32
- Training epochs: 25
- Temperature parameter: 0.1

## Requirements

- Python 3.x
- PyTorch
- Transformers
- OpenCV
- NumPy
- Pandas

## Future Work

- Explore alternative architectures and fusion techniques
- Investigate zero-shot classification
- Develop models for automated reporting
- Evaluate Grad-CAM using segmentation masks
- Extend to larger datasets
