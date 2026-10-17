<div align="center">

<img src="assets/banner.svg" width="100%" alt="DeepSkin Banner">

# DeepSkin

**Deep learning for dermoscopic skin-lesion classification.**

A transparent benchmark, a reasoned model choice, and a working demo — trained on the HAM10000 dataset.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Dataset-HAM10000-0d9488?style=flat-square)](https://doi.org/10.1038/sdata.2018.161)
[![License](https://img.shields.io/badge/License-MIT-12a56b?style=flat-square)](LICENSE)

**[Overview](#overview) · [Results](#results) · [Classes](#the-seven-lesion-classes) · [Pipeline](#pipeline) · [Quick start](#quick-start) · [Author](#author)**

</div>

---

> ⚠️ **DeepSkin is not a medical device.** It is an educational project and must not be used as a substitute for professional diagnosis. If you are worried about a skin lesion, see a dermatologist.

---

## Overview

DeepSkin is an end-to-end deep learning pipeline that classifies dermoscopic images into **seven types of skin lesion**. Trained on **HAM10000** — 10,015 dermoscopic images — it supports rapid preliminary screening that helps flag skin cancer early.

The project rests on three ideas:

- **A fair benchmark.** A custom CNN baseline is compared with two ImageNet-pretrained architectures (MobileNetV2, DenseNet121) under identical preprocessing.
- **A reasoned selection.** The final model balances accuracy, inference speed, and model size — not accuracy alone.
- **A working demo.** A Streamlit app turns an uploaded image into per-class confidence scores.

---

## Results

All models were evaluated on the same HAM10000 test split.

| Model | Architecture idea | Accuracy | Training Time | Takeaway |
|:--|:--|:--:|:--:|:--|
| Custom CNN | 4 conv blocks + dense head | **75.3%** | 431.6 min | Strong baseline, but slow |
| MobileNetV2 | Depthwise separable convolutions | **76.3%** | **31.6 min** | **Fastest training (13x faster)** |
| **DenseNet121** | Dense connections | **76.6%** | 169.1 min | **Selected — highest accuracy** |

> **Insight:** While DenseNet121 achieved the best accuracy (76.6%), MobileNetV2 trained in just 31.6 minutes (compared to 431.6 minutes for the Custom CNN), making it the ideal choice for resource-constrained edge devices.

**DenseNet121** is the selected model: the highest accuracy on the seven-class task, paired with a sound balance of model size and inference speed.

### Training Progress

| DenseNet121 | MobileNetV2 |
|:--:|:--:|
| ![DenseNet121 Training](assets/densenet_graph.png) | ![MobileNetV2 Training](assets/mobilenet_graph.png) |

---

## The seven lesion classes

<table style="display: table; width: 100%;">
  <thead>
    <tr>
      <th align="center">Code</th>
      <th align="left">Lesion</th>
      <th align="left">Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><code>akiec</code></td>
      <td>Actinic keratoses</td>
      <td>🟡 Pre-cancerous</td>
    </tr>
    <tr>
      <td align="center"><code>bcc</code></td>
      <td>Basal cell carcinoma</td>
      <td>🔴 Malignant</td>
    </tr>
    <tr>
      <td align="center"><code>bkl</code></td>
      <td>Benign keratosis</td>
      <td>🟢 Benign</td>
    </tr>
    <tr>
      <td align="center"><code>df</code></td>
      <td>Dermatofibroma</td>
      <td>🟢 Benign</td>
    </tr>
    <tr>
      <td align="center"><code>mel</code></td>
      <td>Melanoma</td>
      <td>🔴 Malignant</td>
    </tr>
    <tr>
      <td align="center"><code>nv</code></td>
      <td>Melanocytic nevi</td>
      <td>🟢 Benign</td>
    </tr>
    <tr>
      <td align="center"><code>vasc</code></td>
      <td>Vascular lesions</td>
      <td>🟢 Benign</td>
    </tr>
  </tbody>
</table>

📊 **The dataset is heavily imbalanced.** Melanocytic nevi (`nv`) account for roughly 67% of all samples. Stratified splits keep that distribution consistent across train, validation, and test, while class weights compensate during training.

---

## Pipeline

<img src="assets/inference.svg" alt="How DeepSkin reads a lesion" width="100%">

*The class confidences shown are illustrative, not real model output.*

**1 · Data preprocessing** — 10,015 dermoscopic images resized to each model's input size; stratified splits that preserve class balance; augmentation with rotation, flips, zoom, brightness, and shear.

**2 · Model training** — a custom CNN baseline alongside transfer learning from ImageNet weights for MobileNetV2 and DenseNet121; class weights for imbalance; early stopping and learning-rate scheduling.

**3 · Evaluation and selection** — per-class precision, recall, F1, and ROC-AUC; confusion-matrix analysis; automated selection weighing accuracy, speed, and size; error analysis on the weakest classes.

**4 · Deployment** — a Streamlit app with real-time upload, confidence scores for every class, and a class-wise metrics dashboard.

---

## Quick start

```bash
# 1. Clone the repository
git clone https://github.com/sonubiswal/deepskin-skin-cancer-detection.git
cd deepskin-skin-cancer-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train a model (pick one)
python train_densenet.py
python train_cnn.py

# 4. Launch the web demo
streamlit run app.py
```

---

## Citation

DeepSkin is trained on the HAM10000 dataset. If you use the data, please cite the original paper:

> Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data 5, 180161 (2018). [doi:10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)

---

## Author

**Sumit Kumar Biswal** — Final-year B.Tech Computer Science student at GIET University, focused on computer vision and deep learning.

[![GitHub](https://img.shields.io/badge/GitHub-sonubiswal-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/sonubiswal)

---

Released under the [MIT License](LICENSE). If DeepSkin helped you, consider giving it a star.
