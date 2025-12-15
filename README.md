# CNN Image Classification (Cats vs Dogs)

This project implements a **Convolutional Neural Network (CNN)** using **TensorFlow / Keras** to perform **binary image classification** (Cats vs Dogs).

The model is trained on images stored in directory structures and uses data augmentation to improve generalization.

---

## 🧠 Model Overview

* **Type:** Convolutional Neural Network (CNN)
* **Task:** Binary Image Classification
* **Framework:** TensorFlow (Keras API)
* **Optimizer:** Adam
* **Loss Function:** Binary Crossentropy
* **Metric:** Accuracy

---

## 📂 Dataset Structure

Your dataset directory should follow this structure:

```
dataset/
├── training_set/
│   ├── cats/
│   └── dogs/
├── test_dataset/
│   ├── cats/
│   └── dogs/
└── single_prediction/
    └── cat_or_dog_1.jpg
```

> Folder names (`cats`, `dogs`) are used automatically as class labels.

---

## 🔧 Data Preprocessing

### Training Set Augmentation

The training images are augmented to reduce overfitting:

* Rescaling pixel values (0–255 → 0–1)
* Random shear
* Random zoom
* Horizontal flipping

```python
ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
```

### Test Set Preprocessing

Only rescaling is applied to the test set (no augmentation).

---

## 🏗️ CNN Architecture

| Layer        | Description                  |
| ------------ | ---------------------------- |
| Conv2D       | 32 filters, 3×3 kernel, ReLU |
| MaxPooling2D | 2×2 pooling                  |
| Conv2D       | 32 filters, 3×3 kernel, ReLU |
| MaxPooling2D | 2×2 pooling                  |
| Flatten      | Converts feature maps to 1D  |
| Dense        | 128 units, ReLU              |
| Output Dense | 1 unit, Sigmoid              |

---

## ⚙️ Model Compilation

```python
cnn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

* **Adam Optimizer:** Adaptive learning rate optimization
* **Binary Crossentropy:** Suitable for two-class classification
* **Accuracy:** Tracks prediction correctness

---

## 🚀 Training the Model

```python
cnn.fit(
    x=training_set,
    validation_data=test_set,
    epochs=25
)
```

* **Batch Size:** 32
* **Image Size:** 64 × 64
* **Epochs:** 25

---

## 🔮 Single Image Prediction

A single image can be classified after training:

```python
result = cnn.predict(test_image)
```

Prediction logic:

* `0 → Cat`
* `1 → Dog`

Example output:

```text
dog
```

---

## 📦 Dependencies

```txt
tensorflow
numpy
```

Install with:

```bash
pip install tensorflow numpy
```

---

## 📌 Notes

* Image size is fixed at **64×64** for faster training
* Designed for educational / beginner CNN projects
* Can be extended with:

  * More convolution layers
  * Dropout for regularization
  * Larger image resolution

---

## 👨‍💻 Author

Created for learning **CNN-based image classification using TensorFlow & Keras**.
