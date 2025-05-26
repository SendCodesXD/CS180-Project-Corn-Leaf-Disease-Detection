# CS 180 Project: Corn Leaf Disease Detection

**Group Members:** John Domingo, Garlu Nepomuceno, Vinz Rentino  
**Assigned Dataset:** Corn Leaf Disease (Classes: Blight, Common Rust, Gray Leaf Spot, Healthy)

## Project Overview

This project is part of the CS 180 shared task on precision agriculture, specifically focusing on plant disease detection through image classification. We were assigned the Corn Leaf Disease dataset and tasked with developing two different solutions to classify images into predefined disease categories or as healthy. This README section details our first solution: a deep learning-based approach that does not employ transformer architectures.

The project operates in a closed mode, meaning only the provided dataset was used for developing the solutions.

## Solution 1: Deep Learning without Transformer (EfficientNetV2B2 with Transfer Learning)

### Approach

For our first solution, we employed a transfer learning approach using a pre-trained EfficientNetV2B2 model. The process involved:
1.  **Loading a Pre-trained Model:** EfficientNetV2B2 pre-trained on ImageNet was used as the base. The top classification layer was excluded to allow for a custom head suited for our 4-class problem (Blight, Common Rust, Gray Leaf Spot, Healthy).
2.  **Freezing Base Layers:** Initially, all layers of the EfficientNetV2B2 base model were frozen.
3.  **Adding Custom Top Layers:** A new classification head was added, consisting of a BatchNormalization layer, a Dropout layer (0.35), and a Dense output layer with softmax activation and L2 regularization (1e-4).
4.  **Initial Training:** Only the custom top layers were trained using the Adam optimizer (learning rate 0.001) and Categorical Crossentropy loss. Class weights were applied during this phase to address the imbalance in the dataset (Gray Leaf Spot being the minority class).
5.  **Fine-Tuning:** After the initial training, the top 20% of layers in the base EfficientNetV2B2 model were unfrozen. The entire model was then fine-tuned with a lower learning rate (Adam optimizer, learning rate 1e-5) to adapt the learned features to the corn leaf dataset.
6.  **Callbacks:** EarlyStopping (monitoring validation F1-score, patience 10, restoring best weights) and ReduceLROnPlateau (monitoring validation F1-score, patience 5, factor 0.2) were used during both training phases.

The primary metric for evaluation and callback monitoring was the F1-score (macro average), as per project guidelines.

### Code Structure

The code for this solution is organized into the following Jupyter Notebooks:

1.  **`CornLeaf_EfficientNetV2B2_TrainingCode.ipynb`**:
    * This notebook contains the complete code for data loading, preprocessing (including augmentation and class weight calculation), model building (EfficientNetV2B2 with custom head), initial training, and fine-tuning.
    * It also includes code for plotting training/validation metrics (loss, accuracy, F1-score) and evaluating the final model on the validation set.
    * **To run training:** Execute all cells in this notebook sequentially. Ensure the `TRAIN_DIR` path in Cell 3 points to your corn leaf training dataset. The trained model (best weights based on validation F1-score) is saved to `/content/drive/MyDrive/CornLeaf_EfficientNetv2b2.keras` (path can be adjusted in Cell 11 of the training code if Cell 14 was not run, or Cell 14 if it was).

2.  **`CornLeaf_EfficientNetV2B2_MakePredictions.ipynb`**:
    * This notebook is designed to load the trained EfficientNetV2B2 model and generate predictions on a given test set (provided as a `.zip` file).
    * It preprocesses each test image and outputs a `predictions_EfficientNetV2B2.csv` file [cite: 1] with two columns: `id` (image filename) and `label` (predicted class name). This notebook was used to generate the submitted `predictions_EfficientNetV2B2.csv` file.
    * **To run for generating predictions:**
        1.  Ensure the `MODEL_PATH` in Cell 2 points to the saved `.keras` model file from the training notebook.
        2.  Ensure the `TEST_ZIP_PATH` in Cell 2 points to the `corn_test.zip` file.
        3.  Adjust `CSV_OUTPUT_PATH` in Cell 6 if needed.
        4.  Execute all cells sequentially.

3.  **`CornLeaf_EfficientNetV2B2_DemoCode.ipynb`**:
    * This notebook serves as the demo code for showcasing the solution in real-time.
    * It loads the trained EfficientNetV2B2 model, allows specification of a single image filename from the (extracted) test set, preprocesses the image, displays it, and prints the predicted class with a confidence score.
    * **To run the demo:**
        1.  Ensure `MODEL_PATH_DEMO` in Cell 2 points to the saved `.keras` model file.
        2.  Ensure `TEST_ZIP_PATH_DEMO` in Cell 2 points to the `corn_test.zip` file.
        3.  In Cell 6, change the `image_filename_to_demo` variable to the name of the image you want to test from the extracted test set.
        4.  Execute all cells sequentially. This notebook is designed to be self-contained for demo purposes, including unzipping the test data.

### Model Storage

* The trained EfficientNetV2B2 model (`CornLeaf_EfficientNetv2b2.keras`) is stored in Google Drive.
* **Link to Model:** https://drive.google.com/file/d/1_QHeODI9AKkfgPZ_M4gsBUjqtTGYIcKV/view?usp=sharing

### Predictions File
* The `predictions_EfficientNetV2B2.csv` file [cite: 1] submitted for this solution was generated using `CornLeaf_EfficientNetV2B2_MakePredictions.ipynb`. It contains the `id` and `label` for each image in the test set[cite: 1].

### Evaluation (on Development/Validation Set)
* The training notebook (`CornLeaf_EfficientNetV2B2_TrainingCode.ipynb`) includes code for evaluating the model on the validation set (Cell 13).
* **Key Validation Metrics (from best model during training):**
    * **Validation F1-Score (Macro): 0.9409**
    * Validation F1-Score (Weighted): 0.9516
    * Validation Accuracy: 0.9509
    * Validation Loss: 0.1550
    * See Cell 13 of 'CornLeaf_EfficientNetV2B2_TrainingCode.ipynb' for the classification report 

---

## Solution 2: Deep Learning with Transformer (DeiT or Data-efficient Image Transformer)

### Approach

Our second solution explores a transformer-based architecture to tackle the classification task, specifically using the **DeiT (Data-efficient Image Transformer)** model. This approach was implemented using PyTorch and the `timm` library.

1.  **Loading a Pre-trained Model:** We used the `deit_tiny_patch16_224` model, which was pre-trained on ImageNet. DeiT is specifically designed to perform well without requiring massive pre-training datasets, making it suitable for our moderately-sized corn leaf dataset.
2.  **Model Adaptation:** The model's original classification head was replaced with a new `nn.Linear` layer to match our 4 output classes (Blight, Common Rust, Gray Leaf Spot, Healthy).
3.  **Fine-Tuning:** The entire model was fine-tuned on the corn leaf dataset. We used the **AdamW optimizer** (learning rate 1e-4, weight decay 0.05), which is well-suited for training transformer models.
4.  **Handling Class Imbalance:** To address the imbalanced dataset, we calculated class weights based on the inverse frequency of each class. These weights were passed to the **CrossEntropyLoss** function, ensuring that the model paid more attention to the minority class (Gray Leaf Spot).
5.  **Callbacks for Robust Training:** The training loop was enhanced with callbacks to ensure robustness and find the best-performing model:
    * **Early Stopping:** Training was monitored based on the **validation F1 Macro score**. The process would stop if the score did not improve for 10 consecutive epochs, and the weights of the best-performing epoch were saved.
    * **`ReduceLROnPlateau`:** The learning rate was dynamically reduced by a factor of 0.1 if the validation F1 Macro score did not improve for 5 consecutive epochs.

### Code Structure

The code for the DeiT solution is organized into the following Jupyter Notebooks:

1.  **`CornLeaf_DeiT_TrainingCode.ipynb`**:
    * This notebook contains the complete PyTorch code for loading and augmenting the data, defining the DeiT model, and fine-tuning it.
    * It implements the custom training loop with class weighting, F1-score calculation, early stopping, and a learning rate scheduler.
    * **To run training:** Execute all cells in this notebook sequentially. Ensure the `dataset_path` in Cell 3 points to your training data. The best model's `state_dict` is saved to `/content/drive/MyDrive/CornLeaf_DeiT.pth` (path can be adjusted in Cell 7).

2.  **`CornLeaf_DeiT_MakePredictions.ipynb`**:
    * This notebook loads the trained DeiT model (`.pth` file) and generates predictions for the test set.
    * It produces a `predictions_DeiT.csv` file with `id` and `label` columns, which was submitted for this solution.
    * **To run for generating predictions:**
        1.  Ensure the `MODEL_PATH` in Cell 2 points to the saved `.pth` model file.
        2.  Ensure the `TEST_ZIP_PATH` in Cell 2 points to the `corn_test.zip` file.
        3.  Execute all cells sequentially.

3.  **`CornLeaf_DeiT_DemoCode.ipynb`**:
    * This notebook provides a real-time demo for the DeiT solution.
    * It loads the trained model, allows you to specify an image from the test set, and then displays the image with its predicted class and confidence score.
    * **To run the demo:**
        1.  Ensure `MODEL_PATH_DEMO` in Cell 2 points to the saved `.pth` model file.
        2.  Ensure `TEST_ZIP_PATH_DEMO` in Cell 2 points to the `corn_test.zip` file.
        3.  In `DEMO CELL 6`, change the `image_filename_to_demo` variable to test different images.
        4.  Execute all cells sequentially.

### Model Storage

* The trained DeiT model's state dictionary (`CornLeaf_DeiT.pth`) is stored in Google Drive.
* **Link to Model:** https://drive.google.com/file/d/1KjsA55WdKIfXelCU86rvTgEExYdLkR13/view?usp=sharing

### Predictions File

* The `predictions_DeiT.csv` file was generated using the `CornLeaf_DeiT_MakePredictions.ipynb` notebook. It contains the predictions for the test set from this transformer-based model.

### Evaluation (on Development/Validation Set)

* The training notebook (`CornLeaf_DeiT_TrainingCode.ipynb`) evaluates the model on the validation set at the end of each epoch and prints a final classification report.
* **Key Validation Metrics (from best model):**
    * **Validation F1-Score (Macro): 0.9623**
    * Validation F1-Score (Weighted): 0.9683
    * Validation Accuracy: 0.9683
    * Validation Loss: 0.1608
    * See Cell 7 of 'CornLeaf_DeiT_TrainingCode.ipynb' for the classification report 

---

## Attribution

* **Dataset:** Corn Leaf Disease dataset provided by the CS 180 course instructors.
* **Pre-trained Models & Frameworks:**
    * TensorFlow and Keras Applications for the EfficientNetV2B2 model.
    * PyTorch framework and the `timm` (PyTorch Image Models) library for the DeiT model.
* **Code Bases:** Primarily our own implementation based on course lectures and standard library documentation
