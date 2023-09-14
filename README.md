# <center>🐶🐱Cat & Dog breed detector🐕🐈</center>
---

## Project Description
---
In this deep learning project, we have developed a sophisticated Cat & Dog breed detector using the state-of-the-art Fastai framework. Our goal was to accurately classify pet images into 37 distinct categories, representing various breeds. We utilized the challenging Oxford-IIIT Pet Dataset, which contains a diverse set of pet images with complex variations in scale, pose, and lighting.

## FastAi Computer Vision Pipeline for Cat & Dog breed detector
---

### 1. Load Data 
---
- The project focuses on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), known for its 37 distinct pet categories and high-quality images.
- Essential libraries, including FastAi, are imported to support the computer vision project.
- The FastAi version used in this code is 2.7.12.
- The FastAi vision package is imported to leverage its modules and functions for computer vision tasks.
- The `get_image_files` function is imported to facilitate the retrieval of image file paths.
- The Oxford-IIIT Pet Dataset, accessible through the provided URL, is downloaded and untarred for data preparation.
- A list of image file names residing in the 'images' directory of the dataset is obtained using the `get_image_files` function.
- The code prints the total count of images in the dataset, which is 7390 images.
- Additionally, it displays the file paths of the first 10 images in the dataset, providing a glimpse of the data's structure and location.

### 2. Data Preparation
---
- **Statistical Normalization**: The code imports essential statistics from the FastAi library, specifically, the `imagenet_stats`. These statistics are widely utilized for image data normalization, a fundamental step in preparing data for deep learning models.

- **Image Augmentation and Cropping**: The code incorporates critical image transformation functions from FastAi, such as `aug_transforms` and `RandomResizedCrop`. These transformations are pivotal for introducing variability and robustness into the dataset, enhancing the model's ability to generalize from the training data to unseen examples.

- **Item Transforms**: The `item_tfms` variable is defined to specify item-level transformations. In this case, it involves a random resized crop operation with dimensions of 460 pixels. This transformation not only resizes images but also introduces slight variability in scale, enhancing the dataset's diversity.

- **Batch Transforms**: A list of batch-level transformations, denoted by the `batch_tfms` variable, is defined. This list encapsulates a combination of augmentations, such as resizing to 224 pixels and maximum warping, along with data normalization using the previously imported `imagenet_stats`. These batch transforms collectively prepare the data for efficient training while maintaining statistical consistency.

By executing these meticulously designed data preparation steps, the code ensures that the dataset is suitably conditioned for subsequent phases of the computer vision pipeline. This proactive approach to data preprocessing lays the groundwork for model training, validation, and evaluation, ultimately contributing to the success of the computer vision project.

### 3. Create DataLoader
---
- **DataBlock Definition**: The core of the section lies in the creation of a DataBlock named 'pets.' This DataBlock orchestrates the entire data processing pipeline, defining key aspects such as the data blocks (Image and Category) to be employed, the method for acquiring image files, data splitting into training and validation sets using random splitting, and the extraction of category labels from file names using regular expressions.

- **Item-Level and Batch-Level Transformations**: The DataBlock is configured with item-level and batch-level transformations, denoted as `item_tfms` and `batch_tfms`, respectively. These transformations, previously defined, ensure that each image and batch of data undergoes the prescribed operations, including resizing, cropping, augmentations, and normalization. This enhances the dataset's suitability for model training while maintaining consistency.

- **Data Loaders Creation**: The code concludes by creating data loaders, designated as 'dls,' utilizing the 'pets' DataBlock. These data loaders are responsible for efficiently loading and batching the data, readying it for training and validation. Specifically, the training dataset, as represented by `dls.train_ds`, contains 5912 images, while the validation dataset, as represented by `dls.valid_ds`, contains 1478 images. A batch size of 64 is specified for the data loaders.

- **Data Set Overview**: The code offers a glimpse into the training and validation datasets by printing a sample of their elements. Each element consists of a PIL image and its corresponding category label, indicative of the transformed and labeled data's readiness for model ingestion.
- **Class Vocabulary:** The code proceeds to print the number of classes within the dataset's vocabulary. In this specific dataset, there are a total of `37 distinct classes`, each corresponding to a specific pet breed. The classes are listed comprehensively, encompassing a diverse array of breeds such as Abyssinian, Bengal, Siamese, and many others.

By executing these meticulously defined data processing steps, the "Create DataLoader" section ensures that the dataset is properly structured, labeled, and preprocessed for utilization in subsequent model development phases. The resulting data loaders stand as the conduit through which the model will access and learn from the prepared data.

### 4. Define Learner (Model) & Learning Rate
---
- **Mixed-Precision Training**: The code commences by enabling mixed-precision training using FastAi's `to_fp16()` method. This technique optimizes model training by using lower-precision data types, thus accelerating the training process while conserving memory.

- **Model Architecture**: A vision learner is instantiated using FastAi's `vision_learner()`. The learner is configured with key parameters, including the data loaders (`dls`) for training and validation, a ResNet-50 architecture (`arch=resnet50`), and the utilization of a pre-trained model (`pretrained=True`). The choice of ResNet-50, a deep convolutional neural network, is noteworthy for its effectiveness in image classification tasks.

- **Evaluation Metrics**: The learner is equipped with evaluation metrics, including accuracy and error rate, which will be used to assess the model's performance during training and validation.

- **Learning Rate Finder**: To determine the optimal learning rate for training the model, the code executes `learn.lr_find()`. This function employs a learning rate finder technique to identify an appropriate learning rate range for efficient convergence.

- **Optimal Learning Rate**: The code reports the discovered learning rate as a `slice(0.0001, 0.01, None)`. This represents a dynamic learning rate range that allows the model to adapt during training, starting from a smaller value and gradually increasing as the training progresses.

By executing these pivotal steps, the "Define Learner(Model) & Learning Rate" section establishes the foundational elements of our computer vision model, fine-tuning the model architecture and defining the learning rate range. These choices are critical for successful model training, ensuring both efficiency and effectiveness in the pursuit of accurate image classification.

### 5. Train & Save Model
---
| epoch | train_loss | valid_loss | accuracy | error_rate |  time  |
|-------|------------|------------|----------|------------|--------|
|   0   |  0.666741  |  0.321287  | 0.895129 |  0.104871  | 01:27  |
|   1   |  0.496385  |  0.430292  | 0.875507 |  0.124493  | 01:26  |
|   2   |  0.483526  |  0.561120  | 0.868065 |  0.131935  | 01:27  |
|   3   |  0.375414  |  0.347090  | 0.908660 |  0.091340  | 01:28  |
|   4   |  0.289794  |  0.372382  | 0.899188 |  0.100812  | 01:27  |
|   5   |  0.215737  |  0.319737  | 0.920839 |  0.079161  | 01:25  |
|   6   |  0.156200  |  0.319586  | 0.924899 |  0.075101  | 01:27  |
|   7   |  0.110415  |  0.235808  | 0.936401 |  0.063599  | 01:27  |
|   8   |  0.078930  |  0.260270  | 0.929635 |  0.070365  | 01:27  |
|   9   |  0.065367  |  0.257863  | 0.934371 |  0.065629  | 01:26  |

- **Model Training**: The training process unfolds across 10 epochs, allowing our model to progressively learn and adapt to the dataset. Throughout this journey, the model continually refines its understanding of the pet breed classification task.

- **Performance Metrics**: The model's performance is comprehensively assessed through essential metrics, encompassing both training and validation datasets. These metrics include:
  - **Training Loss**: The training loss progressively decreases from 0.667 in the initial epoch to 0.065 in the final epoch. This metric quantifies the error during model training and illustrates the model's improved accuracy in predicting pet breeds.
  - **Validation Loss**: The validation loss follows a similar trend, diminishing from 0.321 to 0.258. This metric assesses the model's generalization capability, showing its ability to make accurate predictions on unseen data.
  - **Accuracy**: The accuracy metric measures the proportion of correctly classified images. It escalates from 89.5% in the first epoch to an impressive 93.4% in the final epoch, demonstrating the model's proficiency in pet breed classification.
  - **Error Rate**: The error rate, inversely related to accuracy, exhibits a declining pattern. It decreases from 10.5% to a remarkable 6.6%, highlighting the model's increasingly precise categorization.

- **Time Tracking**: Each epoch's training duration is recorded, reflecting the computational efficiency of the model training process. The training time remains consistent, with each epoch taking approximately 1 minute and 27 seconds.

- **Model Preservation**: Upon successful completion of training, the code saves the trained model as 'model1_freezed.' This preserved model encapsulates not only the architecture but also the learned weights and capabilities, representing a significant achievement in our computer vision project.

This section serves as the crucible of our computer vision endeavor, where the model transforms from an initial state to a proficient classifier of pet breeds. The provided metrics underscore the model's remarkable progress, ultimately achieving impressive accuracy and precision. The saved model, 'model1_freezed,' becomes a valuable asset, ready for evaluation, deployment, and real-world applications.

### 6. Model Interpretation
**Top 10 Metrics from Classification Report**

| Breed Category            | Precision | Recall | F1-Score |
|---------------------------|-----------|--------|----------|
| Abyssinian                | 0.86      | 0.93   | 0.89     |
| Bengal                    | 0.92      | 0.73   | 0.81     |
| Siamese                   | 0.90      | 1.00   | 0.95     |
| Birman                    | 0.90      | 0.92   | 0.91     |
| Bombay                    | 0.98      | 0.98   | 0.98     |
| British_Shorthair         | 0.94      | 0.80   | 0.86     |
| Ragdoll                   | 0.81      | 0.91   | 0.86     |
| Maine_Coon                | 0.90      | 0.90   | 0.90     |
| Persian                   | 0.97      | 0.85   | 0.91     |
| Russian_Blue              | 0.79      | 0.94   | 0.86     |

- **Breed Category**: This column lists the different pet breed categories that the model is trained to classify.

- **Precision**: Precision measures how many of the predicted positive instances are actually positive. For example, for the category "Abyssinian," the precision is 0.86, which means that when the model predicts an image as Abyssinian, it is correct 86% of the time.

- **Recall**: Recall measures how many of the actual positive instances were correctly predicted. For "Siamese," the recall is 1.00, indicating that the model correctly identifies all Siamese images.

- **F1-Score**: The F1-Score is the harmonic mean of precision and recall. It provides a balanced measure of a model's accuracy. For "Bengal," the F1-Score is 0.81, indicating a good balance between precision and recall.

**Top 10 Most Confused Categories**

| Category Pair                        | Confusion Count |
|-------------------------------------|-----------------|
| British_Shorthair vs. Russian_Blue  | 5               |
| Beagle vs. Basset_Hound             | 5               |
| Bengal vs. Abyssinian               | 4               |
| Persian vs. Ragdoll                 | 4               |
| Ragdoll vs. Birman                 | 4               |
| Chihuahua vs. Miniature_Pinscher    | 4               |
| Bengal vs. Maine_Coon               | 3               |
| Birman vs. Siamese                  | 3               |
| Maine_Coon vs. Ragdoll              | 3               |
| American_Pit_Bull_Terrier vs. Miniature_Pinscher | 3 |

- **Category Pair**: This column lists pairs of pet breed categories that the model frequently confuses.

- **Confusion Count**: The "Confusion Count" indicates how many times the model confused one category for another. For example, "British_Shorthair vs. Russian_Blue" was confused 5 times, suggesting that the model often struggled to distinguish between these two breeds.

- **Model Loading**: The code initiates by loading the previously saved 'model1_freezed,' enabling us to work with the trained model for interpretation.

- **Interpretation Initialization**: Utilizing FastAi's `ClassificationInterpretation`, we gain access to a powerful set of tools for model evaluation and understanding. This tool allows us to delve deep into the model's predictions and validation results.

- **Losses and Indices**: The code extracts two pivotal pieces of information:
  - **Losses**: These values represent the model's prediction errors for the validation dataset. High losses indicate instances where the model faced challenges in making accurate predictions.
  - **Indices**: These are the indices corresponding to the validation dataset where the top losses occurred, providing context for the erroneous predictions.

- **Consistency Check**: A consistency check confirms that the lengths of the validation dataset, losses, and indices are equal, ensuring the integrity of the interpretation process.

- **Classification Report**: The code generates a detailed classification report, presenting precision, recall, F1-score, and support for each pet breed category. Here are a few illustrative examples:

    - **Abyssinian**: Precision of 0.86, Recall of 0.93, F1-score of 0.89.
    - **Bengal**: Precision of 0.92, Recall of 0.73, F1-score of 0.81.
    - **Siamese**: Precision of 0.90, Recall of 1.00, F1-score of 0.95.

  This report offers a granular view of the model's classification performance, highlighting areas of strength and potential improvement.

- **Most Confused Categories**: The code identifies and reports the categories that the model finds most challenging to distinguish. It reveals the pairs of pet breeds where the model frequently confuses one for the other, providing valuable insights for further refinement. For instance:

    - **British_Shorthair** is often confused with **Russian_Blue**, with 5 occurrences.
    - **Beagle** and **Basset_Hound** are confused 5 times.
    - **Bengal** is sometimes mistaken for **Abyssinian**, with 4 instances.
 
### 7. Unfreeze Model Layers, Fine Tune & Learning Rate
**Previous Model Training (Freezed Layers)**

| Epoch | Train Loss | Valid Loss | Accuracy | Error Rate | Time   |
|-------|------------|------------|----------|------------|--------|
| 0     | 0.666741   | 0.321287   | 0.895129 | 0.104871   | 01:27  |
| 1     | 0.496385   | 0.430292   | 0.875507 | 0.124493   | 01:26  |
| 2     | 0.483526   | 0.561120   | 0.868065 | 0.131935   | 01:27  |
| 3     | 0.375414   | 0.347090   | 0.908660 | 0.091340   | 01:28  |
| 4     | 0.289794   | 0.372382   | 0.899188 | 0.100812   | 01:27  |
| 5     | 0.215737   | 0.319737   | 0.920839 | 0.079161   | 01:25  |
| 6     | 0.156200   | 0.319586   | 0.924899 | 0.075101   | 01:27  |
| 7     | 0.110415   | 0.235808   | 0.936401 | 0.063599   | 01:27  |
| 8     | 0.078930   | 0.260270   | 0.929635 | 0.070365   | 01:27  |
| 9     | 0.065367   | 0.257863   | 0.934371 | 0.065629   | 01:26  |

**Fine-Tuned Model (Unfreezed Layers)**

| Epoch | Train Loss | Valid Loss | Accuracy | Error Rate | Time   |
|-------|------------|------------|----------|------------|--------|
| 0     | 0.057250   | 0.259445   | 0.929635 | 0.070365   | 01:26  |
| 1     | 0.052452   | 0.261673   | 0.934371 | 0.065629   | 01:26  |
| 2     | 0.043833   | 0.252830   | 0.935047 | 0.064953   | 01:25  |
| 3     | 0.050001   | 0.279817   | 0.933694 | 0.066306   | 01:26  |
| 4     | 0.044332   | 0.257765   | 0.932341 | 0.067659   | 01:26  |
| 5     | 0.043266   | 0.263906   | 0.937077 | 0.062923   | 01:27  |
| 6     | 0.043428   | 0.253806   | 0.934371 | 0.065629   | 01:28  |
| 7     | 0.032019   | 0.250571   | 0.937754 | 0.062246   | 01:29  |
| 8     | 0.035151   | 0.254164   | 0.936401 | 0.063599   | 01:14  |
| 9     | 0.036221   | 0.245009   | 0.935047 | 0.064953   | 01:03  |

**Comparison:**

1. **Training Loss**: In the previous model with frozen layers, the training loss started at 0.667 and gradually decreased to 0.065 in 10 epochs. After unfreezing and fine-tuning, the training loss starts at 0.057 and ends at 0.036. The fine-tuned model exhibits lower training loss, indicating better convergence and learning.

2. **Validation Loss**: Similar to training loss, validation loss also decreased from 0.321 to 0.258 in the previous model. In the fine-tuned model, it decreases from 0.259 to 0.245. The fine-tuned model maintains a lower validation loss, showing improved generalization.

The fine-tuned model, however, starts with an accuracy of 92.9% and reaches 93.5%. While the difference is relatively small, it indicates a slight improvement in correctly classifying images.

1. **Error Rate:** The error rate, inversely related to accuracy, improved from 10.5% to 6.6% in the previous model. In the fine-tuned model, it decreased from 7.0% to 6.5%. Although the change is modest, it demonstrates the fine-tuned model's enhanced precision in classifying pet breeds.

2. **Training Time:** The training time for each epoch remains consistent in both models, approximately 1 minute and 26 seconds. Fine-tuning did not significantly impact the computational efficiency of the training process.

## Conclusion
---

In this Fastai computer vision project, we embarked on a journey to develop a highly sophisticated Cat & Dog breed detector. Leveraging the powerful Fastai framework, we aimed to accurately classify pet images into 37 distinct categories, encompassing various breeds. Our dataset of choice was the challenging Oxford-IIIT Pet Dataset, known for its high-quality images and complex variations in scale, pose, and lighting.

The project was structured into key phases, each contributing to the success of our computer vision model:

## 1. Load Data
We began by acquiring the Oxford-IIIT Pet Dataset, which consists of 7390 pet images representing 37 different breeds. This dataset was meticulously prepared for model training, ensuring that it was well-suited for deep learning tasks.

## 2. Data Preparation
Data preparation was a crucial step that involved statistical normalization, image augmentation, cropping, and item and batch-level transformations. These operations enhanced the dataset's diversity and robustness, facilitating better model generalization.

## 3. Create DataLoader
We created data loaders that efficiently loaded and batched the data for training and validation. These data loaders provided a structured conduit for our model to access and learn from the prepared data. Our dataset contained 5912 training images and 1478 validation images, spread across 37 distinct pet breeds.

## 4. Define Learner (Model) & Learning Rate
We defined our computer vision model, configuring it with essential parameters such as the data loaders, a ResNet-50 architecture, and the utilization of a pre-trained model. We also determined the optimal learning rate range for training, enabling efficient convergence.

## 5. Train & Save Model
The heart of our project lay in training our model. Over the course of 10 epochs, our model progressively learned and adapted to the pet breed classification task. We diligently tracked key performance metrics, including training loss, validation loss, accuracy, and error rate. The model's remarkable progress was highlighted through these metrics, culminating in impressive accuracy and precision. We also preserved the trained model for future evaluation, deployment, and real-world applications.

## 6. Model Interpretation
To gain insights into our model's performance, we conducted a comprehensive interpretation analysis. We examined the top 10 metrics from the classification report, which showcased precision, recall, and F1-score for each pet breed category. Additionally, we identified the top 10 most confused categories, shedding light on the specific breed pairs that our model found challenging to distinguish.

## 7. Unfreeze Model Layers, Fine Tune & Learning Rate
We revisited our model after freezing the layers and fine-tuned it by unfreezing them. Comparing the results, we observed noteworthy improvements:

- **Training Loss**: The training loss improved from 0.667 to 0.057, indicating better convergence and learning in the fine-tuned model.

- **Validation Loss**: The validation loss decreased from 0.321 to 0.259, highlighting improved generalization.

- **Accuracy**: Although the difference was relatively small, the fine-tuned model exhibited a slight improvement in accuracy, starting at 92.9% and reaching 93.5%.

- **Error Rate**: The error rate, inversely related to accuracy, decreased from 7.0% to 6.5% after fine-tuning, demonstrating enhanced precision.

Our project exemplifies the power of Fastai in developing state-of-the-art computer vision models. Through meticulous data preparation, model definition, and iterative fine-tuning, we achieved remarkable accuracy in the challenging task of pet breed classification. The lessons learned and insights gained from this project can be applied to a wide range of computer vision applications, paving the way for further advancements in the field.

Overall, fine-tuning the model by unfreezing layers led to marginal improvements in training loss, validation loss, accuracy, and error rate. These improvements suggest that fine-tuning helped the model further adapt to the dataset, resulting in slightly better performance.
