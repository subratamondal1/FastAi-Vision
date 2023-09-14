# FASTAI COMPUTER VISION

# P1: 🐶🐱Cat & Dog breed detector🐕🐈

## Project Description
In this deep learning project, we have developed a sophisticated Cat & Dog breed detector using the state-of-the-art Fastai framework. Our goal was to accurately classify pet images into 37 distinct categories, representing various breeds. We utilized the challenging Oxford-IIIT Pet Dataset, which contains a diverse set of pet images with complex variations in scale, pose, and lighting.

## FastAi Computer Vision Pipeline for Cat & Dog breed detector

### Load Data 
- The project focuses on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), known for its 37 distinct pet categories and high-quality images.
- Essential libraries, including FastAi, are imported to support the computer vision project.
- The FastAi version used in this code is 2.7.12.
- The FastAi vision package is imported to leverage its modules and functions for computer vision tasks.
- The `get_image_files` function is imported to facilitate the retrieval of image file paths.
- The Oxford-IIIT Pet Dataset, accessible through the provided URL, is downloaded and untarred for data preparation.
- A list of image file names residing in the 'images' directory of the dataset is obtained using the `get_image_files` function.
- The code prints the total count of images in the dataset, which is 7390 images.
- Additionally, it displays the file paths of the first 10 images in the dataset, providing a glimpse of the data's structure and location.

### Data Preparation
- **Statistical Normalization**: The code imports essential statistics from the FastAi library, specifically, the `imagenet_stats`. These statistics are widely utilized for image data normalization, a fundamental step in preparing data for deep learning models.

- **Image Augmentation and Cropping**: The code incorporates critical image transformation functions from FastAi, such as `aug_transforms` and `RandomResizedCrop`. These transformations are pivotal for introducing variability and robustness into the dataset, enhancing the model's ability to generalize from the training data to unseen examples.

- **Item Transforms**: The `item_tfms` variable is defined to specify item-level transformations. In this case, it involves a random resized crop operation with dimensions of 460 pixels. This transformation not only resizes images but also introduces slight variability in scale, enhancing the dataset's diversity.

- **Batch Transforms**: A list of batch-level transformations, denoted by the `batch_tfms` variable, is defined. This list encapsulates a combination of augmentations, such as resizing to 224 pixels and maximum warping, along with data normalization using the previously imported `imagenet_stats`. These batch transforms collectively prepare the data for efficient training while maintaining statistical consistency.

By executing these meticulously designed data preparation steps, the code ensures that the dataset is suitably conditioned for subsequent phases of the computer vision pipeline. This proactive approach to data preprocessing lays the groundwork for model training, validation, and evaluation, ultimately contributing to the success of the computer vision project.

### Create DataLoader
- **DataBlock Definition**: The core of the section lies in the creation of a DataBlock named 'pets.' This DataBlock orchestrates the entire data processing pipeline, defining key aspects such as the data blocks (Image and Category) to be employed, the method for acquiring image files, data splitting into training and validation sets using random splitting, and the extraction of category labels from file names using regular expressions.

- **Item-Level and Batch-Level Transformations**: The DataBlock is configured with item-level and batch-level transformations, denoted as `item_tfms` and `batch_tfms`, respectively. These transformations, previously defined, ensure that each image and batch of data undergoes the prescribed operations, including resizing, cropping, augmentations, and normalization. This enhances the dataset's suitability for model training while maintaining consistency.

- **Data Loaders Creation**: The code concludes by creating data loaders, designated as 'dls,' utilizing the 'pets' DataBlock. These data loaders are responsible for efficiently loading and batching the data, readying it for training and validation. Specifically, the training dataset, as represented by `dls.train_ds`, contains 5912 images, while the validation dataset, as represented by `dls.valid_ds`, contains 1478 images. A batch size of 64 is specified for the data loaders.

- **Data Set Overview**: The code offers a glimpse into the training and validation datasets by printing a sample of their elements. Each element consists of a PIL image and its corresponding category label, indicative of the transformed and labeled data's readiness for model ingestion.

By executing these meticulously defined data processing steps, the "Create DataLoader" section ensures that the dataset is properly structured, labeled, and preprocessed for utilization in subsequent model development phases. The resulting data loaders stand as the conduit through which the model will access and learn from the prepared data.


4. **Data Loading**:
   - Create efficient data loaders with careful consideration of batch sizes.
   - Ensure class balance in dataset splitting and label extraction.

5. **Model Configuration**:
   - Construct a tailored vision model considering dataset complexity.
   - Leverage transfer learning with a pre-trained model for efficient training.
   - Define appropriate evaluation metrics, emphasizing accuracy and error rate.

6. **Initial Training**:
   - Determine optimal learning rates, balancing model stability and convergence.
   - Train the model for an appropriate number of epochs, achieving an initial accuracy of nearly 94% and a low error rate.

7. **Model Fine-Tuning**:
   - Unfreeze model layers to fine-tune for improved accuracy.
   - Find suitable learning rates for fine-tuning.
   - Fine-tune the model, maintaining high accuracy while reducing error rate.

8. **Model Validation**:
   - Evaluate model performance using professional metrics, including an accuracy exceeding 94% and a remarkably low error rate.
   - Highlight significant improvements achieved through fine-tuning.

9. **Inference**:
   - Load the finely-tuned model for accurate predictions.
   - Prepare input images following model preprocessing.
   - Execute inference, providing not only the predicted pet category but also confidence scores.

This streamlined pipeline underscores proficiency in handling data, applying advanced preprocessing, configuring models, and achieving remarkable accuracy. It delivers a professional and efficient approach to image classification on the Oxford-IIIT Pet Dataset, showcasing your expertise in deep learning and model refinement.

### Conclusion
This project showcases our proficiency in deep learning, data preprocessing, model training, and evaluation. The pet classifier we've developed delivers impressive results, accurately identifying pet breeds from a challenging dataset. Our pipeline can be adapted for various image classification tasks, and the model's ability to handle complex, real-world data demonstrates its practical utility.

**Model Performance Metrics (After Fine-Tuning):**
- Accuracy: Exceeding 94%
- Error Rate: Remarkably low
