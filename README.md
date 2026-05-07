# XRay_Segmentation_and_Diagnosis
Deep Learning based Lung X-Ray segmentation system and TBC diagnosis using custom U-Net architecture and transfer learning


This project is based on the dataset available on Kaggle containing images of X-Rays and segmentation masks for both normal and TBC-positive cases. 

Link:
https://www.kaggle.com/datasets/iamtapendu/chest-x-ray-lungs-segmentation?resource=download

The Project is structured as follows:

XRay_Segmentation_and_Diagnosis/  
|  
|- Data/  
|- Exploratory_Analysis/  
|- Dataset/  
|- Segmentation/  
|- Classification/  
|- Models/  
|- Results/  
|- Images/  

1. |- Data/ : The folder "Data/" contains MetaData.csv file that is a table with 6 columns and 704 rows. The columns are "id", "gender", "age", "county", "ptb" and "remarks".  
In order to run the scripts with minimum edit, it is necessary to include in "Data/" the dataset folder available for download at the link provided (~3.8 GB) containing images and masks, and name it "ChestXRay". After doing this, it is necessary just to specify the project root directory at the beginning of the scripts.

2. |- Exploratory_Analysis/ : this folder contains one script for descriptive statistics of the dataset and one for visualizing some of the images/masks pairs.

3. |- Dataset/ : contains the dataset classes for both the segmentation and classification tasks. Both inherit the structure of PyTorch Dataset. The dataset for segmentation returns image-mask couples, resized to a user specified size, and perform train/val/test splitting. The dataset for classification of TBC-positive images is more sophisticated, since it performs stratified splitting in order to handle unbalanced classes, masking of the images for allowing the model to just see the lungs region and data augmentation to enhance robustness; in the end it returns the augmented image and the corresponding label (0/1).

4. |- Segmentation/ : here is stored the segmentation model class and the training script for this model. The architecture for this image binary segmentation is U-Net based, consisting in a convolutional encoder-decoder pair with skip connections. The training routine is based on early stopping method, monitoring the validation performance metrics like pixel-accuracy, intersection over union (IoU) and Dice score.

5. |- Classification/ : for this task I explored two different approaches, a custom CNN consisting in a convolutional encoder plus fully connected classifier head and a Transfer Learning (TL) method with the ResNet18 backbone pre-trained on ImageNet1K followed by a "Squeeze-and-Excitation" (SE) block and a fully connected binary classifier head. The training routine follows the same early stopping logic as the segmentation task, monitoring the validation confusion matrix during training. The custom CNN is trained completely while for the TL approach the first three layers of the backbone are frozen.

6. |- Results/ :  In this folder there are the scripts for testing the models and assessing the performance over the test set. The segmentation model reaches an astonishing average metrics above 99%:
   
Test Set Metrics | Accuracy = 0.998 | IoU = 0.991 | Dice = 0.996

While the classification models score both above 80% in detecting TBC cases:  
Custom CNN: Parameters = 1403521  
Test Accuracy : 0.802 | Precision : 0.86 | Recall : 0.71 | F1 : 0.78  

ResNet18 Backbone : Parameters = 11236449  
Test Accuracy : 0.858 | Precision : 0.86 | Recall : 0.846 | F1 : 0.85 

8. |- Images/ : Contains images used in this description.



