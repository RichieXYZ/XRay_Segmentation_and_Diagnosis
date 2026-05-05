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
|- Segmentation/  
|- Classification/  
|- Models/  
|- Results/  
|- Images/  

The folder "Data/" contains MetaData.csv file that is a table with 6 columns and 704 rows. The columns are "id", "gender", "age", "county", "ptb" and "remarks".  
In order to run the scripts with minimum edit, it is necessary to include in "Data/" the dataset folder available for download at the link provided (~3.8 GB) containing images and masks, and name it "ChestXRay". After doing this, it is necessary just to specify the project root directory at the beginning of the scripts.

