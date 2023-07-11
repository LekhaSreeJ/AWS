# AWS
Brain tumor segmentation

The segmentation_notebook_in_local_jupyter contains the ipynb file of a brain tumor segementation project run on a local jupyter notebook using the ResUNet model for 100 epochs. 
The dataset used for this project is available in the Kaggle - https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

The aws_brain_tumor_segmentation consists of the same project tried by running in the aws sagemaker using tensorflow version of the server image and training the model by uploading the sample of 30 datasets onto the s3 bucket. 

The brain_tumor_seg_project .py file provides the data loader , model and evaluation that is used in the tf.estimator part of the aws ipynb notebook for the model execution purpose. 
