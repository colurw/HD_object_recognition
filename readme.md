# HD_object_recognition 

Uses data augmentation and supervised transfer learning techniques
to categorise individual features found in HD images.  It's written in Ipython, using
Jupyter Notebooks, to be run on Google Colaboratory.

## Transfer Learning

Training convolutional neural networks (CNNs) to recognise images takes a lot of time
and training data.  This is less than ideal, as images typically need to be hand-
labelled when dealing with niche applications.  Fortunately we can take a pre-trained
model, and build in the extra internal representations that we need, to be able to 
make it distinguish the objects that we need.  This is known as transfer learning.  
In this case two additional object categories were taught to the model.

## Image Segmentation

This project uses the Mask R-CNN architecture with a model pre-trained on the MS COCO 
(Microsoft Common Objects in Context) dataset.  Mask R-CNN is able to categorise 
multiple types and/or instances of objects in a single photograph, a feature known
as image segmentation.  The training data required for this consists of image files 
paired with XML files, which contain the bounding-box coordinates and category labels 
for entities found in the image.

Segmenting HD images either uses large amounts of RAM or requires us to reduce the 
image resolution, which is not ideal when working with images containing small 
features.  A (possibly) novel third way is explored here, by splitting the image 
(and any asssociated XML label file) into tiles, then reassembling them after 
analysis by the network.

## Data Augmentation

Data augmentation is a range of techniques to overcome the problems caused by a 
limited amount of training data.  Images can be mirrored, rotated, skewed, or have 
noise added, to provide many more almost-unique training examples that boost the
CNN's ability to learn.  

## Ipython Notebooks

### 1_tiling_and_augmentation.ipynb  

Breaks the images and label data down into smaller 
image tiles, and a group of these are separated out for use as a test dataset.
Running all the cells combines a series of flips and rotations to grow our original
training data by 12 multiples, with many more possible by adding extra rotations.  

### 2_mcrnn_training.ipynb  

Organises and checks these training data, then feeds them into the 
Mask R-CNN / MSCOCO model.  Different layers of the model can be selected to be 
updated through back-propogation.  Finally the ability of the model to detect the 
new feature categories is assessed by comparing its predictions against labelled
images.

### 3_image_segmentation.ipynb  

Allows the use of this re-trained model to categorise objects found
in previously-unseen HD photos.  It uses the same tiling procedure as before to 
send a batchof six image tiles to the model, then returns an .xml file the categories 
and absolute coordinates of any objects found, with respect to to the original HD photo.



