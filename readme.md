# HD_object_recognition 

Uses data augmentation and supervised transfer learning techniques
to categorise individual features found in HD images.  It is written in Ipython, using
Jupyter Notebooks, to be run on Google Colaboratory. 

## Transfer Learning

Training convolutional neural networks (CNNs) to recognise images takes a lot of time
and vast amounts of training data.  This is less than ideal, as images typically need to be hand labelled.  

Fortunately we can take a pre-trained model, and build in the extra internal representations that we need, to be able to 
make it distinguish the objects that we need.  This is known as transfer learning.  For this project two additional object categories were taught to the model.

## Image Segmentation

This project uses the Mask R-CNN architecture with a model pre-trained on the MS COCO 
(Microsoft Common Objects in Context) dataset.  Mask R-CNN can categorise 
multiple types and/or instances of objects in a single photograph, a feature known
as image segmentation.  The training data required for this consists of image files 
paired with XML files, which contain the bounding-box coordinates and category labels 
for entities found in the image.

Segmenting (labelling the objects in) HD images either uses large amounts of RAM, or requires us to reduce the 
image resolution, which is not ideal when working with images containing small 
features.  A third way is explored in this code, by splitting the image 
(and any asssociated XML label file) into tiles, then reassembling them after 
analysis by the network.

## Data Augmentation

Data augmentation is a range of techniques utilised to overcome the problems caused by a 
limited amount of training data.  Images can be mirrored, rotated, skewed, or have 
noise added, to provide many additional almost-unique training examples that boost the
CNN's ability to recognise the new objects.  

## Ipython Notebooks

### 1_tiling_and_augmentation.ipynb  

Breaks the images and label data down into smaller 
image tiles, and a group of these are separated out for use as a test dataset.
Running all the cells combines a series of flips and rotations to grow the original
training data by a multiple of 12 (with more multiplications possible by adding extra rotation steps).  

### 2_mcrnn_training.ipynb  

Organises and checks these training data, then feeds them into the 
Mask R-CNN / MSCOCO model.  Different layers of the model can be selected to be 
updated through back-propagation.  Finally the ability of the model to detect the 
new feature categories is assessed by comparing its predictions against previously-unseen labelled images.

### 3_image_segmentation.ipynb  

Allows the use of this re-trained model to categorise objects found
in previously-unseen HD photos.  It utilises the same tiling procedure as before to break down images and
send a batch of six tiles to the model, then returns an .xml file containing the categories and the absolute coordinates (with respect to the original HD photo) of any objects found.



