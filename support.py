"""Functions for building the face recognition network."""

import os
import numpy as np
import cv2
import math
import random

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)
  
      
def get_dataset(path, has_class_directories=True):
    """ Get the training dataset stored in the form of dir/classes/images """  
      
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))
    return dataset


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths


def rerec(bboxA):
    """Convert bboxA to square."""
    
    h = bboxA[3]-bboxA[1]
    w = bboxA[2]-bboxA[0]
    l = np.maximum(w, h)
    bboxA[0] = bboxA[0]+w*0.5-l*0.5
    bboxA[1] = bboxA[1]+h*0.5-l*0.5
    bboxA[2:4] = bboxA[0:2] + l
    return bboxA


def add_noise(image):
      """ Adds gaussian noise to the image"""
      image=image/255.0
      
      row,col,ch= image.shape
      mean = 0
      var = random.uniform(0.007,0.012)
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss

      noisy= np.clip(noisy*255,0,255)
      noisy= noisy.astype('uint8')
      return noisy

def aug(img, mode='implement'):
      """Augment the given image by flipping and changing illuminations. Takes input in RGB format"""
      
      assert mode in ('register','implement')
      flipped=np.fliplr(img)
      if mode=='implement':
            return np.array([img,flipped])
      
      
      dark= cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
      a= dark[:,:,2]
      a = cv2.pow(a/255.0,1.4)
      a *=255
      a= a.astype(np.uint8)
      dark[:,:,2]=a
      dark= cv2.cvtColor(dark,cv2.COLOR_HSV2RGB)
      dark= add_noise(dark)
        
      light= cv2.cvtColor(flipped,cv2.COLOR_RGB2HSV)
      a= light[:,:,2]
      a = cv2.pow(a/255.0,0.7)
      a *=255
      a= a.astype(np.uint8)
      light[:,:,2]=a
      light= cv2.cvtColor(light,cv2.COLOR_HSV2RGB)
      light=add_noise(light)
      #noise is only added during registration

      return np.array([img,flipped,dark,light])
    
      
def ratio(predictions,index):
      """Calculate the ratio between the top most 2 probable values"""
      
      predictions= np.reshape(predictions,-1)
      l,r=(0,0)
      try:
            l= max(predictions[:index])
      except:
            pass
      try:
            r= max(predictions[index+1:])
      except:
            pass
      return predictions[index]/max(l,r)


def dist(emb1,emb2):
      """Returns the cosine distance between 2 vectors(arrays)"""
      
      def mod(img):
            return np.linalg.norm(img)
      
      dist= np.sum(emb2*emb1)/(mod(emb1)*mod(emb2))
      dist= min(dist,1)
      dist= max(dist,-1)
      deg= math.degrees(math.acos(dist))
      return deg