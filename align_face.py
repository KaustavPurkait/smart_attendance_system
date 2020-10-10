"""Performs face alignment and stores embedding into database"""

import sys
import os
import argparse
import numpy as np
import cv2
import insightface
import support
import imutils
import math
import csv
import pandas as pd
import pickle
from sklearn.svm import SVC
import gc

#def main(args):
def main():
  #database_dir = os.path.expanduser(args.database_dir)
#  database_dir = os.path.expanduser('store/')
#  if not os.path.exists(database_dir):
#        os.makedirs(database_dir)
    
  ctx_id = -1 #gpu:0 cpu:-1
  face_detect = insightface.model_zoo.get_model('retinaface_r50_v1')
  face_detect.prepare(ctx_id = ctx_id, nms=0.4)
  face_recog = insightface.model_zoo.get_model('arcface_r100_v1')
  face_recog.prepare(ctx_id = ctx_id)
  
  
  #path_exp=os.path.expanduser(args.input_dir)
  path_exp=os.path.expanduser('register2/')
  schoolid = [path for path in os.listdir(path_exp) if os.path.isdir(os.path.join(path_exp, path))]
  for sc in schoolid:
    sch=os.path.join(path_exp,sc)
    batchid= [path for path in os.listdir(sch) if os.path.isdir(os.path.join(sch, path))]
    for bt in batchid:
          bat=os.path.join(sch,bt)
          sectionid= [path for path in os.listdir(bat) if os.path.isdir(os.path.join(bat, path))]
          for se in sectionid:
              sec=os.path.join(bat,se)
              dataset = support.get_dataset(sec)
                
              nrof_images_total = 0
              nrof_successfully_aligned = 0
              
              #images=[]
              labels=[]
              emb_array=[]
              
              for cls in dataset:
                    #store_dir= args.store_dir
                    store_dir= 'registered_faces'
                    store_dir= os.path.join(store_dir,sc,bt,se,str(cls.name))
                    if not os.path.exists(store_dir):
                          os.makedirs(store_dir)
                    
                    a=0
                    for image_path in cls.image_paths:
                          nrof_images_total += 1
                          
                          print(image_path) 
                          imgorig = cv2.imread(image_path)
                          imgorig = imgorig[:,:,0:3]
                          imgorig = cv2.cvtColor(imgorig,cv2.COLOR_BGR2RGB)
                          
                          h,w,c= imgorig.shape
                          if c<3:
                                continue
                          
                          if h>600:
                                img=imutils.resize(imgorig,height=600)
                                r= h/600
                          else:
                                img=imgorig.copy()
                                r=1
                          
                          h,w,_= img.shape
                          bbox, landmark = face_detect.detect(img, threshold=0.5, scale=1.0)

                          if len(bbox)>0:
                                bb= np.zeros(4,dtype=np.int32)
                                for i,j in zip(bbox,landmark):
                                      width=i[2]-i[0]
                                      height=i[3]-i[1]
                                      bb[0]=int(np.maximum(0,i[0]-width/4)*r)
                                      bb[1]=int(np.maximum(0,i[1]-height/4)*r)
                                      bb[2]=int(np.minimum(i[2]+width/4,w)*r)
                                      bb[3]=int(np.minimum(i[3]+height/4,h)*r)
                                      face= imgorig[bb[1]:bb[3],bb[0]:bb[2],:]

                                      angle= (j[1][1]-j[0][1])/(j[1][0]-j[0][0])
                                      angle= math.degrees(math.atan(angle))
                                      s= imutils.rotate_bound(face,-angle)
                                      h,w,_= s.shape
                                      if h>250:
                                            s= imutils.resize(s,height=250)
                                            
#                                      cv2.imshow('rot',cv2.cvtColor(s,cv2.COLOR_RGB2BGR))
#                                      cv2.waitKey(0)
                                      
                                      bbox_new, _ = face_detect.detect(s, threshold=0.5, scale=1.0)
            
                                      if len(bbox_new)>0:
                                            h,w,_= s.shape
                                            x= support.rerec(bbox_new[0][:4])
                                            x[0]=max(0,int(x[0]))
                                            x[1]=max(0,int(x[1]))
                                            x[2]=min(w,int(x[2]))
                                            x[3]=min(h,int(x[3]))  
                                            
                                            rot_face= s[int(x[1]):int(x[3]),int(x[0]):int(x[2]),:]
#                                            print(rot_face.shape)
                                          
                                            scaled = cv2.resize(rot_face,(112,112), interpolation=cv2.INTER_AREA)
                                            cv2.imwrite(os.path.join(store_dir, str(a)+'.jpg'),
                                                                     cv2.cvtColor(scaled,cv2.COLOR_RGB2BGR))
                                            a+=1
                                            nrof_successfully_aligned += 1
                                            
#                                            arr= support.aug(scaled,mode='register')                                                                                                               
#                                            emb_array.append(face_recog.get_embedding(arr))
#                                            labels.extend([cls.name]*4)
                                            break
                    if a<10:
                          print('Fewer than 10 images received. Please enter %d more images'%(10-a))
    
                                        
              print('Total number of images: %d' % nrof_images_total)
              print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
              
#              emb_array= np.array(emb_array)
#              emb_array= emb_array.reshape((-1,512))
#              #print(emb_array.shape)
#              labels= np.array(labels)
#              labels=labels.reshape((-1,1))
#              data= np.hstack((emb_array,labels))
#              
#              
#              filename= str(sc)+'_'+str(bt)+'_'+str(se)+'.csv'
#              filename= os.path.join(database_dir,filename)
#              with open(filename,'a+') as csvfile:
#                    writer=csv.writer(csvfile,delimiter=',',lineterminator='\n')
#                    writer.writerows(data)
#            
#              dataset= pd.read_csv(filename,header=None)
#              emb_array= dataset.iloc[:,0:-1].values
#              labels=dataset.iloc[:,-1].values
#              
#              print('Training classifier')
#              model= SVC(kernel='sigmoid',probability=True,class_weight='balanced',decision_function_shape='ovo',
#                         C= 0.1, gamma= 0.01)
#              model.fit(emb_array, labels)
#              class_names= np.unique(labels)
#              
#              classifier = str(sc)+'_'+str(bt)+'_'+str(se)+'.pkl'
#              classifier_filename_exp= os.path.join(database_dir,classifier)
#              with open(classifier_filename_exp, 'wb') as outfile:
#                    pickle.dump((model, class_names), outfile)
#                    print('Saved classifier model to file "%s"' % classifier_filename_exp)
#              
#              gc.collect()

def parse_arguments(argv):
  parser = argparse.ArgumentParser()
    
  parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
#  parser.add_argument('database_dir', type=str, help='Directory with aligned face embedding csv and classifier.pkl')
  parser.add_argument('aligned_dir', type=str, help='Directory with aligned faces')
  return parser.parse_args(argv)

if __name__ == '__main__':
  #main(parse_arguments(sys.argv[1:]))
  main()