"""Performs face alignment and stores embedding into database"""

import sys
import os
import argparse
import numpy as np
import cv2
import insightface
import support
import imutils
import csv
import pandas as pd
import pickle
from sklearn.svm import SVC
import gc

#def main(args):
def main():
  #database_dir = os.path.expanduser(args.database_dir)
  database_dir = os.path.expanduser('store/')
  if not os.path.exists(database_dir):
        os.makedirs(database_dir)
    
  ctx_id = -1 #gpu:0 cpu:-1
  face_recog = insightface.model_zoo.get_model('arcface_r100_v1')
  face_recog.prepare(ctx_id = ctx_id)
  
  
  #path_exp=os.path.expanduser(args.input_dir)
  input_dir=os.path.expanduser('registered_faces/')
  #input_dir=os.path.expanduser('registered_faces_try/')
  for sc in [path for path in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, path))]:
    for bt in [path for path in os.listdir(os.path.join(input_dir,sc)) if os.path.isdir(os.path.join(input_dir,sc,path))]:
          bat=os.path.join(sc,bt)
          for se in [path for path in os.listdir(os.path.join(input_dir,bat)) if os.path.isdir(os.path.join(input_dir,bat, path))]:
              sec=os.path.join(bat,se)
              dataset = support.get_dataset(os.path.join(input_dir,sec))
                
              nrof_successfully_aligned = 0
              
              #images=[]
              labels=[]
              emb_array=[]
              
              for cls in dataset:
                    print(cls.name)
                    for image_path in cls.image_paths:                        
                          print(image_path) 
                          scaled = cv2.imread(image_path)
                          scaled = cv2.cvtColor(scaled,cv2.COLOR_BGR2RGB)
                         
                          nrof_successfully_aligned += 4
                          arr= support.aug(scaled,mode='register')
#                          arr= (arr-127.5)/128
                          emb_array.append(face_recog.get_embedding(arr))
                          labels.extend([cls.name]*4)
                                        
              print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
              
              emb_array= np.array(emb_array)
              emb_array= emb_array.reshape((-1,512))
              #print(emb_array.shape)
              labels= np.array(labels)
              labels=labels.reshape((-1,1))
              data= np.hstack((emb_array,labels))
              
              ##### Updating the existing csv ##########
              filename= str(sc)+'_'+str(bt)+'_'+str(se)+'_data.csv'
              filename= os.path.join(database_dir,filename)
              with open(filename,'a+') as csvfile:
                    writer=csv.writer(csvfile,delimiter=',',lineterminator='\n')
                    writer.writerows(data)
                    
             ######### Reading the entire csv to build classifier #######
              dataset= pd.read_csv(filename,header=None)
              emb_array= dataset.iloc[:,0:-1].values
              names=dataset.iloc[:,-1].values
              
              print('Training classifier')
              model= SVC(kernel='sigmoid',probability=True,class_weight='balanced',decision_function_shape='ovo',
                         C= 0.1, gamma= 0.01)
              model.fit(emb_array, names)
              class_names= np.unique(names)
              
              classifier = str(sc)+'_'+str(bt)+'_'+str(se)+'.pkl'
              classifier_filename_exp= os.path.join(database_dir,classifier)
              with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                    print('Saved classifier model to file "%s"' % classifier_filename_exp)
              
              ####### STORING MEANS #########
              means=[]    
              for i in class_names:
                    means.append(np.mean(emb_array[np.where(names== i)],axis=0))
              means= np.array(means).reshape((-1,512))
              means= np.hstack((means,class_names.reshape((-1,1))))
              
              means_file= filename[:-9]+'_means.csv'
              with open(means_file,'w') as csvfile:
                    writer=csv.writer(csvfile,delimiter=',',lineterminator='\n')
                    writer.writerows(means)
                  
                  
                  
              ##### In case of new registration #####    
              if os.path.isfile(filename[:-4]+'_att.csv'):
                    att_csv= filename[:-4]+'_att.csv'
                    frame= pd.read_csv(att_csv,dtype= {'date':str})
                    classes= set(frame.columns[3:])
                    diff= sorted(set(labels).difference(classes))
                    if diff:
                          for i in diff:
                                frame[i]=np.zeros(len(frame))
                          else:
                                frame.to_csv(att_csv,header=True,index=False)
                    
              gc.collect()

def parse_arguments(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('input_dir', type=str, help='Directory with aligned images.')
  parser.add_argument('database_dir', type=str, help='Directory with aligned face embedding csv and classifier.pkl')
  return parser.parse_args(argv)

if __name__ == '__main__':
  #main(parse_arguments(sys.argv[1:]))
  main()