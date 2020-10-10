# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 09:22:17 2019

@author: kosprpv69
"""
"""Stores undetected pictures and gives the teachers the opportunity to mark unrecognised faces"""

import os
import cv2
import numpy as np
import insightface
import pickle
import imutils
import argparse
import support
import csv
import gc
from sklearn.svm import SVC
import pandas as pd


#def main(args):
def main():
      #database_dir = os.path.expanduser(args.database_dir)
      database_dir = os.path.expanduser('store/')
      if not os.path.exists(database_dir):
            os.makedirs(database_dir)
      #store_dir= args.store_dir
      store_dir= 'to_be_verified'
      if not os.path.exists(store_dir):
            os.makedirs(store_dir)
      
      ctx_id = -1 #gpu:0 cpu:-1
      face_recog = insightface.model_zoo.get_model('arcface_r100_v1')
      face_recog.prepare(ctx_id = ctx_id)
  
  
      #input_dir=os.path.expanduser(args.input_dir)
      input_dir=os.path.expanduser('unrecognised')
      for sc in [path for path in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, path))]:
            for bt in [path for path in os.listdir(os.path.join(input_dir,sc)) if os.path.isdir(os.path.join(input_dir,sc,path))]:
                  bat=os.path.join(sc,bt)
                  for se in [path for path in os.listdir(os.path.join(input_dir,bat)) if os.path.isdir(os.path.join(input_dir,bat, path))]:
                        sec=os.path.join(bat,se)
                        data= '_'.join((sc,bt,se))

                        dateid= [path for path in os.listdir(os.path.join(input_dir,sec)) if os.path.isdir(os.path.join(input_dir,
                                 sec, path))]
                        data_csv= os.path.join(database_dir,data)
                        #### pandas dataframe containing attendance ####
                        #att= pd.read_csv(data_csv+'_att.csv',dtype={'date':str}) 
                        att= pd.read_csv(data_csv+'_att.csv')
                        
                        #load means data
                        means= pd.read_csv(data_csv+'_means.csv',index_col=512,header=None)   
                        
                        #### Load classifier to extract probabilities ####
                        classifier= data_csv+'.pkl'
                        with open(classifier, 'rb') as infile:
                                    (model, class_names) = pickle.load(infile)
                                    print('Loaded classifier model from file "%s"' % classifier)
            
                        
                        with open(data_csv+'_data.csv','a') as csvfile:
                              writer=csv.writer(csvfile,delimiter=',',lineterminator='\n')
                              for dt in dateid:
                                    date= os.path.join(input_dir,sec,dt)
                                    dataset=[os.path.join(date,i) for i in os.listdir(date) 
                                                if os.path.isfile(os.path.join(date,i)) and i[-4:] in ('.png','.jpg')]
                                    
                                    #index of that date in att_csv
                                    ind=att.index[att['date']==dt+'\t'][0]
                                    
                                    #### horizontally stack all the images ####
                                    if dataset:
                                          img= cv2.imread(dataset[0])
                                          for i in dataset[1:]:
                                                s= cv2.imread(i)
                                                img=np.hstack((img,s))
                                          cv2.imshow('img',img)
                                          cv2.waitKey(0)                                    

                                    for i in dataset:
                                          label= input('Please enter the name of the student. Enter "unknown" if not known : \n')
                                          while label not in class_names and label != 'unknown':
                                                label= input('Name entered is not present in class. Please re-enter name :\n')

                                          if label != 'unknown':
                                                s=cv2.imread(i)
                                                s=cv2.cvtColor(s,cv2.COLOR_BGR2RGB)
                                                emb = face_recog.get_embedding(s)                                                                   
                                                predictions = model.predict_proba(emb)
                                                index= np.argmax(predictions)
                                                r= support.ratio(predictions,index)
                                                
                                                mindist=90
                                                for x in means.index:
                                                      dist=support.dist(emb,np.array(means.loc[x,:]))
                                                      if dist<mindist:
                                                            nm=x
                                                            mindist=dist
                                                print(nm,mindist)
                                                
                                                #date is stored in format dt+\t so that csv doesnt convert it to number
                                                print(class_names[index],r,att.loc[att['date']==dt+'\t',label])
                                                
                                                #### only added to csv directly if ratio>4, student marked absent that day and label 
                                                #### coincides with highest probability
                                                if r>1.5 and nm==label and mindist<70 and label==class_names[index] and att.loc[att['date']==dt+'\t',label].values[0]==0:
                                                      arr = support.aug(s,mode='implement')
                                                      emb = face_recog.get_embedding(arr)                                                                   
                                                      d= np.hstack((emb,np.array([label]*4).reshape(-1,1)))
                                                      writer.writerows(d)
                                                      
                                                      #modify attendance
                                                      
                                                      att.loc[ind,nm]=1
                                                      att.loc[ind,'unknown']-=1
                                                      
                                                else:
                                                      target= data+'_'+dt+'_'+label+'.jpg'
                                                      cv2.imwrite(os.path.join(store_dir,target),cv2.cvtColor(s,cv2.COLOR_RGB2BGR))
                                          else:
                                                att.loc[ind,'present']-=1
                                                att.loc[ind,'unknown']-=1
                                          #os.remove(i)
                                                
                                    else:
                                          #### Delete the folder once all the images have been checked ###
                                          #os.rmdir(date)
                                          pass
                                    
                                    cv2.destroyAllWindows()           
                                    gc.collect()
                                    
                        att.to_csv(data_csv+'_att.csv',index=False)
      
def parse_arguments(argv):
      parser = argparse.ArgumentParser()
    
      parser.add_argument('input_dir', type=str, help='Directory with unrecognised faces.')
      parser.add_argument('database_dir', type=str, 
                          help='Directory with aligned face embedding csv and classifier.pkl')
      parser.add_argument('store_dir',type=str, help='labelled unrecognised faces for officials to verify')
      return parser.parse_args(argv)
      
if __name__=='__main__':
      #main(parse_arguments(sys.argv[1:]))
      main()
      