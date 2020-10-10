# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 10:33:01 2019

@author: kosprpv69
"""

import cv2
import sys
import os
import argparse
import insightface
import numpy as np
import pandas as pd
from scipy import stats
import support
import pickle
import imutils
import math
import csv
import gc

#def main(args):
def main():
      #input_dir = os.path.join(os.path.dirname(__file__), args.input_dir)
      input_dir='to_be_verified'
      #aligned_dir= args.aligned_dir
      aligned_dir= 'registered_faces'
      #database_dir= args.database_dir
      database_dir= 'store'
      
      ctx_id = -1 #gpu:0 cpu:-1
      face_recog = insightface.model_zoo.get_model('arcface_r100_v1')
      face_recog.prepare(ctx_id = ctx_id)
      
      dataset= [i for i in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir,i))]
      for i in dataset:
            img=cv2.imread(os.path.join(input_dir,i))
            sp= i.split('_')
            label_dir= os.path.join(aligned_dir,sp[0],sp[1],sp[2],sp[4][:-4])
            
            #create img collage of the teacher label
            s= [os.path.join(label_dir,i) for i in os.listdir(label_dir)[:10]]
            col1= np.hstack((pic(s[0]),pic(s[1]),pic(s[2]),pic(s[3]),pic(s[4])))
            col2= np.hstack((pic(s[5]),pic(s[6]),pic(s[7]),pic(s[8]),pic(s[9])))
            col=np.vstack((col1,col2))
            
            #create collage of predicted label
            data=pd.read_csv(os.path.join(database_dir,"_".join([sp[0],sp[1],sp[2]])+'_data.csv'),header=None)
            labels= data.iloc[:,[-1]]
            data= data.iloc[:,:-1].values
            emb = face_recog.get_embedding(img)  
            a= [support.dist(j,emb) for j in data]
            a= sorted(enumerate(a),key=lambda x: x[1])[:10]
            ind,dis= zip(*a)
            labels=labels.loc[list(ind)]
            pred_label= str(np.squeeze(stats.mode(labels)[0]))
            pred_label_dir= os.path.join(aligned_dir,sp[0],sp[1],sp[2],pred_label)
            
            s= [os.path.join(pred_label_dir,i) for i in os.listdir(pred_label_dir)[:10]]
            col1= np.hstack((pic(s[0]),pic(s[1]),pic(s[2]),pic(s[3]),pic(s[4])))
            col2= np.hstack((pic(s[5]),pic(s[6]),pic(s[7]),pic(s[8]),pic(s[9])))
            pred_col=np.vstack((col1,col2))
            
            cv2.imshow('img',img)
            cv2.imshow('col',col)
            cv2.imshow('pred_col',pred_col)
            
            cv2.waitKey(0)
            
            while True:
                  inp= input('Enter YES if its the same person else enter NO:\n').lower()
                  if inp in ('yes','no'):
                        break
            if inp== 'yes':
                  with open(os.path.join(database_dir,'_'.join([sp[0],sp[1],sp[2]]))+'_data.csv','a') as csvfile:
                        writer=csv.writer(csvfile,delimiter=',',lineterminator='\n')
                        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                        arr = support.aug(img,mode='register')
                        emb = face_recog.get_embedding(arr)                                                                   
                        d= np.hstack((emb,np.array([sp[4][:-4]]*4).reshape(-1,1)))
                        writer.writerows(d)
            else:
                  pass
            os.remove(os.path.join(input_dir,i))
            cv2.destroyAllWindows() 
            gc.collect()           
             
            
def pic(path):
      return cv2.imread(path)

def parse_arguments(argv):
      parser = argparse.ArgumentParser()
    
      parser.add_argument('input_dir', type=str, help='Directory with unrecognised faces.')
      parser.add_argument('aligned_dir', type=str, 
                          help='Directory with unrecognised faces')
      parser.add_argument('database_dir',type=str, help='directory of database csv')
      return parser.parse_args(argv)
      
if __name__=='__main__':
      #main(parse_arguments(sys.argv[1:]))
      main()