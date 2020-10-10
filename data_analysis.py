# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:32:26 2019

@author: kosprpv69
"""

import pandas as pd
import numpy as np
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import gc

def main():
      #database_dir = os.path.expanduser(args.database_dir)
      database_dir = os.path.expanduser('store/')
      dataset=[os.path.join(database_dir,i) for i in os.listdir(database_dir) if i[-7:-4]=='att']
      
      att_dict=OrderedDict()
      recognition=OrderedDict()
      for i in dataset:
            d=pd.read_csv(i,index_col='date',parse_dates=True,usecols=['date','total','present','unknown'])
            d['att_percent']=d['present']/d['total']
            d['recognition']=(d['present']-d['unknown'])/d['present']
            att_dict[os.path.basename(i)[:-8]]=d['att_percent']
            recognition[os.path.basename(i)[:-8]]=d['recognition']
      
      gc.collect()
      att_percent= pd.concat(att_dict.values(),axis=1,sort=True,keys=att_dict.keys())
      recognition= pd.concat(recognition.values(),axis=1,sort=True,keys=recognition.keys())
      
      att_percent['mean']=att_percent.mean(axis=1,skipna=True)
      recognition['mean']=recognition.mean(axis=1,skipna=True)
      
      num_schools= att_percent.shape[1]-1
      watch_schools=[]
      for i in range(num_schools):
          name= att_percent.columns[i]
          s= att_percent[[name,'mean']].copy()
          
          #s=s.resample(rule='W').mean()
          s.dropna(inplace=True)
          s=s.ewm(span=5,adjust=False).mean()
          corr=pearsonr(s[name],s['mean'])[0]
          if corr<0.2:
              print(i,"\t",corr,"\t",len(s))
              watch_schools.append(name)
      
      
      
      print(watch_schools)
      
      fig1=att_percent.plot(figsize=(12,5),ylim=[-0.1,1.1],title='Attendance Percentage', style=['bs-','ro-','y^-']).get_figure()
      plt.ylabel('Ratio')
      fig1.savefig('att_percent.jpg')
      fig2=recognition.plot(figsize=(12,5),ylim=[-0.1,1.1],title='Recognition Effectiveness',style=['bs-','ro-','y^-']).get_figure()
      plt.ylabel('Ratio')
      fig2.savefig('rec_plot.jpg')
        
      
if __name__ == '__main__':
    #main(parse_arguments(sys.argv[1:]))
    main()