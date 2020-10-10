"""Detects faces from images and recognises them and stores them into the aligned faces directory"""


import cv2
import sys
import os
import argparse
import insightface
import numpy as np
import support
import pickle
import imutils
import math
import csv
import gc
import pandas as pd


#def main(args):
def main():
    #path_exp = os.path.join(os.path.dirname(__file__), args.input_dir)
    #path_exp = os.path.join(os.getcwd(),'trialpic')
    path_exp= 'Att2'

    dataset=[os.path.join(path_exp,i) for i in os.listdir(path_exp) if os.path.isfile(os.path.join(path_exp,i)) and i[-4:] in ('.png','.jpg')]
    #database_dir= args.database_dir
    database_dir='store'

    ctx_id = -1 #gpu:0 cpu:-1
    face_detect = insightface.model_zoo.get_model('retinaface_r50_v1')
    face_detect.prepare(ctx_id = ctx_id, nms=0.4)
    face_recog = insightface.model_zoo.get_model('arcface_r100_v1')
    face_recog.prepare(ctx_id = ctx_id)
    
    
    for image_path in dataset:  
        emb_array=[]
        images=[]
        print('image_path:',image_path)
        imgorig= cv2.imread(image_path)
        imgorig= cv2.cvtColor(imgorig,cv2.COLOR_BGR2RGB)
        h,w,_=imgorig.shape
        #resize image for detection for faster detection. Store resize ratio in r. If image is small, not resized
        if h>1200:
              r= h/1200
              img= imutils.resize(imgorig,height=1200)
        else:
              r=1
              img=imgorig

        img = img[:,:,0:3]
                
        bbox, landmark = face_detect.detect(img, threshold=0.5, scale=1.0)
        
        #Reject bounding boxes having less than 0.5 of the side length of average side length
        area=[]
        for i in bbox:
              area.append(((i[2]-i[0])*(i[3]-i[1]))**0.5)
        
        avg_area= sum(area)/len(area)
        area_ind= list(filter(lambda x: area[x]>0.5*avg_area,range(len(area))))
        bbox=bbox[area_ind,:]
        landmark= landmark[area_ind,:,:]
        
        #draw_image_with_boxes(bbox,landmark,img.copy())
        
        ####### GET EXTENDED BOUNDARY BOX AND ROTATE IMAGE #########
        if len(bbox)>0:
            bb= np.zeros(4,dtype=np.int32)
            
            for i,j in zip(bbox,landmark):
                    h,w,_= img.shape
                    width=i[2]-i[0]
                    height=i[3]-i[1]
                    
                    #find bounding box coordinates on the original picture. 
                    #Width/height buffer used so that image is not cut off after rotation.
                    bb[0]=int(np.maximum(0,i[0]-width/4)*r)
                    bb[1]=int(np.maximum(0,i[1]-height/4)*r)
                    bb[2]=int(np.minimum(i[2]+width/4,w)*r)
                    bb[3]=int(np.minimum(i[3]+height/4,h)*r)
                    face= imgorig[bb[1]:bb[3],bb[0]:bb[2],:]
                    
                    #find angle between the 2 landmarks representing eyes and the x axis and rotate the image to align it
                    angle= (j[1][1]-j[0][1])/(j[1][0]-j[0][0])
                    angle= math.degrees(math.atan(angle))
                    s= imutils.rotate_bound(face,-angle)
                    h,w,_= s.shape
                    if h>250:
                          s= imutils.resize(s,height=250)
                    
                   #find new bounding box in the rotated and resized face image
                    bbox_new, _ = face_detect.detect(s, threshold=0.5, scale=1.0)

                    if len(bbox_new)>0:
                          h,w,_= s.shape
                          #make the face image square so that it doesnt get distored when resized to (112,112)
                          x= support.rerec(bbox_new[0][:4])
                          x[0]=max(0,int(x[0]))
                          x[1]=max(0,int(x[1]))
                          x[2]=min(w,int(x[2]))
                          x[3]=min(h,int(x[3]))
                          rot_face= s[int(x[1]):int(x[3]),int(x[0]):int(x[2]),:]
                          
                          #store the image in an array to run recognition                      
                          scaled = cv2.resize(rot_face,(112,112),interpolation=cv2.INTER_AREA)
                          images.append(scaled)
            
            #convert images from list to array and run recognition
            images= np.array(images)
            emb_array= face_recog.get_embedding(images)              
            emb_array= emb_array.reshape((-1,512))
            
            ###### LOAD CLASSIFIER AND PREDICT FROM EMBEDDING ARRAY ######
            classifier = os.path.basename(image_path)[:-15]+'.pkl'
            classifier = os.path.join(database_dir,classifier)
            try:
                  with open(classifier, 'rb') as infile:
                        (model, class_names) = pickle.load(infile)
                        print('Loaded classifier model from file "%s"' % classifier)
            except FileNotFoundError:
                  print('Please register the school first')
                  continue
            
            #load the means data to be compared
            mean_csv = os.path.basename(image_path)[:-15]+'_means.csv'
            mean_csv = os.path.join(database_dir,mean_csv)
            means= pd.read_csv(mean_csv,index_col=512,header=None)        
            
            
            predictions = model.predict_proba(emb_array)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            dist= list(map(lambda x:support.dist(np.array(means.loc[class_names[best_class_indices[x]],:]),emb_array[x]),
                               range(len(best_class_indices))))
                        
            for i in range(len(best_class_indices)):
                  print('%s  %s %s: %.3f' % (os.path.basename(image_path)[-14:-4], str(i), 
                                          class_names[best_class_indices[i]],best_class_probabilities[i]))
            
            #print(predictions,best_class_indices,best_class_probabilities,sep='\n')
            
            

            ###### EXTRACT ATTENDANCE AND UNRECOGNISED FACES #########
            # ensure no repitition of faces occurs
            storage=dict({})
            
            for i,j in enumerate(best_class_indices):
                  if dist[i]<storage.get(j,[-1,90])[1]:
                        storage[j]=[i,dist[i]]
            
            present=[] #students present beyond doubt : ratio > 10
            add=[] #images to be added to csv database to increase accuracy. ratio: 10-17
            ims=[] #index of images marked present. Complement of this is the images unrecognised

            for i,j in storage.items():
                  ratio= support.ratio(predictions[j[0]],best_class_indices[j[0]])
                  print(j[0],'ratio:', round(ratio,3),'dist:',j[1])
                  if (ratio>=10 and dist[j[0]]<60) or (2.5<=ratio<10 and dist[j[0]]<50):
                        present.append(i)       #index of students present
                        ims.append(j[0])        #index of pic in images which dont need to be confirmed
                        if (10<=ratio<=17 and dist[j[0]]>40) or (2.5<=ratio<10 and dist[j[0]]<50):
                              add.append(j[0])
            
            unrecognised= list(filter(lambda x: x not in ims, range(len(best_class_indices)))) 

            present_names= list(map(lambda x:class_names[x],present))
            #print('present index',present)
            print('present',present_names)
            print('data insertion to database',add)
            print('index which are present',ims)
            print('unrecognised index', unrecognised)
            
            for i in images:
                  cv2.imshow('img',cv2.cvtColor(i,cv2.COLOR_RGB2BGR))
                  cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            
            ######## APPEND THE NEW VALUES TO DATABASE CSV #########
            data_csv = os.path.basename(image_path)[:-15]+'_data.csv'
            data_csv = os.path.join(database_dir,data_csv)
            with open(data_csv,'a') as csvfile:
                  writer= csv.writer(csvfile,delimiter=',',lineterminator='\n')
                  for i in add:
                        arr = support.aug(images[i],mode='implement')
                        emb = face_recog.get_embedding(arr)                                                                   
                        data= np.hstack((emb,np.array([class_names[best_class_indices[i]]]*2).reshape(-1,1)))
                        writer.writerows(data)
            
            
            ###### UPDATE ATTENDANCE #########
            att_csv= classifier = os.path.basename(image_path)[:-15]+'_att.csv' #image_path in format sch_cl_bat_yyyy-mm-dd.jpg
            att_csv = os.path.join(database_dir,att_csv)
            
            #if attendance file doesnt exist previously, create one
            if not(os.path.exists(att_csv) and os.path.isfile(att_csv)):
                  with open(att_csv,'w') as write:
                        writer= csv.writer(write,delimiter=',',lineterminator='\n')
                        string=['date','total','present','unknown']+list(class_names)
                        writer.writerow(string)
            else:
                  with open(att_csv,'r') as csvfile:
                        reader = csv.reader(csvfile)
                        class_names=next(reader)[4:] #first 4 fields are date, total, present, unknown
            
            # update attendance file
            with open(att_csv,'a') as write:
                  writer= csv.writer(write,delimiter=',',lineterminator='\n')
                  date= os.path.basename(image_path)[-14:-4]
                  total= len(class_names)
                  pres= len(best_class_indices)
                  string=[]
                  for i in class_names:
                        if i in present_names:
                              string.append(1)
                        else:
                              string.append(0)
                              
                  unknown= pres-sum(string)
                  string=[str(date)+'\t',total,pres,unknown]+string
                  writer.writerow(string)
            
            
            ##### STORE THE UNRECOGNISED FACES ######
            if len(unrecognised)>0:
                  s= os.path.basename(image_path).split('_')
                  #store_dir= os.path.join(args.store_dir,s[0],s[1],s[2],s[3])[:-4]
                  store_dir= os.path.join('unrecognised',s[0],s[1],s[2],s[3])[:-4]
                  if not os.path.exists(store_dir):
                        os.makedirs(store_dir)
                  try:
                        num= int(os.listdir(store_dir)[-1][:-4])+1
                  except:
                        num=0
                        
                  for i,j in enumerate(unrecognised,num):
                        im= cv2.cvtColor(images[j],cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(store_dir,str(j)+'.jpg'),im)
                        
      
#        img= cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
#        cv2.imshow('img',img)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
                  
        del images
        del emb_array
        gc.collect()    #garbage collection
        #break            


                  
def draw_image_with_boxes(bbox,landmark, data):
    for i in bbox:
        x1, y1, x2, y2 = i[:4]
        cv2.rectangle(data,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),1)
    for i in landmark:
        for j in range(len(landmark[0])):
              cv2.circle(data,(int(i[j][0]),int(i[j][1])), 1, (255,0,0), thickness=-1)
    data=cv2.cvtColor(data,cv2.COLOR_RGB2BGR)
    cv2.imshow('Image',data)
    cv2.waitKey(0)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('database_dir', type=str, help='Directory with output csv and classifier')
    parser.add_argument('store_dir',type=str, help='temporary storage for unrecognised pictures')
    return parser.parse_args(argv)

if __name__ == '__main__':
    #main(parse_arguments(sys.argv[1:]))
    main()
