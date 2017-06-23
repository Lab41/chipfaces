import numpy as np
import cv2
import os
import sys
import time
from math import sin, cos, radians

#stolen from https://stackoverflow.com/questions/5015124/rotated-face-detection
#download model from:
#curl https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml > haarcascade_frontalface_alt.xml 
#curl https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml > haarcascade_frontalface_alt2.xml 

settings = {
    'scaleFactor': 1.05, 
    'minNeighbors': 3, 
    'minSize': (50, 50), 
    'flags': cv2.CASCADE_SCALE_IMAGE
}

def rotate_image(image, angle):
    if angle == 0: return image
    height, width = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 0.9)
    result = cv2.warpAffine(image, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
    return result

def rotate_point(pos, img, angle):
    if angle == 0: return pos
    x = pos[0] - img.shape[1]*0.4
    y = pos[1] - img.shape[0]*0.4
    newx = x*cos(radians(angle)) + y*sin(radians(angle)) + img.shape[1]*0.4
    newy = -x*sin(radians(angle)) + y*cos(radians(angle)) + img.shape[0]*0.4
    return int(newx), int(newy), pos[2], pos[3]


def find_work(topdir,filetypes):
    retval = []
    for dirpath, dirnames, files in os.walk(topdir):
        for name in files:
            if name.lower().split('.')[-1] in filetypes:
                retval.append(name)
    return retval

def process(in_dir,out_dir):
    #cv2.startWindowThread()
    face = cv2.CascadeClassifier('/prog/haarcascade_frontalface_alt2.xml')
    work_items = find_work(in_dir,['jpg','png','bmp'])
    
    for work_item in work_items:
        work_file = os.path.join(in_dir,work_item)
        print('processing: {0}'.format(work_file))
        img = cv2.imread(work_file)
        for angle in [0, -25, 25, -45, 45, -90, 90]:
            rimg = rotate_image(img, angle)
            detected = face.detectMultiScale(rimg, **settings)
            if len(detected):
                detected = [rotate_point(detected[-1], img, -angle)]
                break

        for x, y, w, h in detected[-1:]:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
            roi = img[y:y+h,x:x+w]
            out_filename = '{0}-{1}-{2}-{3}-{4}.png'.format(work_item,x,y,x+w,y+h) 
            cv2.imwrite(os.path.join(out_dir,out_filename),roi)

        #cv2.imshow('facedetect',img)

        #if cv2.waitKey(5) != -1:
        #    break

    #cv2.destroyWindow('facedetect')

if __name__ == '__main__':
    process(sys.argv[1],sys.argv[2])
