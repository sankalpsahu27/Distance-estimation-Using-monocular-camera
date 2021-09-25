'''
Purpose: Generates csv file of annotations from .txts
'''
import pandas as pd
import os
from tqdm import tqdm
import argparse
import numpy as np 
import json

argparser = argparse.ArgumentParser(description='Generate annotations csv file from .txts')
argparser.add_argument('-i', '--input',
                       help='input dir name',default = "F:/KITTI-distance-estimation/original_data/train_annots/")
argparser.add_argument('-o', '--output',
                       help='output file name', default = "annotations1.csv")

args = argparser.parse_args()

# parse arguments
INPUTDIR = args.input
FILENAME = args.output

'''
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
'''

classes = ['Pedestrian', 'Van', 'Truck', 'Car', 'Cyclist', 'Tram', 'Misc', 'Person_sitting']
class_shape = [[175, 55, 30], [170, 170, 340], [400, 350, 1500], [160, 180, 400],[110, 50, 180], [346, 257, 1466],
                       [161, 121, 207], [118, 56, 85]]

df = pd.DataFrame(columns=['name', 'class', 'truncated', 'occluded', 'observation angle', \
                           'xmin', 'ymin', 'xmax', 'ymax', 'heightm', 'widthm', 'lengthm', \
                           'xloc', 'yloc', 'zloc', 'rot_y', 'distance','width','height','size_h','size_w','size_d','diagonal'])

def assign_values(filename, idx, list_to_assign):
    df.at[idx, 'name'] = filename

    df.at[idx, 'class'] = list_to_assign[0]
    df.at[idx, 'truncated'] = list_to_assign[1]
    df.at[idx, 'occluded'] = list_to_assign[2]
    df.at[idx, 'observation angle'] = list_to_assign[3]

    # bbox coordinates
    df.at[idx, 'xmin'] = list_to_assign[4]
    df.at[idx, 'ymin'] = list_to_assign[5]
    df.at[idx, 'xmax'] = list_to_assign[6]
    df.at[idx, 'ymax'] = list_to_assign[7]

    # 3D object dimensions
    df.at[idx, 'heightm'] = list_to_assign[8]
    df.at[idx, 'widthm'] = list_to_assign[9]
    df.at[idx, 'lengthm'] = list_to_assign[10]

    zlocm = int(np.multiply(float(list_to_assign[13]),100))
    # 3D object location 
    df.at[idx, 'xloc'] = list_to_assign[11]
    df.at[idx, 'yloc'] = list_to_assign[12]
    df.at[idx, 'zloc'] = list_to_assign[13]

    # rotation around y-axis in camera coordinates
    df.at[idx, 'rot_y'] = list_to_assign[14]
    Distance = np.sqrt(np.square(float(list_to_assign[11]))+np.square(float(list_to_assign[12]))+np.square(float(list_to_assign[13])))
    df.at[idx, 'distance'] = Distance
    
    width = float(list_to_assign[6]) - float(list_to_assign[4])
    height = float(list_to_assign[7]) - float(list_to_assign[5])
    diagonal = np.sqrt(np.square(width) + np.square(height))
    
    imgdiag = round(np.sqrt(np.square(1224) + np.square(370)),2)
    width = round(1/float(width/1224),2)
    df.at[idx, 'width'] = width
    
    height = round(1/float(height/370),2) 
    df.at[idx, 'height'] = height  
    
    diagonal = round(1/round(float(diagonal/imgdiag),2),2)
    df.at[idx, 'diagonal'] = (diagonal)
    
    if list_to_assign[0] == 'Pedestrian':   
        df.at[idx, 'size_h'] = 175
        df.at[idx, 'size_w'] = 55
        df.at[idx, 'size_d'] = 30
    elif list_to_assign[0] == 'Van':
        df.at[idx, 'size_h'] = 170
        df.at[idx, 'size_w'] = 170
        df.at[idx, 'size_d'] = 340
    elif list_to_assign[0] =='Truck':
        df.at[idx, 'size_h'] = 400
        df.at[idx, 'size_w'] = 350
        df.at[idx, 'size_d'] = 1400
    elif list_to_assign[0] =='Car':
        df.at[idx, 'size_h'] = 160
        df.at[idx, 'size_w'] = 180
        df.at[idx, 'size_d'] = 400
    elif list_to_assign[0] =='Cyclist':
        df.at[idx, 'size_h'] = 110
        df.at[idx, 'size_w'] = 50
        df.at[idx, 'size_d'] = 80
    elif list_to_assign[0] =='Tram':
        df.at[idx, 'size_h'] = 346
        df.at[idx, 'size_w'] = 257
        df.at[idx, 'size_d'] = 1466
    elif list_to_assign[0] =='Misc':
        df.at[idx, 'size_h'] = 161
        df.at[idx, 'size_w'] = 121
        df.at[idx, 'size_d'] = 207
    elif list_to_assign[0] =='Person_sitting':
        df.at[idx, 'size_h'] = 118
        df.at[idx, 'size_w'] = 56
        df.at[idx, 'size_d'] = 85
    else :
        print('nothing such like it')
    
     

all_files = sorted(os.listdir(INPUTDIR))
pbar = tqdm(total=len(all_files), position=1)

count = 0
for idx, f in enumerate(all_files):
    pbar.update(1)
    file_object = open(INPUTDIR + f, 'r')
    file_content = [x.strip() for x in file_object.readlines()]

    for line in file_content:
        elements = line.split()
        if elements[0] == 'DontCare': 
            continue
        assign_values(f, count, elements)
        count += 1
 
df.to_csv(FILENAME, index=False)

mask = np.random.rand(len(df)) < 0.8
train = df[mask]
test = df[~mask]

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
