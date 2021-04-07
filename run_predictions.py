import os
import numpy as np
import json
from PIL import Image
np.seterr(divide='ignore', invalid='ignore')
def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    
    '''
    BEGIN YOUR CODE
    '''
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''
    # precision
    N=20
    # load kernel
    K = Image.open("kernel.jpg")
    K=K.resize((int(640/N),int(480/N)))
    K = np.asarray(K)
    K=K[:,:,:3]
    (K_n_rows,K_n_cols,K_n_channels) = np.shape(K)
    for i in range(19):
        for j in range(19):
            if np.shape(I)==(480,640,3):
                subimage = I[int(480/N)*i:int(480/N)*i+K_n_rows,int(640/N)*j:int(640/N)*j+K_n_cols]
                v_1 = np.matrix(subimage.ravel())
                v_2 = np.matrix(K.ravel())
                normalizedv_1 = v_1/np.linalg.norm(v_1)
                normalizedv_2 = v_2/np.linalg.norm(v_2)
                inner = np.inner(normalizedv_1, normalizedv_2)
                if inner> 0.75:
                     bounding_boxes.append([int(480/N),int(640/N),int(480/N)*i+K_n_rows,int(640/N)*j+K_n_cols])
              
  
    '''
    END YOUR CODE
    '''
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes

# set the path to the downloaded data: 
data_path = '/Users/Ismail/Documents/Github/ComputerVision/RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = '/Users/Ismail/Documents/Github/ComputerVision/HW1_preds' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

preds = {}
for i in range(len(file_names)):
    
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(I)
    preds[file_names[i]] = detect_red_light(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
