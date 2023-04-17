# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 09:51:18 2021

@author: HYDN02
"""
import warnings
import numpy as np
from astropy.modeling import models, fitting
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2
from imutils import paths
import imutils
from time import time
from glob import glob
import sys
import os
from microscopestitching import stitch
import m2stitch
from matplotlib import pyplot as plt


start = time()



def collect_images(sample_path):
    files = []
    img_array = []

    # for img in sorted(samples_path,key=os.path.getmtime):
    for img in sorted(samples_path):
        files.append(img)
        ocv_img = cv2.imread(img,1) ##OpenCV read image
        # gray = cv2.cvtColor(ocv_img, cv2.COLOR_RGB2GRAY)
        img_array.append(ocv_img)
            
    return files, img_array

def get_cols_rows_from_filename(path):
    cols = []
    rows = []
    
    for file in path:
        cols.append(int(file.split("\\")[-1][8:11]))
        rows.append(int(file.split("\\")[-1][-10:-7]))
        
    return cols,rows

## Select images to fit
def sample_total_fitting(files):
    side = len(files)**(1/2)
    if side.is_integer():
        collected_R = []
        collected_G = []
        collected_B = []
        RGB = ["R","G","B"]
        ## To fit images in diagonal, except the corners (due to background)
        diagonal = np.linspace(0,len(files)-1,int(side))
        for cf, f in enumerate(files):
            if cf in diagonal[1:-1]:
                print(cf)
                for n, rgb in enumerate(RGB):
                    fit = quantify_image(f[:,:,n])
                    if rgb == "R":
                        collected_R.append(fit)
                    elif rgb == "G":
                        collected_G.append(fit)
                    else:
                        collected_B.append(fit)
    else:
        print("not an equal x,y number of pictures")
        sys.exit()
        
    return collected_R,collected_G,collected_B
        
        
def quantify_image(file):
    ## Get image size
    xs,ys = file.shape
    
    ## This gives the position to each element in the matrix
    y, x = np.mgrid[:ys, :xs]
    z = file
    
    fit = resize_matrix(x,y,xs, ys, z)
    return fit

##Resize matrix to smaller size (to speed up fitting calculation)
def resize_matrix(x,y,xs,ys,z,dim=100):
    # dim = 100
    
    ## Make a reduced, equally-spaced sample of the fitting 
    sim_x = np.linspace(0,xs-1,dim).astype("int")
    sim_y = np.linspace(0,ys-1,dim).astype("int")
    
    mat_z = np.zeros([dim,dim]).astype("uint8")
    
    for ci,i in enumerate(sim_x):
        for cj,j in enumerate(sim_y):
            mat_z[ci,cj] = z[i,j]
    
    mat_x,mat_y =np.meshgrid(sim_x,sim_y)
    
    fit = fit_polynomial_to_matrix(mat_x, mat_y, mat_z, x, y)
    
    return fit


def fit_polynomial_to_matrix(mat_x,mat_y,mat_z,x,y):
    ##Fit reduced matrix
    deg = 6
    p_init = models.Polynomial2D(degree=deg)
    fit_p = fitting.LevMarLSQFitter()
    
    # print(f"1- {time()-start}")
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p = fit_p(p_init, mat_x, mat_y, mat_z)

    
    fit = p(x,y).astype("uint8")
    
    return fit


## Build collage of all microscopic images
def build_collage(path, images, united, bright = 160):
    
    # Open all images
    img_obj = []
    
    ##TODO bright might not be needed in the function
    # bright = np.mean(united)+1
    bright = np.max(united)+1
    
    tsize = 128,128
    for f in images:
        # grey = ImageOps.grayscale(f)
        new_img = Image.fromarray(f-united+bright)
        # new_img = new_img.resize(tsize,Image.ANTIALIAS)
        img_obj.append(new_img)

    # Coordinates of areas of interest
    xsize, ysize = img_obj[0].size
    # xsize, ysize =tsize
    length = int(len(img_obj)**(1/2))
    # i_size = xsize*1//3
    # sp = 1 #spacing between images
    # coords = (i_size, ysize//2-i_size//2, 2*i_size-sp , (ysize//2)+(i_size//2)-sp)

    # Crop all images to the coordinates chosen
    # cropp_img = []
    # for c in img_obj:
    #     cropp_img.append(c.crop(coords))

    # Calculate the number of sides of the matrix(image)
    rat = 1100/10000 ## ratio of image overlap for stitching
    pix = 100
    composed_image = Image.new('RGB', (int(xsize*length*(1-rat)), int(ysize*length*(1-rat))))   #Creates a new/blank image

    positions = []
    for m in list(range(length)):
        # ysis = int(m*ysize-ysize*rat*m)
        ysis = int(m*ysize-pix*m)
        # ysis = int(m*ysize-pix)
        
        for n in list(range(length)):
            # positions.append((int(m*ysize), int(n*xsize)))

            # xsis = int(n*xsize-xsize*rat*n)
            xsis = int(n*xsize-pix*n)
            # xsis = int(m*ysize-pix)
            positions.append((ysis,xsis))
    # print(positions)

    #Stitch cropped images together
    for c, ci in enumerate(img_obj):
        border = 10
        ci = ci.crop((border,border,xsize-border,ysize-border))
        composed_image.paste(ci, positions[c])

    # sample_name = path.rsplit('\\',2)[-2]
    # end_path = path

    ##Add text with sample name
    font = ImageFont.truetype("arial.ttf", 20)
    ImageDraw.Draw(composed_image).text((0,0),"test",(0,0,0),font=font)

    #Save image to same folder where original is located
    # print(end_path+sample_name+"_defects_coax_composed.png")
    composed_image.save(path+"..//pinhole_collage.png")
    
def detect_keypoints(images):
    ys, xs = images[0].shape
    rat = 5
    #img1 = images[0][:,int(xs*(1-1/rat)):xs-1]
    #img2 = images[0][:,0:int(xs/rat)]
    img1 = images[0][int(ys*(1-1/rat)):ys-1,:]
    img2 = images[1][0:int(ys/rat),:]
    
    print(img1.shape,img2.shape)
    
    orb = cv2.ORB_create(nfeatures=2000)

    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    # print(images[0])
    # cv2.imshow(cv2.drawKeypoints(images[0], keypoints1, None, (255, 0, 255)))
    
    # Create a BFMatcher object.
    # It will find all of the matching keypoints on two images
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)
    
    # Find matching points
    matches = bf.knnMatch(descriptors1, descriptors2,k=2)
    # print(matches)
    # print(keypoints1[0].size)
    
    # Finding the best matches
    good = []
    for m, n in matches:
        # print(m.distance, n.distance)
        if m.distance < 0.8 * n.distance:
            good.append(m)
            
    print(good)
    
    # Set minimum match condition
    MIN_MATCH_COUNT = 2
    
    if len(good) > MIN_MATCH_COUNT:
        # Convert keypoints to an argument for findHomography
        src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    
        # Establish a homography
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        
        print(M)
        result = warpImages(img2, img1, M)

    cv2.imshow("img1",img1)    
    cv2.imshow("img2",img2)    
    cv2.imshow("hola",result)

def warpImages(img1, img2, H):

    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    
    list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
    
    # When we have established a homography we need to warp perspective
    # Change field of view
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    
    list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)
    
    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    
    translation_dist = [-x_min,-y_min]
    
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    
    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1
    
    return output_img

folder_path = "C:\\Users\\HYDN02\\Seafile\\Code\\DataExamples\\stitch\\"

## List all samples (folders) inside folder_path
samples_path = glob(folder_path+"*")

# samples_path = [samples_path[0],samples_path[1],samples_path[12],samples_path[13]]

image_paths, img_array = collect_images(samples_path)

img_array = np.array(img_array)

cols, rows = get_cols_rows_from_filename(image_paths)

R,G,B = sample_total_fitting(img_array)
unitedR=np.mean(R,axis=0).astype("uint8")
unitedG=np.mean(G,axis=0).astype("uint8")
unitedB=np.mean(G,axis=0).astype("uint8")
# build_collage(folder_path, img_array, united)

# merged = stitch([(samples_path[0],0,0),(samples_path[1],1,0)])

fixed_pics = []
extra = 10
brightR = np.max(unitedR)+extra
brightG = np.max(unitedG)+extra
brightB = np.max(unitedB)+extra
# folder = "C:\\Users\\HYDN02\\Seafile\\Code\\DataExamples\\stitch_rgb\\"
folder = "D:\\PerfectStitch\\"
for n, f in enumerate(img_array):
    # print(n,f[:,:,1]-unitedR+brightR)
    # grey = ImageOps.grayscale(f)
    Rarr = f[:,:,0]-unitedR+brightR
    Garr = f[:,:,1]-unitedG+brightG
    Barr = f[:,:,2]-unitedB+brightB
    # new_img = new_img.resize(tsize,Image.ANTIALIAS)
    # fixed_pics.append(new_img)
    cv2.imwrite(folder+"X"+str(cols[n])+"_Y"+str(rows[n])+".jpg", np.stack([Rarr,Garr,Barr],axis=2))
# merged = m2stitch.stitch_images(np.array(fixed_pics),rows,cols,full_output=True)

# cv2.imshow("res",merged)

# stitched = detect_keypoints(img_array)


# plt.imshow(merged, cmap='gray')##multiplying by 1000 makes them int
# plt.show()

# cv2.imshow("Stitched", stitched)
# cv2.waitKey(0)
# cv2.destroyAllWindows()