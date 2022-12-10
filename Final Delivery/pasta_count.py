import cv2 #to do image manipulation
import numpy as np #to do some calculations
import math #to do some calculations
import sys #to read arguments passed to the script

#-----------------------------------------------------------
#-----------------------------------------------------------
#FUNCTIONS
#-----------------------------------------------------------
#-----------------------------------------------------------


def blob_detection(img,se): # our connected component analys algorithm implementation
   cnt_label = 1 #label of blob detected
   IDS = np.zeros(pasta.shape) #our matrix to keep track of assignments
   for j,row in enumerate(IDS): 
      for i,IDS_pix in enumerate(row):
         if(img[j,i]!=0): #if pixel is not 0
            widw_half_len_x = math.floor(se.shape[0]/2) #get structuring element half widht
            widw_half_len_y = math.floor(se.shape[1]/2) #get structuring element half height
            ids_window = IDS[j-widw_half_len_x:j+widw_half_len_x+1 , i-widw_half_len_y:i+widw_half_len_y+1] #window of IDS same size of structuring element
            neigh = np.multiply(ids_window ,se) #element-wise multiplication
            if(np.sum(neigh)==0): #if all neighbours are not assigned(0)
               IDS[j,i] = cnt_label #assign label to that pixel
               cnt_label += 1 #next label
            else:
                #to assign the label by analyzing neighbour pixels we take the mean of them.
                IDS[j,i] = int(np.sum(neigh)/(9-len(np.where(neigh==0.)[0]))) #mean value of neigh different than 0
   
   return IDS, cnt_label #return image but labeled and the number of labels
   #return IDS, cnt_label-1 #return image but labeled and the number of labels

def area_threshold(IDS,cnt_label,pix_thr): #threshold labeled blobs by pixel area, otherwise we have a lot of noise
    IDS_new = np.zeros(IDS.shape) #start a new matrix with all zeors
    cnt_newidx = 1
    for i in range(1,cnt_label+1):
        num_pix = len(np.where(IDS == i)[0]) #count number of pixels in that blob
        if num_pix >= pix_thr: #passed threshold
            IDS_new[np.where(IDS==i)] = cnt_newidx #save to the new matrix interesting blob with new id
            cnt_newidx += 1 #add1 to counter   
    
    return IDS_new ,cnt_newidx-1 #-1 as the last new id generated when +=1 doens't exist


#-----------------------------------------------------------
#-----------------------------------------------------------

print("WELCOME TO PASTA IMAGE ANALYZER!")

#assert check if a file was passed
assert len(sys.argv) > 1 ,"Please append an image path as an argument to this script"

path = sys.argv[1:][0] #read path passed as argument
image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY) #read image and convert to greyscale
#print("Displaying provided image...")
#print_image(image)


#firts step, apply threshold to convert image to binary
(thresh, image_bw) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

##NOW APPLY A SERIES OF EROSIONS AND DILATIONS TOGETHER WITH IMAGE SUBSTRACTION TO SEPARATE PASTA TYPES
print("")
print("Applying morphological operators...")

#erode image to isolate bigger pasta blobs
kernel = np.ones((18, 18), np.uint8) #kernel in use
img_erosion = cv2.erode(image_bw, kernel, iterations=1)
pasta = cv2.dilate(img_erosion, kernel)

#dilate pasta to substract more cleanly
kernel = np.ones((30, 30), np.uint8) #kernel in use
img_again = cv2.dilate(pasta, kernel)

#substract pasta from the complete image
sub = cv2.subtract(image_bw,img_again)

#erode substracted image to get rid of spaguetti, isolate grains
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)) #we are using a kernel 15x15 with a circular shape of se
grains = cv2.erode(sub, kernel)

#dilate grains to substract more cleanly
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))#kernel in use
grains = cv2.dilate(grains, kernel,iterations=2) #dilate twice

#isolat spaguetti by substracting grains from previous substraction
spaguetti = cv2.subtract(sub, grains)

#BLOB detection, now we count the number of blobs in each separated image

#We'll use a 8 connectivity structuring element for our connected component analysis
se = np.array(
      [[1,1,1],
      [1,0,1],
      [1,1,1]]
)

#Now go over all images:
images = {"pasta":pasta,"grains":grains,"spaguetti":spaguetti}
THR_pix = 140 #this is a general threshold for all images,
#could achive better results with a more general thr 
# if it was more or less adjusted to each image

print("blob detection...")
print("")
for key in images:
    IDS, cnt_label = blob_detection(images[key],se)
    IDS, cnt_label = area_threshold(IDS,cnt_label,THR_pix)
    print(f"detected {cnt_label} {key} objects in image ")

print("")
print("finished.")
print("Bye! :)")

#A way we could have made this program faster is by downsampling images.