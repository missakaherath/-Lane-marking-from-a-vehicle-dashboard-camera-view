
import cv2
import sys
import math
import numpy as np



def wrap(img):  # wrap around the edge pixels
    print("wrapping edge pixels...")
    img.append(img[1])
    for a in range (len(img)-2):
        img[a].insert(0, img[a][-1])
        img[a].append(img[a][1])
    return(img)

def bilinearInterpolation(img, h, w): #img - image, h, w - expected height and width
    
    print("Scaling...")
    
    scaledImg = np.empty([h, w])
    
    initial_h = img.shape[0]
    initial_w = img.shape[1]
    
    xratio = float(initial_w - 1)/(w - 1)
    yratio = float(initial_h - 1)/(h - 1)
    
    for i in range(h):
        for j in range(w):
            xl = math.floor(xratio*j)
            yl = math.floor(yratio*i)
            xh = math.ceil(xratio*j)
            yh = math.ceil(yratio*i)
            
            wx = (xratio * j) - xl
            wy = (yratio * i) - yl
    
            a,b,c,d = img[yl][xl],img[yl][xh],img[yh][xl],img[yh][xh]
            
            x = a*(1-wx) + b*wx
            y = c*(1-wx) + d*wx
            
            zpixel = (x*(1-wy)) + (y*wy)
            
            scaledImg[i][j] = zpixel/255.0
            
    print("Scaling - done!")
    return scaledImg

def normalization(img): #normalizing
    
    normalizedImg = np.empty([len(img), len(img[0])]) #initializing matrix to store normalized image
    
    print("Normalizing...")
    flatten= []
    for i in img:
        for j in i:
            flatten.append(j)
    mean = sum(flatten)/len(flatten)
    
    denominator = max(flatten)-min(flatten)
    
    for a in range(len(img)):
        for b in range(len(img[0])):
            normalizedImg[a][b] = (img[a][b]-mean)/denominator   #normalize individual pixels
    
    #shifting the normalized image
    
    shiftedImg = np.empty([len(img), len(img[0])])
    
    normalized_flatten= []
    for i in normalizedImg:
        for j in i:
            normalized_flatten.append(j)
    min_normalized = min(normalized_flatten)
    for x in range(len(img)):
        for y in range(len(img[0])):
            shiftedImg[x][y] = normalizedImg[x][y] - min_normalized
            
    #contrast streching
    print("Contrast streching...")
    
    contrastStrechedImg = np.empty([len(img), len(img[0])])
    
    shifted_flatten = []
    for i in shiftedImg:
        for j in i:
            shifted_flatten.append(j)
    max_shifted = max(shifted_flatten)
    for u in range(len(img)):
        for v in range(len(img[0])):
            contrastStrechedImg[u][v] = (shiftedImg[u][v] /  max_shifted )
    
    print("Intensity normalized!")
    return contrastStrechedImg

def linearFilter(mask, img):
    
    print("applying linear filter...")
    
    offset = len(mask)//2

    filteredImage = []
    
    for i in range(offset, len(img)-offset):
        filteredRow = []
        for j in range(offset, len(img[0])-offset):
            val = 0
            for x in range(len(mask)):
                for y in range(len(mask)):
                    xn = i + x - offset
                    yn = j + y - offset
                    val += (img[xn][yn] * mask[x][y])
            filteredRow.append(val)
        filteredImage.append(filteredRow)
    print("linear filter applied!")
    return filteredImage

def np_wrap(img):  # wrap around the edge pixels
    print("wrapping edge pixels...")
    np.insert(img, 0, img[-1])
    np.append(img, img[1])
    
    for a in range (len(img)-2):
        np.insert(img, 0, img[a][-1])
        np.append(img, img[a][1])
    return(img)
    
def nonLinearFilter(img, mask): 
    print("applying non linear filter...")
    
    offset = len(mask)//2
    filteredImg = []
    
    #img = np_wrap(img)
    
    for i in range(offset, len(img)-offset):
        filteredRow = []
        for j in range(offset, len(img[0])-offset):
            val = 0
            valarr = []
            for x in range(len(mask)):
                for y in range(len(mask)):
                    xn = i+x-offset
                    yn = j+y-offset
                    val = img[xn][yn]
                    valarr.append(val)
            valarr.sort()
            filteredRow.append(valarr[(len(valarr)-1)//2])
        filteredImg.append(filteredRow)
    print("non linear filter applied!")
    return filteredImg

def getGaussianKernel(sigma, size): # generate the gaussian kernel
    kernel = [[0.0]*size for i in range(size)]
    
    h = size//2
    w = size//2
    
    for x in range(-h, h+1):
        for y in range(-w, w+1):
            normal = 1 / (2.0 * math.pi * sigma**2) 
            hx = math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
 
            kernel[x+h][y+w] = hx*normal
  
    return kernel

def applyFilter(img, mask): #apply filter, mask to img
    offset = len(mask)//2
    outputImg = []

    for i in range(offset, len(img)-offset):
        outputRow = []
        for j in range(offset, len(img[0])-offset):
            val = 0
            for x in range(len(mask)):
                for y in range(len(mask)):
                    xn = i+x-offset
                    yn = j+y-offset
                    val += (img[xn][yn] * mask[x][y])
            outputRow.append(val)
        outputImg.append(outputRow)
    return outputImg

#apply sobel filters
def sobel(img):
    sobelX = [[-1,0,1],[-2,0,2],[-1,0,1]]
    sobelY = [[1,2,1],[0,0,0],[-1,-2,-1]]
    
    #wrapedgradientEstimationX = wrap(img)
    gradientEstimationX = applyFilter(img, sobelX)
    
    #wrapedgradientEstimationY = wrap(gradientEstimationX)
    gradientEstimationY = applyFilter(img, sobelY)
    
    #print(len(gradientEstimationX), len(gradientEstimationX[0]), len(gradientEstimationY), len(gradientEstimationY[0]))
    
    gradArray = []
    epsilon = 0.00000000001
    
    angleMatrixRow = [0] * len(gradientEstimationY[0])
    angleMatrix = []
    for i in range(len(gradientEstimationY)):
        angleMatrix.append(angleMatrixRow)
    
    #generate matrices
    for i in range(len(gradientEstimationX)):
        gradRow = []
        angleRow = []
        for j in range(len(gradientEstimationX[i])):
            
            grad = math.sqrt(gradientEstimationX[i][j]**2 + gradientEstimationY[i][j]**2)
            theta = math.degrees(math.atan((gradientEstimationY[i][j]*1.0)/(gradientEstimationX[i][j]+epsilon)))
            
            gradRow.append(grad)
            angleRow.append(theta)
            
        gradArray.append(gradRow)
        angleMatrix.append(angleRow)
        
    return gradArray, angleMatrix

def nonMaxSuppression(img, angles): #non maxima suppression
  M, N = img.shape
  suppressedImg = np.zeros((M,N))
  for i in range(1,M-1):
      for j in range(1, N - 1):
          if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
              val = max(img[i, j+1], img[i, j-1])
          elif (22.5 <= angles[i, j] < 67.5):
              val = max(img[i+1, j-1], img[i-1, j+1])
          elif (67.5 <= angles[i, j] < 112.5):
              val = max(img[i+1, j], img[i-1, j])
          elif (112.5 <= angles[i,j] < 157.5):
              val = max(img[i-1, j-1], img[i+1, j+1])
          
          if img[i, j] >= val:
            suppressedImg[i, j] = img[i, j]
          else:
            suppressedImg[i, j] = 0

  return suppressedImg

def cannyEdgeDetection(img):
    print("Applying canny edge detection...")
    
    #Filter noise using gaussian filtering
    kernel = getGaussianKernel(1.4,5) #getting the gaussian kernel, parametes - (sigma, size)

    image_height = len(img)
    image_width = len(img[0])
    
    gaussianNoiseFiltered = applyFilter(img, kernel)
    
    wrapped_gaussianNoiseFiltered = np_wrap(gaussianNoiseFiltered)
    gradient, theta = sobel(wrapped_gaussianNoiseFiltered)

    maximum_gradient = 0
    for i in gradient:
        if(max(i)>=maximum_gradient):
            maximum_gradient = max(i)
    
    lowerThreshold = maximum_gradient * 0.025
    upperThreshold = maximum_gradient * 0.5
    
    nonmaxsupImg = nonMaxSuppression(np.array(gradient), np.array(theta))
    
    #apply double threshold
    
    ids =  [[0.0]*image_width]*image_height
    
    for i in range(image_width):
        for j in range(image_height):
            try:
                z = nonmaxsupImg[j][i]
                if z < lowerThreshold:  #ignore
                    nonmaxsupImg[j][i]= 0
                elif upperThreshold > z >= lowerThreshold:  #weak pixel
                    ids[j][i]= 1
                else:   #strong pixel
                    ids[j][i]= 2
            except IndexError:
                pass

    print("Canny edge detection - done!")
    return nonmaxsupImg

#Hough transform

def houghTransform(img):
    print("Applying hough transform...")
    imageheight = len(img)
    imagewidth = len(img[0])
    
    diagonal = math.ceil(math.sqrt(imageheight**2+imagewidth**2))
    
    thetarange =(0, 90)
    rhorange = (-diagonal,diagonal)

    thetaVals1 = [i for i in range(120,150)]
    thetaVals2 = [i for i in range(30,60)]
    
    thetaVals = thetaVals1 + thetaVals2

    houghspace = [[0]*(thetarange[1]-thetarange[0])]*(rhorange[1]-rhorange[0])

    for i in range(imageheight):
        for j in range(imagewidth):
            pixelVal = img[i][j]
            if(pixelVal!=0):
                for theta in thetaVals:
                    rhoVal = round(j*math.cos(math.radians(theta)) + i*math.sin(math.radians(theta)))
                    houghspace[rhoVal][theta] += 1  #increase hits

    hits = []
    
    for row in range(len(houghspace)):      
        for i in  range(len(houghspace[row])):
            hits.append((row,i,houghspace[row][i]))

    thresh = 70
    maxHoughSpace = list(filter(lambda x: x[2]>=thresh, hits))  #if number of hits are greater than the threshold
    output = [[0]*imagewidth for i in range(imageheight)]
    
    for i in range(imageheight):
        for j in range(imagewidth):
            print(i,j)
            pixel = img[i][j]
            if(pixel!=0):
                for k in maxHoughSpace:
                    rho = round(j*math.cos(math.radians(k[1])) + i*math.sin(math.radians(k[1])))
                    if(rho == k[0]):
                        output[i][j] = 255
                        
    print("Hough transform applied!")
    return output

#mark the lane in red - not functioning properly but attempted
def drawRedLine(greyScale ,houghOutput):
    print("drawing red lines...")
    markedImg = []
    
    #had an isue with the image resolutions, didn't have enough time to look into it.. so I handled it this way
    greyScaleImg = []
    greyScaleImgEdited = greyScale[6:len(greyScale)-6]
    for row in greyScaleImgEdited:
        row = row[6:len(row)-6]
        greyScaleImg.append(row)

    '''print(len(greyScaleImg), len(greyScaleImg[0]))
    print(len(houghOutput), len(houghOutput[0]))'''

    for i in range(len(greyScaleImg)):
        row = []
        for j in range(len(greyScaleImg[i])):
            try:
                row.append([greyScaleImg[i][j],greyScaleImg[i][j],greyScaleImg[i][j]+255*houghOutput[i][j]])
            except IndexError:
                pass
        markedImg.append(row)
    print("Drew the lines!")
    return markedImg

#segmentation - incomplete
def regionGrow(img,seeds,thresh):
    imgh, imgw = img.shape
    seedmark = np.zeros(imgh, imgw)
    labels =  [[0,0,255], [255,255,0], [0,255,0]]

    while(len(seeds)>0):
        growpoint = seeds.pop(0)
    
    seedmark[growpoint[0], growpoint[1]] = 1
    
    return seedmark

#main function
def main_170214B():
    filename = "dashcam_view_1" + ".JPG"   #enter image name here
    #filename = "low" + ".JPG" 

    img = cv2.imread(filename, 0)   # image will process after converting it to grey scale
    
    if(img is None):
        print("Invalid image name")
        sys.exit()
        
    cv2.imshow("170214B - input image - img", img)
    cv2.waitKey(0)
    
    #apply bilinear interpolation
    img_h = img.shape[0]
    img_w = img.shape[1]
    scaled_image = bilinearInterpolation(img, int(img_h*0.8), int(img_w*0.8))
    np_scaledImage = np.array(scaled_image)
    cv2.imshow("170214B - scaled image - scaled_image", np_scaledImage)
    cv2.waitKey(0)
    #cv2.imwrite("scaled image.jpg", np_scaledImage)

    #normalizing intensity
    normalized = normalization(scaled_image)
    np_normalizedImage = np.array(normalized)
    cv2.imshow("170214B - normalized image - intensity_normalized", np_normalizedImage)
    cv2.waitKey(0)
    
    #non linear filter (median) - because there are irregularities in the white lines on the road (salt and pepper noise), the median filter is applied to remove salt and pepper noise 
    median_mask = [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]] #size 5
    #median_mask = [[1,1,1],[1,1,1],[1,1,1]] #size 3
    wrapped_normalized = np_wrap(normalized)
    nonlinear_filter_applied_image = nonLinearFilter(wrapped_normalized, median_mask)
    np_nonlinear_filter_applied = np.array(nonlinear_filter_applied_image)
    cv2.imshow("170214B - non linear filter applied image - nonlinear_filter_applied_image", np_nonlinear_filter_applied)
    cv2.waitKey(0)
    #cv2.imwrite("non linear filter applied image.jpg", np_nonlinear_filter_applied)
    
    #applying a linear filter (gaussian filter)
    #gaussian_mask = [[0.0625, 0.125, 0.0625],[0.125, 0.25, 0.125],[0.0625, 0.125, 0.0625]]  #3*3 gaussian kernel
    gaussian_mask = getGaussianKernel(1.5, 3) 
    #linear_filter_applied = linearFilter(gaussian_mask, scaled_image)
    wrapped_nonlinear_filter_applied_image = np_wrap(nonlinear_filter_applied_image)
    linear_filter_applied = linearFilter(gaussian_mask, wrapped_nonlinear_filter_applied_image)
    np_linear_filter_applied = np.array(linear_filter_applied)
    cv2.imshow("170214B - linear filter applied image - linear_filter_applied", np_linear_filter_applied)
    cv2.waitKey(0)
    #cv2.imwrite("linear filter applied image.jpg", np_linear_filter_applied)
    
    wrapped_linear_filter_applied = np_wrap(linear_filter_applied)
    cannyEdgeDetectionAdded_img = cannyEdgeDetection(wrapped_linear_filter_applied)   #apply canny edge detection
    np_cannyEdgeDetectionAdded_img = np.array(cannyEdgeDetectionAdded_img)
    cv2.imshow("170214B - Canny edge detection added Image", np_cannyEdgeDetectionAdded_img)
    cv2.waitKey(0)
    
    houghTransformAppliedImg = houghTransform(cannyEdgeDetectionAdded_img)
    
    np_hough_applied = np.array(houghTransformAppliedImg, dtype=np.uint8)
    cv2.imshow("170214B - hough transform", np_hough_applied)
    cv2.waitKey(0)
    
    redLineMarkedImg = drawRedLine(scaled_image, houghTransformAppliedImg)
    np_redmark = np.array(redLineMarkedImg)
    cv2.imshow("170214B - lanes marked in red image", np_redmark)
    cv2.waitKey(0)
        
            
main_170214B()