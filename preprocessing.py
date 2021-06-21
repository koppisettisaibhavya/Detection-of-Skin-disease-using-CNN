import cv2,os
for f in os.listdir("C:\\Users\\Admin\\Data\\hair_img"):
    n="C:\\Users\\Admin\\Data\\hair_img\\"+f

    #loading images from directory
    src = cv2.imread(n)
    #converting to grayscale
    grayscale = cv2.cvtColor( src, cv2.COLOR_RGB2GRAY )

    #defining structuring element
    kernel_1 = cv2.getStructuringElement(1,(17,17))

   #applying black-hat transform
    black_hat = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, kernel_1)

   #performing thresholding operation
    r,thr = cv2.threshold(black_hat,10,255,cv2.THRESH_BINARY)

    #applying inpainting algorithm
    dest = cv2.inpaint(src,thr,1,cv2.INPAINT_TELEA)

    #storing the image back to dataset
          cv2.imwrite("C:\\Users\\Admin\\Data\\HAM10000_images_part_1\\"+f,dest,
                                                                                   [int(cv2.IMWRITE_JPEG_QUALITY), 90])
