import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

#funções

def RGB_to_YIQ(img):
    imgRGBf = img/255.0
    A = np.array([[0.299, 0.587, 0.114],
             [0.5959, -0.2746, -0.3213],
             [ 0.2115, -0.5227, 0.3112]])
    imgYIQ = np.dot(imgRGBf,A)
    return imgYIQ

def YIQ_to_RGB(img):
    B = np.array([[ 1.0, 0.956, 0.619],
             [1.0, -0.272, -0.647],
             [1.0, -1.106, 1.703]])
    imgRGBtrans = np.dot(img,B)
    return imgRGBtrans

def RGBneg(img):
    b,g,r = cv.split(img)
    b=255-b
    g=255-g
    r=255-r
    inversa = cv.merge([b,g,r])
    return inversa

def YIQneg(img):
    y,i,q= cv.split(img)
    y1 = 1 - y
    inver = cv.merge([y1,i,q])
    return inver

def mean(img,filter,pivo):
    output = np.empty_like(img)

    width = output.shape[0]
    height = output.shape[1]

    for w in range(width):
        # filter's rows
        row_start = w - (filter.shape[0] + pivo[0][0])
        row_end = w + (filter.shape[0] - pivo[0][0])
        # Check invalid index
        if row_start < 0:
            row_start = w
        if row_end > width:
            row_end = width
        # Columns
        for h in range(height):
            # filter's columns
            column_start = h - (filter.shape[1] + pivo[0][1])
            column_end = h + (filter.shape[1] + pivo[0][1])

            # Check invalid index
            if column_start < 0:
                column_start = 0
            if column_end > height:
                column_end = height

            # Calculate R, G and B according to filter function type
            output[w, h, 2] = np.mean(imgRGB[row_start: row_end, column_start: column_end, 2]).astype(int)# R
            output[w, h, 1] = np.mean(imgRGB[row_start: row_end, column_start: column_end, 1]).astype(int)# G
            output[w, h, 0] = np.mean(imgRGB[row_start: row_end, column_start: column_end, 0]).astype(int) # B
    return output 

def applyOffset(img,offset):
    r,g,b = cv.split(img)
    width =img.shape[0]
    height = img.shape[1]

    for w in range(width):
        for h in range(height):
            # The new brightness according to function argument "calculate"
            if offset < 0:
                if np.absolute(offset) > r[w][h]:
                    r[w][h] = 0
                else:
                    r[w][h] = r[w][h]+ offset
                if np.absolute(offset) > g[w][h]:
                    g[w][h] = 0
                else:
                    g[w][h] = g[w][h]+ offset
                if np.absolute(offset) > b[w][h]:
                    b[w][h] = 0
                else:
                    b[w][h] = b[w][h]+ offset
            else:
                if  (r[w][h]+ offset)>255:
                    r[w][h] = 255
                else:
                    r[w][h] = r[w][h]+ offset
                if  (g[w][h]+ offset)>255:
                    g[w][h] = 255
                else:
                    g[w][h] = g[w][h]+ offset
                if  (b[w][h]+ offset)>255:
                    b[w][h] = 255
                else:
                    b[w][h] = b[w][h]+ offset
            img[w][h][2] = r[w][h]
            img[w][h][1] = g[w][h]
            img[w][h][0] = b[w][h]
    img = cv.merge([r,g,b])
    return img

def calc_sobel_x(img):
    sobx = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    return img * sobx

def calc_sobel_y(img):
    soby = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    return img * soby

def sobelx(img, dim):
    saida = np.empty_like(img)
    width = saida.shape[0]
    height = saida.shape[1]
    for w in range(dim, width - dim):
 
        for h in range(dim, height - dim):
            

            saida[w][h][2] = np.mean(calc_sobel_x(img[w:w+dim, h:h+dim,2])).astype(int)
            saida[w][h][1] = np.mean(calc_sobel_x(img[w:w+dim, h:h+dim,1])).astype(int)
            saida[w][h][0] = np.mean(calc_sobel_x(img[w:w+dim, h:h+dim,0])).astype(int)

    return saida

def sobely(img, dim):
    saida = np.empty_like(img)
    width = saida.shape[0]
    height = saida.shape[1]
    for w in range(dim, width - dim):
 
        for h in range(dim, height - dim):
            

            saida[w][h][2] = np.mean(calc_sobel_y(img[w:w+dim, h:h+dim,2])).astype(int)
            saida[w][h][1] = np.mean(calc_sobel_y(img[w:w+dim, h:h+dim,1])).astype(int)
            saida[w][h][0] = np.mean(calc_sobel_y(img[w:w+dim, h:h+dim,0])).astype(int)

    return saida

def YIQfilter(img,filter):
    output = np.empty_like(imgRGB)
    y,i,q=cv.split(img)
    img = img*255
    width = output.shape[0]
    height = output.shape[1]

    for w in range(width):
        # filter's rows
        row_start = w - (filter.shape[0] + 1)
        row_end = w + (filter.shape[0] - 1)
        # Check invalid index
        if row_start < 0:
            row_start = w
        if row_end > width:
            row_end = width
        # Columns
        for h in range(height):
            # filter's columns
            column_start = h - (filter.shape[1] + 1)
            column_end = h + (filter.shape[1] + 1)

                # Check invalid index
            if column_start < 0:
                column_start = 0
            if column_end > height:
                column_end = height

                # Calculate R, G and B according to filter function type
            output[w, h,0] = np.median(img[row_start: row_end, column_start: column_end,0]).astype(int)# Y
    output[:, :,1] = i
    output[:, :,2] = q
    return output
#Parametros
imgRGB = cv.imread('./Imagens/Woman.png')
imgYIQ = RGB_to_YIQ(imgRGB)

'''
#RGB to YIQ to RGB
img = YIQ_to_RGB(imgYIQ)
cv.imshow('YIQ', imgYIQ)  
cv.imshow('RGBtoYIQtoRGB', img)   
cv.waitKey(0)
'''

'''
#Negative RGB
img = RGBneg(imgRGB)
cv.imshow('Negative RGB', img)   
cv.waitKey(0)
'''

'''
#Negative YIQ
img = YIQneg(imgYIQ)
img = YIQ_to_RGB(img)
cv.imshow('Negative YIQ', img)   
cv.waitKey(0)
'''


'''
#Mean RGB filter
file = open('media3x3.txt','r')
lines = []
lines = file.readlines()
pivo = []
piv = 0
offset = 0
count = 0
filter = []

#leitura do filtro
for line in lines:
    count+=1
    if count == 1:
        continue
    if line == 'PIVO\n':
        break
    filter.append([float(num) for num in line.split(',')])
filter = np.array(filter)
#leitura do pivo
for line in lines:
    if line == 'OFFSET\n':
        break
    if piv == 1:
        pivo.append([int(num) for num in line.split(',')])   
    if line == 'PIVO\n':
        piv = 1


#leitura do OFFSET
for line in lines:   
    if piv == 0:
        offset = int(line)
    if line == 'OFFSET\n':
        piv = 0

pivo = np.array(pivo)

img = mean(imgRGB,filter,pivo)
img = applyOffset(img, offset)
cv.imshow('Mean Filter', img)   
cv.waitKey(0)
'''

'''
#Applying Sobel Filter

x_sobel = sobelx(imgRGB, 3)
y_sobel = sobely(imgRGB, 3)
cv.imshow('Sobel X', x_sobel)
cv.imshow('Sobel Y', y_sobel)
cv.waitKey(0)
'''

'''
#Applying YIQ filter
filter = np.array([[1,1,1,1,1],
                  [1,1,1,1,1],
                  [1,1,1,1,1],
                  [1,1,1,1,1],
                  [1,1,1,1,1]])
img = YIQfilter(imgYIQ,filter)
img = YIQ_to_RGB(img)/255
cv.imshow('YIQ median filter', img) 
cv.waitKey(0)
'''


#Correlaton between Woman_eye.png and Woman.png
template = cv.imread('./imagens/Woman_eye.png')
woman = cv.imread('./imagens/Woman.png') 
woman2=woman
woman3=cv.imread('./imagens/Woman.png')
height, width, channels = template.shape  
method = eval('cv.TM_CCORR_NORMED') 
res = cv.matchTemplate(woman, template, method)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + width, top_left[1] + height)
woman = cv.cvtColor(woman, cv.COLOR_BGR2GRAY)
cv.rectangle(woman,top_left, bottom_right, 255, 2)

woman2[184 : 217,247: 291] = (0, 0, 0)

res = cv.matchTemplate(woman2, template, method)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + width, top_left[1] + height)
woman3 = cv.cvtColor(woman3, cv.COLOR_BGR2GRAY)
cv.rectangle(woman3,top_left, bottom_right, 255, 2)

plt.subplot(121),plt.imshow(woman,cmap = 'gray')
plt.title('Max Correlation Region'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(woman3,cmap = 'gray')
plt.title('Second Max Correlation Region'), plt.xticks([]), plt.yticks([])
plt.suptitle('Normalized Correlation')
plt.show()
