import os
import cv2
import numpy as np
import seaborn as sns
import random
import matplotlib.pyplot as plt
%matplotlib inline






positive_srcdir = './dataset/Positive'
negative_srcdir = './dataset/Negative'


# positive_srcdir = './DetectSurfaceOriginal/Positive'
# negative_srcdir = './DetectSurfaceOriginal/Negative'
all_images = [positive_srcdir, negative_srcdir]


positive_images = len(os.listdir(positive_srcdir))
negative_images = len(os.listdir(negative_srcdir))
total_images = negative_images + positive_images


nums = []
for i in range(positive_images):
    nums.append("Postive")
for i in range(negative_images):
    nums.append("Negative")
np.array(nums).shape

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
sns.set_style('darkgrid')
axl = sns.displot(nums)
# axl.set_title("Number of Test Data")

nums[19998:20006]



# 2/

rows=4
cols = 4
img_count = 0

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15,15))

for i in range(rows):
    for j in range(cols):        
        if img_count < positive_images:
            axes[i, j].imshow(\
            cv2.imread(os.path.join(positive_srcdir, os.listdir(positive_srcdir)[img_count])))
            img_count+=1
            
rows=4
cols = 4
img_count = 0

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15,15))

for i in range(rows):
    for j in range(cols):        
        if img_count < negative_images:
            axes[i, j].imshow(\
            cv2.imread(os.path.join(negative_srcdir, os.listdir(negative_srcdir)[img_count])))
            img_count+=1
            
            
os.listdir(positive_srcdir)[40]


os.path.join(positive_srcdir, os.listdir(positive_srcdir)[40])


################test show image ############
def show_image(image, gray=False):
  plt.figure()
  if gray:
    plt.imshow(image, cmap='gray')
  else:
    plt.imshow(image)
  plt.axis("off")
  plt.show()

temp_img = cv2.cvtColor(gabor, cv2.COLOR_BGR2RGB)
show_image(temp_img)


test_contour= cv2.drawContours(blur, contours_, -1, (0, 255, 0), 2)
test_contour = cv2.cvtColor(test_contour, cv2.COLOR_BGR2RGB)
show_image(test_contour,True)
###############



all =[]
#read image
img=[]  ## bao gồm ảnh có vết nứt và không có vết nứt
gabors = []   ##bao gồm các ảnh làm mờ và detect feature cạnh (theo các góc khác nhau)
grays = []    ##bao gồm các ảnh xám (227,227) đã blur và gabor
blurred = []    ## bao gồm các ảnh sau khi làm mờ để giảm nhiễu cho ảnh
threshInv = []  ##bao gồm các ảnh có các giá trị vượt ngưỡng chỉ có ngưỡng 0 hoặc 255
contours = []   ## bao gồm số điểm tối đa để bao feature vết nứt dài nhất trong ảnh

img.append(cv2.imread(os.path.join(positive_srcdir, os.listdir(positive_srcdir)[40])))
img.append(cv2.imread(os.path.join(negative_srcdir, os.listdir(negative_srcdir)[40]))) 

# plt.imshow(cv2.cvtColor(cv2.imread(os.path.join(positive_srcdir, os.listdir(positive_srcdir)[40])), cv2.COLOR_BGR2RGB ))

### actual
# for j in range(2):
#     for i in range(4):        
#         retval = cv2.getGaborKernel(ksize = (5,5), sigma=10, theta=45*i, lambd=5, gamma=1)    ## detect cạnh và feature bawfg các kernel theo các góc lọc khác nhau 
#         blur = cv2.GaussianBlur(img[j], (9, 9), 0)    ##lam mo anh để giảm nhiễu ảnh và giảm chi tiết.
#         blurred.append(blur)   
#         gabor = cv2.filter2D(blur, -1, retval) ## giam nhieu và detect canh 
#         gabors.append(gabor)
#         gray =  cv2.cvtColor(gabor, cv2.COLOR_BGR2GRAY)   ##ảnh xám sau khi đã làm mờ và lọc cạnh
#         grays.append(gray)
#         (T, threshInv_) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)       ##>127 có giá trị 255(màu đen) , <127: giá trị là 0 (màu trắng)
#         threshInv.append(threshInv_)
#         contours_, h = cv2.findContours(threshInv_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#         if contours_:
#             max_len_cnt = max([len(x) for x in contours_])
#             contours.append(max_len_cnt)
#         else:
#             contours.append(0)
    
##test góc gabor 0 - 180 độ
for j in range(2):
    for i in range(6):        
        retval = cv2.getGaborKernel(ksize = (5,5), sigma=10, theta=30*i, lambd=5, gamma=1)    ## detect cạnh và feature bawfg các kernel theo các góc lọc khác nhau 
        blur = cv2.GaussianBlur(img[j], (9, 9), 0)    ##lam mo anh để giảm nhiễu ảnh và giảm chi tiết.
        blurred.append(blur)   
        gabor = cv2.filter2D(blur, -1, retval) ## giam nhieu và detect canh 
        gabors.append(gabor)
        gray =  cv2.cvtColor(gabor, cv2.COLOR_BGR2GRAY)   ##ảnh xám sau khi đã làm mờ và lọc cạnh
        grays.append(gray)
        (T, threshInv_) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)       ##>127 có giá trị 255(màu đen) , <127: giá trị là 0 (màu trắng)
        threshInv.append(threshInv_)
        contours_, h = cv2.findContours(threshInv_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours_:
            max_len_cnt = max([len(x) for x in contours_])
            contours.append(max_len_cnt)
        else:
            contours.append(0)

all.append(img)
all.append(blurred)
all.append(gabors)
all.append(grays)
all.append(threshInv)
all.append(contours)

# contours

# plt.clf()
# plt.figure()
# plt.imshow(all[5][4])
# plt.show()

rows = 12
cols = 5
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15,15))

for i in range(rows):
    print(all[5][i])
    for j in range(5):
        if j==0:
            axes[i][j].imshow(all[j][int(i/(rows/2))])
        else:
            axes[i][j].imshow(all[j][i])

        
# 3/ Tiến hành phân loại và phân tích kết quả

def detect_count(image_name, loc_thres):   ## dectect tổng số điểm bao vết nứt cho 4 góc gabor
    #read image
    image = cv2.imread(image_name)
    count=0
    for i in range(4):
        retval = cv2.getGaborKernel(ksize = (5,5), sigma=10, theta=45*i, lambd=5, gamma=1)
        blur = cv2.GaussianBlur(image, (9, 9), 0)
        # gabor = cv2.filter2D(blur, -1, retval)   ##edit
        gabor = cv2.filter2D(image, -1, retval)
        gray =  cv2.cvtColor(gabor, cv2.COLOR_BGR2GRAY)
        (T, threshInv_) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours_, h = cv2.findContours(threshInv_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours_:
            max_len_cnt = max([len(x) for x in contours_])
            if max_len_cnt >= loc_thres:#loại bỏ tiếng ồn nhỏ
                count += max_len_cnt
    
    return count

# all_images

# random.randint(0,1)

# all_images[0].endswith("Positive")


mincount = 10000
maxcount = 0
poscou = []    ## lưu lại crack status (tổng số điểm vết nứt của 4 loại phép chiếu gabor) cho 500 tấm ảnh vết nứt
negcou = []         # lưu lại crack status (tổng số điểm vết nứt của 4 loại phép chiếu gabor) cho 500 tấm ảnh ko vết nứt
mintot = 1000

#looping over both class
for classes in all_images:
    
    # classes = all_images[0]
    count = 476                             ## test mỗi loại 500 ảnh 
    for files in os.listdir(classes):    ## loop qua 20000 ảnh vêt nứt và ko vết nứt  
        # files = os.listdir(classes)[0]
        # if random.randint(0,3) == 2:    ##
        #     continue
        image_name = os.path.join(classes, files)
        crack_status = detect_count(image_name,65)
        # ############# test detect_count(image_name,65) 
        # image_name=image_name
        # loc_thres=65
        # image = cv2.imread(image_name)
        # count=0
        # for i in range(4):
        #     i=0
        #     retval = cv2.getGaborKernel(ksize = (5,5), sigma=10, theta=45*i, lambd=5, gamma=1)
        #     blur = cv2.GaussianBlur(image, (9, 9), 0)
        #     gabor = cv2.filter2D(blur, -1, retval)
        #     gray =  cv2.cvtColor(gabor, cv2.COLOR_BGR2GRAY)
        #     (T, threshInv_) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        #     contours_, h = cv2.findContours(threshInv_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #     if contours_:
        #         max_len_cnt = max([len(x) for x in contours_])
        #         if max_len_cnt >= loc_thres:#loại bỏ tiếng ồn nhỏ
        #             count += max_len_cnt    ## ảnh nào có điểm ảnh bao vết nứt >65 điểm thì cộng tổng số điểm bao vết nứt đó vô count
                    
        # crack_status =count
        # #################
        
        if classes.endswith('Positive'):
            poscou.append(crack_status)
        else:
            negcou.append(crack_status)
        
        count-=1
        if count==0:
            break

print(mincount, maxcount)
poscou = np.array(poscou)

negcou = np.array(negcou)

print(poscou.mean(),negcou.mean())

# len(negcou)


## 500 sample and 500 neg sample
mintot = 10000
for thres in range(5,100,1):
    # thres = 5
    samples = 0
    nsamples = 0    
    for i in range(len(poscou)):
        # i=0
        if poscou[i] >= thres:
            samples+=1                                          ### số ảnh có vết nứt dự đoán  đúng
        if negcou[i] >= thres:              
            nsamples+=1                         ### số ảnh có không vết nứt dự đoán sai
    tot = 476 - samples + nsamples ###### tot = 0 đúng tuyệt đối, tot càng lớn thì lỗi càng nhiều
    if tot<mintot:
        mintot = tot
        ls = samples                ## số ảnh dự đoán là có vết nứt
        ln = 476-nsamples           ## số ảnh dự đoán là không có vết nứt
        
        print('thresho:',thres,'\tpositive:',samples,'\tnegative:',476-nsamples,'\ttot:',tot)
        
mintot = 10000 
for thres in range(100,800,5):
    samples = 0
    nsamples = 0    
    for i in range(len(poscou)):
        if poscou[i] >= thres:
            samples+=1
        if negcou[i] >= thres:
            nsamples+=1
    tot = 500 - samples + nsamples
    if tot<mintot:
        mintot = tot
        ls = samples;
        ln = 500-nsamples
        
    print('thresho:',thres,'\tpositive:',samples,'\tnegative:',500-nsamples,'\ttot:',tot)
    
    
ln



acur = (ls + ln)/(len(poscou) + len(negcou)) * 100
pre = ls / (ls + 500-ln) * 100
recall = ls / len(poscou) * 100
f1 = (2*pre*recall)/(pre+recall)
print(' 准确率:',acur,'\n','精确率:',pre,'\n','召回率:',recall,'\n','F1分数:',f1)


########## avg_thres should 20
# def detect_crack(image_name, contour_threshold=210, local_threshold = 50, avg_thres=50):
def detect_crack(image_name, contour_threshold=105, local_threshold = 50, avg_thres=20):   ##edit
    #read image
    image = cv2.imread(image_name)
    count = 0
    avg = 0
    thres = []
    for i in range(4):
        retval = cv2.getGaborKernel(ksize = (5,5), sigma=10, theta=45*i, lambd=5, gamma=1)
        gabor = cv2.filter2D(image, -1, retval)
        gray =  cv2.cvtColor(gabor, cv2.COLOR_BGR2GRAY)
        (T, threshInv_) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        thres.append(threshInv_)
        
    threshInv_ = thres[0] | thres[1] | thres[2] | thres[3]
    contours_, h = cv2.findContours(threshInv_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours_:
        data = [len(x) for x in contours_]
        data = np.array(data)
        # count = data.max()
        # avg = data.mean()
        #### edit
        data= data[data>avg_thres]
        if (len(data)==0):
            count=0
        else:
            count = data.max()
                
    # if (count >= contour_threshold) and (avg >= avg_thres):
    if (count >= contour_threshold):
        return True
    return False


cv2.imread("./DetectSurfaceOriginal/Positive/00001.jpg").shape






detect_crack("./DetectSurfaceOriginal/Positive/00682.jpg")

############# TEST detect_crack()
contour_threshold=210
local_threshold = 50
avg_thres=50 ##############26.777
image = cv2.imread("./DetectSurfaceOriginal/Positive/00001.jpg")
count = 0
avg = 0
thres = []
for i in range(4):
    retval = cv2.getGaborKernel(ksize = (5,5), sigma=10, theta=45*i, lambd=5, gamma=1)
    gabor = cv2.filter2D(image, -1, retval)
    gray =  cv2.cvtColor(gabor, cv2.COLOR_BGR2GRAY)
    (T, threshInv_) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    thres.append(threshInv_)
    
threshInv_ = thres[0] | thres[1] | thres[2] | thres[3]
contours_, h = cv2.findContours(threshInv_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
if contours_:
    data = [len(x) for x in contours_]
    data = np.array(data)
    count = data.max()
    avg = data.mean()
            
if (count >= contour_threshold) and (avg >= avg_thres):
    print(True)
print(False)


############






#define dictionary to store class result
class_result = dict()
#contour_threshold is set to 250
loc_thres = 15
contour_threshold = 210


######### test for 5000 images crack and 5000 image with no crack
#looping over both class
for classes in all_images:
    class_count = 0
    imgcount = 0
    for files in os.listdir(classes):
        image_name = os.path.join(classes, files)
        crack_status = detect_crack(image_name, contour_threshold,loc_thres)
        if crack_status:
            class_count+=1
        imgcount += 1
        if imgcount == 5000:
            break
    class_result[os.path.basename(classes)] = class_count 
total = 5000 - class_result['Positive'] + class_result['Negative']
print(class_result,total)


#accuracy
true_positive = class_result['Positive']
true_negative = 5000 - class_result['Negative'] ####### test for 5000 image per class so pass 5000, else pass total image (20000)
# accuracy = (true_positive + true_negative) / total_images
accuracy = (true_positive + true_negative) / 10000
print(f'Accuracy is {round(accuracy*100, 2)}%')


# def detect_direct(image_name):
#     #read image
#     image = cv2.imread(image_name)
#     count = 0
#     direction = -1
#     for i in range(4):
#         retval = cv2.getGaborKernel(ksize = (5,5), sigma=10, theta=45*i, lambd=5, gamma=1)
#         gabor = cv2.filter2D(image, -1, retval)
#         gray =  cv2.cvtColor(gabor, cv2.COLOR_BGR2GRAY)
#         (T, threshInv_) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#         contours_, h = cv2.findContours(threshInv_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#         if contours_:
#             max_len_cnt = max([len(x) for x in contours_])
#             if max_len_cnt >= 20: #Xóa tiếng ồn nhỏ
#                 if max_len_cnt > count:
#                     count = max_len_cnt
#                     direction = i
    
#     return count,direction

# ################# test  detect_direct(image_name)

# image = cv2.imread("./DetectSurfaceOriginal/Positive/00001.jpg")
# count = 0
# direction = -1
# for i in range(4):
#     retval = cv2.getGaborKernel(ksize = (5,5), sigma=10, theta=45*i, lambd=5, gamma=1)
#     gabor = cv2.filter2D(image, -1, retval)
#     gray =  cv2.cvtColor(gabor, cv2.COLOR_BGR2GRAY)
#     (T, threshInv_) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#     contours_, h = cv2.findContours(threshInv_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     if contours_:
#         max_len_cnt = max([len(x) for x in contours_])
#         if max_len_cnt >= 20:#去除小的噪点
#             if max_len_cnt > count:
#                 count = max_len_cnt
#                 direction = i
# print(count,direction)
# #########################


# all =[]
# #read image
# impath = []
# counts = []
# directions = []
# num = 0

# for files in os.listdir(positive_srcdir):
#     image_name = os.path.join(positive_srcdir, files)
#     count,direction = detect_direct(image_name)
    
#     counts.append(count)
#     directions.append(direction)
#     impath.append(image_name)
    
#     num += 1
#     if num==16:
#         break
    
    

# all.append(impath)
# all.append(counts)
# all.append(directions)


# len(counts)

# rows=4
# cols = 4
# img_count = 0

# fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15,15))

# for i in range(rows):
#     for j in range(cols):    
#         print('Hướng：{}°角\t Số đường bao theo hướng này là：{}'.format(all[2][i+j]*45,all[1][i+j]))
#         if img_count < 16:
#             axes[i, j].imshow(\
#             cv2.imread(impath[img_count]))
#             img_count+=1
            
# #define dictionary to store class result
# class_result = dict()

# directs =[0,0,0,0,0]
# #looping over positive class

# for files in os.listdir(positive_srcdir):
#     image_name = os.path.join(positive_srcdir, files)
#     count, direct = detect_direct(image_name)
#     if direct == -1:
#         directs[4] += 1
#     else:
#         directs[direct] += 1

# for i in range(4):   
#     class_result['{}°'.format(i*45)] = directs[i] 
    
# class_result['Not Found'] = directs[4]


# class_result

# res = []

# for i in class_result:
#     print(i)
#     for j in range(class_result[i]):
#         res.append(i)
        
        
# res[10000]

# plt.figure(figsize=(20, 20))
# plt.subplot(2, 2, 1)
# sns.set_style('darkgrid')
# axl = sns.displot(res)
# # axl.set_title("Number of Directions")





























