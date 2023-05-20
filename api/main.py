
import numpy as np
from PIL import Image
from keras.models import load_model
from utils import load_unet_vgg16, load_unet_resnet_101, load_unet_resnet_34
import cv2
from torch.autograd import Variable
from unet.unet_transfer import UNet16, input_size
import torchvision.transforms as transforms
import base64
import io
import os
import matplotlib.pyplot as plt
import glob
import requests
import json
import torch
import datetime
import pytz



# image_test = cv2.imread("./test_imgs/negative/noncrack_noncrack_concrete_wall_74_27.jpg.jpg")

# t1= np.asarray(im_file)
# t2= cv2.cvtColor(t1, cv2.COLOR_RGB2BGR)
# t3= t2 == image_test

### save txt test file:::
# test_file= "test_imgs/peeling"
# save_txtfile="file_test/test/peeling"
# d = os.listdir(test_file)
# for i in range(len(d)):
#     with open(os.path.join(test_file, d[i]), "rb") as img_file:
#         my_string = base64.b64encode(img_file.read())
#     f = open(os.path.join(save_txtfile,"sample"+str(i)+".txt"), "wb")
#     f.write(my_string)
#     f.close()



def detect_crack(im_file, loc_thres=39, thres=49):   ## dectect tổng số điểm bao vết nứt cho 4 góc gabor
    #read image
    image= cv2.resize(cv2.cvtColor(np.asarray(im_file), cv2.COLOR_RGB2BGR) , (448,448) , interpolation=cv2.INTER_AREA)
    count=0
    for i in range(6):
        retval = cv2.getGaborKernel(ksize = (5,5), sigma=10, theta=30*i, lambd=5, gamma=1)
        blur = cv2.GaussianBlur(image, (9, 9), 0)
        gabor = cv2.filter2D(image, -1, retval) ## giam nhieu và detect canh 
        # gabor = cv2.filter2D(blur, -1, retval)   ##edit
        # gabor = cv2.filter2D(image, -1, retval)
        gray =  cv2.cvtColor(gabor, cv2.COLOR_BGR2GRAY)
        (T, threshInv_) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours_, h = cv2.findContours(threshInv_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours_:
            max_len_cnt = max([len(x) for x in contours_])
            if max_len_cnt >= loc_thres:#loại bỏ tiếng ồn nhỏ
                count += max_len_cnt
    crack_status = count
    if (crack_status >= thres):
        return True
    return False

#Load the trained model. (Pickle file)
classify_model = load_model('./Models/VGG16_batchsize=6_final.h5')

def TypePrediction(im_file):
    SIZE = 150 
    # im_bytes = base64.b64decode(my_string)   # im_bytes is a binary image
    # im_file = io.BytesIO(im_bytes)  # convert image to file-like object
    # im_file=Image.open(im_file)
    img = np.asarray(im_file.resize((SIZE,SIZE)))
    
    # img_path="https://lptech.asia/uploads/files/2020/07/10/seo-hinh-anh-la-gi-lptech.png"
    # urllib.request.urlretrieve(img_path, "PredictImage")
    # img = np.asarray(Image.open("PredictImage").resize((SIZE,SIZE)))
    
    img = img/255.      #Scale pixel values
    
    img = np.expand_dims(img, axis=0)  #Get it tready as input to the network       
    
    pred = classify_model.predict(img) #Predict 
    prob = np.array(pred[0]) 
    classes = ["Peeling","Crack", "Normal"]

    print(pred[0])
    print("-"*50)
    print((round(prob[np.argmax(prob)],4), classes[np.argmax(prob)]) )
    return (round(prob[np.argmax(prob)],4),classes[np.argmax(prob)]) 


# # model predict segmentation:
    
##load model:
model_path ="Models/model_unet_vgg_16_best.pt"
model_type='vgg16'   
if model_type == 'vgg16':
    model = load_unet_vgg16(model_path)
elif model_type  == 'resnet101':
    model = load_unet_resnet_101(model_path)
elif model_type  == 'resnet34':
    model = load_unet_resnet_34(model_path)
    print(model)
else:
    print('undefind model name pattern')
    exit()
    
def evaluate_img(model, img):
    input_width, input_height = input_size[0], input_size[1]
    img_height, img_width, img_channels = img.shape
    img_1 = cv2.resize(img, (input_width, input_height), cv2.INTER_AREA)
    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])

    X = train_tfms(Image.fromarray(img_1))
    # X = Variable(X.unsqueeze(0)).cuda()  # [N, 1, H, W]
    X = Variable(X.unsqueeze(0)).cpu()

    mask = model(X)

    mask = torch.sigmoid(mask[0, 0]).data.cpu().numpy()
    mask = cv2.resize(mask, (img_width, img_height), cv2.INTER_AREA)
    return mask


# def find_name():
#     arr_img=glob.glob("./test_imgs/*")
#     arr_img=list(map(lambda x: x.rsplit("\\",1)[-1] ,arr_img))
#     arr_num= list(map(int,filter(lambda t: t.isnumeric(),map(lambda x: x.split(".",1)[0] ,arr_img))))
#     if (len(arr_img)==0):
#         return "1.jpg"
    
#     file_name= str(max(arr_num)+1) + ".jpg"
#     return file_name


    # #encode:
    # with open(join(*[out_viz_dir, f'{path.stem}.jpg']), "rb") as img_file:
    #     my_string = base64.b64encode(img_file.read())

def segment_image(im_file):

    img_dir= 'test_imgs'
    
    out_viz_dir='test_results'
    out_pred_dir='test_imgs_pred'
    # threshold =0.2

    # # #test
    # with open(os.path.join(img_dir, "negative/noncrack_noncrack_concrete_wall_74_28.jpg.jpg"), "rb") as img_file:
    #     my_string = base64.b64encode(img_file.read())
        
    # f = open("sample1.txt", "wb")
    # f.write(my_string)
    # f.close()

    if out_viz_dir != '':
        os.makedirs(out_viz_dir, exist_ok=True)
    if out_pred_dir != '':
        os.makedirs(out_pred_dir, exist_ok=True)

    
    # my_string=bytes(my_string,'utf-8') #convert input string to byte:
    # #decode to Pil Image:
    # im_bytes = base64.b64decode(my_string)   # im_bytes is a binary image
    # im_file = io.BytesIO(im_bytes)  # convert image to file-like object
    # img_0 = Image.open(im_file)

    # filename= find_name()  #file name to save result
    img_0=im_file
    # img_0.save(os.path.join(img_dir,filename))   ## save input image
    img_0 = np.asarray(img_0)
    


    if len(img_0.shape) != 3:
        print('incorrect image shape : must have 3 channels')
    img_0 = img_0[:,:,:3]
    img_height, img_width, img_channels = img_0.shape
    

    prob_map_full = evaluate_img(model, img_0)
    
    
    # ########################################## test evaluate_img(model, img_0) take so much times
    # input_width, input_height = input_size[0], input_size[1]
    # img_height, img_width, img_channels = img_0.shape
    # img_1 = cv2.resize(img_0, (input_width, input_height), cv2.INTER_AREA)
    # channel_means = [0.485, 0.456, 0.406]
    # channel_stds  = [0.229, 0.224, 0.225]
    # train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])

    # X = train_tfms(Image.fromarray(img_1))
    # # X = Variable(X.unsqueeze(0)).cuda()  # [N, 1, H, W]
    # X = Variable(X.unsqueeze(0)).cpu()

    # mask = model(X)
    
    
    # ###########model(X) so long time............. ===========> test model(X)

    # mask = torch.sigmoid(mask[0, 0]).data.cpu().numpy()
    # mask = cv2.resize(mask, (img_width, img_height), cv2.INTER_AREA)

    # ###########################


    ##save image mask:
    # plt.imsave(os.path.join(out_pred_dir,filename), prob_map_full, cmap='gray')

    ########### test
    plt.clf()
    plt.imshow(img_0)
    plt.imshow(prob_map_full, alpha=0.4)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.gcf().set_size_inches(img_width/100, img_height/100)    
    # plt.savefig(os.path.join(out_viz_dir,filename),dpi=100) 
    plt.savefig(os.path.join(out_viz_dir,"out.jpg"),dpi=100) 

    #encode:
    # with open(os.path.join(out_viz_dir,filename), "rb") as img_file_t:
    with open(os.path.join(out_viz_dir,"out.jpg"), "rb") as img_file_t:
        output = base64.b64encode(img_file_t.read())
    # print(output.decode("utf-8"))
    ##############
    return output.decode("utf-8")



def saveResult(result):
    url = "https://ap-southeast-1.aws.data.mongodb-api.com/app/data-wlatu/endpoint/data/v1/action/insertOne"
    payload = json.dumps({
        "collection": "fault_detection",
        "database": "thesis",
        "dataSource": "Cluster0",
        "document": {'building': str(result.get('building',"")),'date':result['date'], 'original_image': result['original_image'], 'prediction': result['prediction'],'type':result['type'], 'segment_image':result.get('segment_image',"")}
    })
    headers = {
    'Content-Type': 'application/json',
    'Access-Control-Request-Headers': '*',
    'api-key': 'LFyT8MWcEraGxtCsMJpceBO8q72WLX8mInon25j6kbVCgv2j5vSwVYzNVzdxFsqh', 
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)

# with open(os.path.join(img_dir, "crack/CFD_002.jpg"), "rb") as img_file:
#     my_string = base64.b64encode(img_file.read())

def predictImage(img):
    my_string=bytes(img,'utf-8') #convert input string to byte:
    #decode to Pil Image:
    im_bytes = base64.b64decode(my_string)   # im_bytes is a binary image
    im_file = io.BytesIO(im_bytes)  # convert image to file-like object
    im_file= Image.open(im_file)
    
    anomaly =  detect_crack(im_file)
    if (anomaly):  
        (pred,type)  = TypePrediction(im_file)
        if (type == 'Normal'):
            segment_img=""
        else:            
            segment_img= segment_image(im_file)    #edit

        # # print(pred)
        # if(pred>0.5):
        #     type="Crack"
        # else:
        #     type="Normal"
        result= {
            'date': str(datetime.datetime.now(pytz.timezone('Asia/Ho_Chi_Minh') )),
            'original_image':img,
            'prediction': float(pred),
            'type': type,
            'segment_image': segment_img     #edit
            # 'segment_image': 'segment_img'
        }
        return (result,1)
    else:
        return ({},0)

# model = load_model("./Models/VGG16_batchsize=32_epo=10.h5")
# def getPrediction(img_path):
#     urllib.request.urlretrieve(img_path, "PredictImage")
    
#     SIZE=120
#     a=[]
#     img = np.asarray(Image.open("PredictImage").convert('L').resize((SIZE,SIZE)))
#     a.append(img)
#     a=np.array(a)
#     a=a.reshape(-1, SIZE,SIZE, 1)
#     a=a/255    
    
#     pred = model.predict(a)[0][0] #Predict    
#     return pred





# test_prediction =getPrediction('https://phuongnamcons.vn/wp-content/uploads/2021/06/vet-nut-tuong-nho.jpg')
# print(test_prediction)

