from super_gradients.training import Trainer
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import(
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val
)
from super_gradients.training import models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import(
    DetectionMetrics_050,
    DetectionMetrics_050_095
)

#download
import os.path
import requests
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from tqdm import tqdm
def download_file(url,save_name):
    if not os.path.exists(save_name):
        print(f"Downloading file")
        file=requests.get(url,stream=True)
        total_size=int(file.headers.get("content-length",0))
        block_size=1024
        progress_bar=tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True
        )
        with open(os.path.join(save_name),'wb') as f:
            for data in file.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()
    else:
        print('File already present')

download_file('https://www.dropbox.com/s/xc2890eh8ujy3cu/hituav-a-highaltitude-infrared-thermal-dataset.zip?dl=1',
              '0329data.zip')

#unzip, 모두 압축해제하는 과정
import zipfile
def unzip(zip_file=None):
    try:
        with zipfile.ZipFile(zip_file) as z:
            z.extractall("./")
            print("Exctracted all")
    except:
        print("Invalid file")
unzip('0329data.zip')

#Dataset set up
ROOT_DIR='hit-uav'
train_imgs_dir='images/train'
train_labels_dir='labels/train'
val_imgs_dir='images/val'
val_labels_dir='labels/val'
test_imgs_dir='images/test'
test_labels_dir='labels/test'
classes=['Person','Car','Bicycle','OtherVechicle','DontCare']

dataset_params={
    'data_dir':ROOT_DIR,
    'train_images_dir':train_imgs_dir,
    'train_labels_dir':train_labels_dir,
    'val_images_dir':val_imgs_dir,
    'val_labels_dir':val_labels_dir,
    'test_images_dir':test_imgs_dir,
    'test_labels_dir':test_labels_dir,
    'classes':classes
}
#모든 트레이닝에 적용하는 parameters
EPOCHS=5
BATCH_SIZE=16
WORKERS=8

#gound truth image들을 시각화하는 함수
import cv2
#   색 뽑기
import numpy as np
colors=np.random.uniform(0, 255, size=(len(classes),3))
#   yolo format의 바운딩 박스를 xmin, ymin, xmax, ymax로 변환
def yolo2bbox(bboxes):
    xmin, ymin=bboxes[0]-bboxes[2]/2,bboxes[1]-bboxes[3]/2
    xmax, ymax=bboxes[0]+bboxes[2]/2,bboxes[1]+bboxes[3]/2
    return xmin,ymin,xmax,ymax

def plot_box(image,bboxes,labels):
    height, width = image.shape
    lw=max(round(sum(image.shape)/2*0.003),2) #선 굵기? 상자 넓이?
    tf=max(lw-1,1) #폰트 얇기
    for box_num, box in enumerate(bboxes):
        x1,y1,x2,y2=yolo2bbox(box)
        #coordinate들 역정규화
        xmin=int(x1*width)
        ymin=int(y1*height)
        xmax=int(x2*width)
        ymax=int(y2*height)
        p1,p2=(int(xmin),int(ymin)),(int(xmax),int(ymax))
        class_name=classes[int(labels[box_num])]
        color=colors[classes.index(class_name)]

        #좌측상단점과 우측하단점으로 이미지에 바운딩 박스 만들기
        cv2.rectangle(
            image,
            p1,p2,
            color=color,
            thickness=lw,
            lineType=cv2.LINE_AA
        )
        #text가 들어갈 rectangle 사이즈 구하기
        w,h=cv2.getTextSize(
            class_name,
            0,
            fontScale=lw/3,
            thickness=tf
        )[0]
        outside=p1[1]-h>=3
        p2=p1[0]+w,p1[1]-h-3 if outside else p1[1]+h+3
        #text가 들어갈 rectangle
        cv2.rectangle(
            image,
            p1,p2,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA
        )
        cv2.putText(
            image,
            class_name,
            (p1[0],p1[1]-5 if outside else p1[1]+h+2),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=lw/3.5,
            color=(255,255,255),
            thickness=tf,
            lineType=cv2.LINE_AA
        )
    return image

#이미지를 바운딩박스로 구성하기
import glob
import random
from matplotlib import pyplot as plt
def plot(image_path,label_path,num_samples):
    all_training_images=glob.glob(image_path+'/*')
    all_training_labels=glob.glob(label_path+'/*')
    all_training_images.sort()
    all_training_labels.sort()

    temp=list(zip(all_training_images,all_training_labels))
    random.shuffle(temp)
    all_training_images, all_training_labels=zip(*temp)
    all_training_images, all_training_labels=list(all_training_images),list(all_training_labels)
    num_images=len(all_training_images)
    if num_samples==-1:
        num_samples=num_images

    plt.figure(figsize=(15,12))
    for i in range(num_images):
        image_name=all_training_images[i].split(os.path.sep)[-1]
        image=cv2.imread(all_training_images[i])
        with open(all_training_images[i],'r') as f:
            bboxes=[]
            labels=[]
            label_lines=f.readlines()
            for label_line in label_lines:
                label, x_c, y_c, w, h=label_line.split(' ')
                x_c=float(x_c)
                y_c=float(y_c)
                w=float(w)
                h=float(h)
                bboxes.append([x_c,y_c,w,h])
                labels.append(label)
        result_image=plot_box(image,bboxes,labels)
        plt.subplot(2,2,i+1) #이미지의 2x2 그리드 시각화
        plt.imshow(image[:,:,::-1]) #이게 무슨 뜻이지?
        plt.axis('off')
    plt.tight_layout()
    plt.show()

#몇 개의 training 이미지들을 시각화
# plot(image_path=os.path.join(ROOT_DIR,train_imgs_dir),
#      label_path=os.path.join(ROOT_DIR,train_labels_dir),
#      num_samples=4)

#data preparation 데이터 준비
train_data=coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['train_images_dir'],
        'labels_dir': dataset_params['train_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':BATCH_SIZE,
        'num_workers':WORKERS
    }
)
val_data=coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['val_images_dir'],
        'labels_dir': dataset_params['val_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':BATCH_SIZE,
        'num_workers':WORKERS
    }
)

# transforms and augmentations??
# train_data.dataset.transforms
# train_data.dataset.plot(plot_transformed_data=True)

# training parameter 정의
train_params={
    'silent_mode':False,
    'average_best_models':True,
    'warmup_mode':'linear_epoch_step',
    'warmup_initial_lr': 1e-6,
    'lr_warmup_epochs':3,
    'initial_lr':5e-4,
    'lr_mode':"cosine",
    'cosine_final_lr_ratio':0.1,
    'optimizer':'Adam',
    'optimizer_params':{'weight_decay':0.0001},
    'zero_weight_decay_on_bias_and_bn':True,
    'ema':True,
    'ema_params':{'decay':0.9,'decay_type':'threshold'},
    'max_epochs':EPOCHS,
    'mixed_precision':True,
    'loss':PPYoloELoss(
        use_static_assigner=False,
        num_classes=len(dataset_params['classes']),
        reg_max=16
    ),
    'valid_metrics_list':[
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        ),
        DetectionMetrics_050_095(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    ],
    'metric_to_watch':'mAP@0.50:0.95'
}

#training을 위한 모델
models_to_train=[
    'yolo_nas_s',
    # 'yolo_nas_m',
    # 'yolo_nas_l'
]
CHECKPOINT_DIR='checkpoints'

#모델 training
def train():
    for model_to_train in models_to_train:
        trainer=Trainer(
        experiment_name=model_to_train,
        ckpt_root_dir=CHECKPOINT_DIR
        )
        model=models.get(
            model_to_train,
            num_classes=len(dataset_params['classes']),
            pretrained_weights="coco"
        )
        trainer.train(
            model=model,
            training_params=train_params,
            train_loader=train_data,
            valid_loader=val_data
        )

# if __name__ == '__main__':
#     train()

#result 확인 준비
import torch
os.makedirs('inference_result/images/',exist_ok=True)
device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model=models.get(
    model_name='yolo_nas_s',
    checkpoint_path='checkpoints/yolo_nas_s/RUN_20240331_071829_558616/ckpt_best.pth',
    num_classes=5
).to(device)
#test image에 적용
ROOT_TEST='hit-uav/images/test/'
all_images=os.listdir(ROOT_TEST)
for image in tqdm(all_images,total=len(all_images)):
    image_path=os.path.join(ROOT_TEST,image)
    out=model.predict(image_path)
    out.save('inference_result/images','jpg')
    os.rename(
        'inference_result/images/pred_0.jpg',
            os.path.join('inference_result/images/',image)
        )