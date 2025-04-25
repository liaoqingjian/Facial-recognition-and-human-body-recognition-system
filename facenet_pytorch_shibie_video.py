#视频识别
import face_recognition
import cv2
from PIL import ImageFont, ImageDraw, Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import os
import numpy
import time
t1=time.time()

workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# mtcnn网络负责检测人脸
mtcnn = MTCNN(keep_all=True, device=device)
try:
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to('cuda')
except:
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to('cpu')

names = torch.load("./database/names.pt")
try:
    embeddings = torch.load("./database/database.pt").to('cuda')
except:
    embeddings = torch.load("./database/database.pt").to('cpu')


def cv2ImgAddText(img, text, a, b, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, numpy.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype("./font/msyhbd.ttc", textSize, encoding="utf-8")
    draw.text((a, b), text, textColor, font=fontStyle)
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
def write_chinese(img,  content,position,font_type, font_size,color):
    # 图像从OpenCV格式转换成PIL格式
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 字体  字体*.ttc的存放路径一般是： /usr/share/fonts/opentype/noto/ 查找指令locate *.ttc
    font = ImageFont.truetype(font_type, font_size)
    # 字体颜色
    # 文字输出位置
    # 输出内容
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, content, font=font, fill=color)
    # 转换回OpenCV格式
    img_OpenCV = cv2.cvtColor(numpy.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img_OpenCV



def detect_frame(img):
    faces = mtcnn(img)
    boxes, _ = mtcnn.detect(img)  # 检测出人脸框 返回的是位置
    if boxes is not None:
        for i, box in enumerate(boxes):
            try:
                face_embedding = resnet(faces[i].unsqueeze(0).to('cuda'))
            except:
                face_embedding = resnet(faces[i].unsqueeze(0).to('cpu'))
            # 计算距离
            probs = [(face_embedding - embeddings[i]).norm().item() for i in range(embeddings.size()[0])]
            # 我们可以认为距离最近的那个就是最有可能的人，但也有可能出问题，数据库中可以存放一个人的多视角多姿态数据，对比的时候可以采用其他方法，如投票机制决定最后的识别人脸
            index = probs.index(min(probs))  # 对应的索引就是判断的人脸
            if probs[index] < 1:
                name = names[index]  # 对应的人脸
                # print(name)
            else:
                name = "未知人员"
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            img = cv2ImgAddText(img, name, box[0], box[1] - 100, (255, 0, 0), 80)
            ket=(box[0], box[1])
            print((box[0], box[1]))
    return name,ket


if __name__ == '__main__':
    cap=cv2.VideoCapture(0)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            draw,ket = detect_frame(frame)
            # frame = cv2.putText(frame, (draw), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame = write_chinese(frame, (draw),ket,'font/simhei.ttf', 20, (255, 0, 255), )
            cv2.imshow('Video', frame)
            cv2.waitKey(1)
            if draw!="未知人员":
                print(draw)
        else:
            break