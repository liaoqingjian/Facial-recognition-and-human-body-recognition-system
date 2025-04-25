import face_recognition
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import numpy as np
import threading
import time
import sys
from torch.utils.data import DataLoader
from torchvision import datasets
from ultralytics import YOLO

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("人脸识别系统")
        self.root.geometry("1200x800")
        
        # 初始化变量
        self.camera_running = False
        self.cap = None
        
        # 控制人体检测的变量 - 确保在创建标签页前初始化
        self.enable_body_detection = tk.BooleanVar(value=True)
        
        # 初始化模型
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        try:
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to('cuda')
        except:
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to('cpu')
            
        # 初始化YOLOv8模型 (如果不存在模型文件，需要下载)
        self.yolo_model = None
        try:
            # 尝试加载YOLOv8模型
            # 创建models目录
            if not os.path.exists("./models"):
                os.makedirs("./models", exist_ok=True)
                
            yolo_model_path = './models/yolov8n.pt'
            if not os.path.exists(yolo_model_path):
                # 如果模型不存在，将使用在线版本
                self.yolo_model = YOLO('yolov8n.pt')
                # 保存模型到本地
                self.yolo_model.save(yolo_model_path)
            else:
                self.yolo_model = YOLO(yolo_model_path)
                
            print(f"YOLOv8模型加载成功，使用设备: {self.device}")
        except Exception as e:
            print(f"YOLOv8模型加载失败: {str(e)}")
            messagebox.showwarning("警告", "人体检测模型加载失败，将只使用人脸识别功能")
            
        # 创建主框架
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建标签页
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 创建三个标签页
        self.create_training_tab()
        self.create_image_recognition_tab()
        self.create_video_recognition_tab()
        
        # 创建font目录
        if not os.path.exists("./font"):
            os.makedirs("./font", exist_ok=True)
        
    def create_training_tab(self):
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="人脸入库")
        
        # 创建输入框和按钮
        ttk.Label(training_frame, text="姓名:").pack(pady=10)
        self.name_entry = ttk.Entry(training_frame)
        self.name_entry.pack(pady=5)
        
        ttk.Button(training_frame, text="选择图片", command=self.select_image).pack(pady=10)
        ttk.Button(training_frame, text="开始训练", command=self.train_model).pack(pady=10)
        
        # 显示选择的图片
        self.training_image_label = ttk.Label(training_frame)
        self.training_image_label.pack(pady=10)
        
    def create_image_recognition_tab(self):
        image_frame = ttk.Frame(self.notebook)
        self.notebook.add(image_frame, text="图片识别")
        
        # 创建左右两个框架
        left_frame = ttk.Frame(image_frame)
        right_frame = ttk.Frame(image_frame)
        left_frame.pack(side=tk.LEFT, padx=10, pady=10)
        right_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # 原始图片显示
        ttk.Label(left_frame, text="原始图片").pack()
        self.original_image_label = ttk.Label(left_frame)
        self.original_image_label.pack()
        
        # 识别结果图片显示
        ttk.Label(right_frame, text="识别结果").pack()
        self.result_image_label = ttk.Label(right_frame)
        self.result_image_label.pack()
        
        # 按钮
        ttk.Button(image_frame, text="选择图片", command=self.select_recognition_image).pack(pady=10)
        ttk.Button(image_frame, text="开始识别", command=self.recognize_image).pack(pady=10)
        
    def create_video_recognition_tab(self):
        video_frame = ttk.Frame(self.notebook)
        self.notebook.add(video_frame, text="实时识别")
        
        # 视频显示区域
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(pady=10)
        
        # 按钮框架
        button_frame = ttk.Frame(video_frame)
        button_frame.pack(pady=5)
        
        # 按钮
        self.start_button = ttk.Button(button_frame, text="打开摄像头", command=self.start_camera)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="关闭摄像头", command=self.stop_camera, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # 人体检测开关
        self.body_detection_check = ttk.Checkbutton(video_frame, 
                                                  text="启用人体检测", 
                                                  variable=self.enable_body_detection)
        self.body_detection_check.pack(pady=5)
        
    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            try:
                self.training_image_path = file_path
                # 使用PIL打开图片
                image = Image.open(file_path)
                image = image.resize((300, 300))
                photo = ImageTk.PhotoImage(image)
                self.training_image_label.configure(image=photo)
                self.training_image_label.image = photo
            except Exception as e:
                messagebox.showerror("错误", f"无法打开图片: {str(e)}")
            
    def train_model(self):
        if not hasattr(self, 'training_image_path'):
            messagebox.showerror("错误", "请先选择图片")
            return
            
        name = self.name_entry.get()
        if not name:
            messagebox.showerror("错误", "请输入姓名")
            return
            
        try:
            # 创建保存目录
            if not os.path.exists("database"):
                os.makedirs("database")
            if not os.path.exists("database/orgin"):
                os.makedirs("database/orgin")
                
            # 使用ASCII字符作为文件夹名
            safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
            if not safe_name:
                safe_name = "unknown"
                
            save_path = os.path.join("database", "orgin", safe_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                
            # 使用PIL读取和保存图片
            image = Image.open(self.training_image_path)
            image.save(os.path.join(save_path, "1.jpg"))
            
            # 训练模型
            # 加载数据库
            dataset = datasets.ImageFolder('./database/orgin')
            dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
            
            def collate_fn(x):
                return x[0]
                
            loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=0)
            
            # 处理图片并提取特征
            aligned = []
            names = []
            
            for x, y in loader:
                # 处理图像并检测人脸    
                # 转换PIL图像为numpy数组
                img_array = np.array(x)
                # 使用MTCNN检测人脸
                x_aligned, prob = self.mtcnn(img_array, return_prob=True)
                
                if x_aligned is not None:
                    # 正确处理标量值
                    if hasattr(prob, 'item'):  # 如果是tensor
                        prob_value = prob.item()
                    elif hasattr(prob, 'tolist'):  # 如果是numpy数组
                        prob_value = prob.tolist()
                        if isinstance(prob_value, list):
                            prob_value = prob_value[0] if prob_value else 0
                    else:  # 如果已经是标量
                        prob_value = float(prob)
                    
                    print(f'人脸检测概率: {prob_value:.8f}')
                    
                    # 确保x_aligned有正确的维度
                    if x_aligned.dim() == 4:  # 批处理维度
                        for i, face in enumerate(x_aligned):
                            aligned.append(face)
                            names.append(dataset.idx_to_class[y])
                    else:  # 单个图像
                        aligned.append(x_aligned)
                        names.append(dataset.idx_to_class[y])
            
            if not aligned:
                messagebox.showerror("错误", "未检测到人脸，请重新选择图片")
                return
                
            # 提取特征向量
            aligned = torch.stack(aligned).to(self.device)
            embeddings = self.resnet(aligned).detach().cpu()
            
            # 保存特征向量和名字
            torch.save(embeddings, './database/database.pt')
            torch.save(names, './database/names.pt')
            
            messagebox.showinfo("成功", "训练完成")
            
        except Exception as e:
            messagebox.showerror("错误", f"训练失败: {str(e)}")
            print(f"详细错误信息: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def select_recognition_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            try:
                self.recognition_image_path = file_path
                image = Image.open(file_path)
                image = image.resize((300, 300))
                photo = ImageTk.PhotoImage(image)
                self.original_image_label.configure(image=photo)
                self.original_image_label.image = photo
            except Exception as e:
                messagebox.showerror("错误", f"无法打开图片: {str(e)}")
            
    def recognize_image(self):
        if not hasattr(self, 'recognition_image_path'):
            messagebox.showerror("错误", "请先选择图片")
            return
            
        try:
            # 检查数据库文件是否存在
            if not os.path.exists("./database/names.pt") or not os.path.exists("./database/database.pt"):
                messagebox.showerror("错误", "人脸数据库不存在，请先添加人脸")
                return
                
            # 加载数据库
            names = torch.load("./database/names.pt")
            embeddings = torch.load("./database/database.pt").to(self.device)
            
            # 使用PIL读取图片
            img_pil = Image.open(self.recognition_image_path)
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
            # 检测人脸
            boxes, _ = self.mtcnn.detect(img)
            
            if boxes is not None:
                # 获取所有人脸的对齐图像
                faces = self.mtcnn(img)
                
                # 确保faces不为None并处理多个人脸
                if faces is not None:
                    # 如果返回单个人脸（不是批量），将其转换为列表
                    if not isinstance(faces, list) and faces.dim() == 3:
                        faces = [faces]
                        
                    # 遍历所有检测到的人脸
                    for i, box in enumerate(boxes):
                        if i < len(faces):  # 确保faces[i]存在
                            # 计算人脸嵌入向量
                            face_tensor = faces[i].unsqueeze(0).to(self.device)
                            face_embedding = self.resnet(face_tensor)
                            
                            # 计算与数据库中所有人脸的距离
                            distances = [(face_embedding - embeddings[j]).norm().item() for j in range(embeddings.size()[0])]
                            closest_idx = distances.index(min(distances))
                            
                            # 如果距离小于阈值，认为是已知人脸
                            if distances[closest_idx] < 1:
                                name = names[closest_idx]
                            else:
                                name = "未知人员"
                                
                            # 绘制边界框和名字
                            x1, y1, x2, y2 = [int(float(coord)) for coord in box]
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 10)
                            # 使用draw_chinese_text方法替代cv2.putText以显示中文
                            img = self.draw_chinese_text(img, name, (x1, y1-30))
                    
            # 显示结果图片
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img)
            image = image.resize((300, 300))
            photo = ImageTk.PhotoImage(image)
            self.result_image_label.configure(image=photo)
            self.result_image_label.image = photo
            
        except Exception as e:
            messagebox.showerror("错误", f"识别失败: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def start_camera(self):
        self.camera_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        self.cap = cv2.VideoCapture(0)
        self.update_video()
        
    def stop_camera(self):
        self.camera_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        if self.cap is not None:
            self.cap.release()
            
    def draw_chinese_text(self, img, text, position, text_color=(0, 0, 255), text_size=60):
        # 将OpenCV图像转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 加载中文字体
        try:
            font = ImageFont.truetype("./font/simhei.ttf", text_size)
        except:
            # 如果找不到字体文件，使用系统默认字体
            font = ImageFont.load_default()
            
        # 绘制文字
        draw.text(position, text, text_color, font=font)
        
        # 转换回OpenCV格式
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def detect_human_bodies(self, frame):
        """使用YOLOv8检测人体"""
        if self.yolo_model is None:
            return frame, []
            
        # 只检测人类（类别0）
        results = self.yolo_model(frame, classes=0)
        
        # 获取检测框
        boxes = []
        for result in results:
            for box in result.boxes:
                # 获取类别
                cls = int(box.cls.item())
                # 获取置信度
                conf = box.conf.item()
                # 只处理人类类别且置信度大于0.5的检测结果
                if cls == 0 and conf > 0.5:
                    x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
                    boxes.append((x1, y1, x2, y2, conf))
                    # 绘制人体框 - 使用绿色
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # 添加标签
                    label = f"people: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame, boxes

    def update_video(self):
        if self.camera_running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # 先进行人体检测（如果启用）
                if self.enable_body_detection.get() and self.yolo_model is not None:
                    frame, body_boxes = self.detect_human_bodies(frame)
                
                # 检查数据库文件是否存在
                if os.path.exists("./database/names.pt") and os.path.exists("./database/database.pt"):
                    try:
                        # 加载数据库
                        names = torch.load("./database/names.pt")
                        embeddings = torch.load("./database/database.pt").to(self.device)
                        
                        # 检测人脸
                        boxes, _ = self.mtcnn.detect(frame)
                        
                        if boxes is not None:
                            # 获取所有人脸的对齐图像
                            faces = self.mtcnn(frame)
                            
                            # 确保faces不为None并处理多个人脸
                            if faces is not None:
                                # 如果返回单个人脸（不是批量），将其转换为列表
                                if not isinstance(faces, list) and faces.dim() == 3:
                                    faces = [faces]
                                    
                                # 遍历所有检测到的人脸
                                for i, box in enumerate(boxes):
                                    if i < len(faces):  # 确保faces[i]存在
                                        # 计算人脸嵌入向量
                                        face_tensor = faces[i].unsqueeze(0).to(self.device)
                                        face_embedding = self.resnet(face_tensor)
                                        
                                        # 计算与数据库中所有人脸的距离
                                        distances = [(face_embedding - embeddings[j]).norm().item() for j in range(embeddings.size()[0])]
                                        closest_idx = distances.index(min(distances))
                                        
                                        # 如果距离小于阈值，认为是已知人脸
                                        if distances[closest_idx] < 1:
                                            name = names[closest_idx]
                                        else:
                                            name = "未知人员"
                                            
                                        # 绘制边界框和名字
                                        x1, y1, x2, y2 = [int(float(coord)) for coord in box]
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                        # 使用新的绘制中文文字的方法
                                        frame = self.draw_chinese_text(frame, name, (x1, y1-30))

                    except Exception as e:
                        print(f"识别错误: {str(e)}")
                        import traceback
                        traceback.print_exc()
                    
                # 添加状态信息
                status_text = "状态: "
                if self.enable_body_detection.get() and self.yolo_model is not None:
                    status_text += "人脸+人体检测"
                else:
                    status_text += "仅人脸检测"
                frame = self.draw_chinese_text(frame, status_text, (10, 30), text_color=(255, 0, 0))
                
                # 显示视频帧
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                image = image.resize((640, 480))
                photo = ImageTk.PhotoImage(image)
                self.video_label.configure(image=photo)
                self.video_label.image = photo
                
            self.root.after(10, self.update_video)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()