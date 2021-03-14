import os #เรียกฟังชั่นเกี่ยวกับไฟล์
import argparse #การรับค่าเข้าไปยังโปรแกรมเมื่อรันผ่าน cmd
import cv2
import numpy as np #เกี่ยวกับคณิตศาสตร์
import sys #ป้อนค่าเพิ่มเติมเข้าไปให้โปรแกรมเวลารัน
import time
from threading import Thread #ทำงานพร้อมกันหลาย process
import importlib.util #การนำเข้าโมดูลที่กำหนดเส้นทางแบบเต็ม

class VideoStream: #คลาส VideoStream
    """กล้องตรวจจับวัตถุที่ผ่านเข้ามายัง webcame"""
    
    def __init__(self,resolution=(640,480),framerate=30):
        # PiCamera และ ภาพจาก camera 
        self.stream = cv2.VideoCapture(0) #ใช้กล้องหลัก
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) #ถ่ายภาพจากกล้องทุกๆ frame
        ret = self.stream.set(3,resolution[0]) #
        ret = self.stream.set(4,resolution[1]) #
            
        # อ่าน frame แรก จาก stream
        (self.grabbed, self.frame) = self.stream.read()

    # ตัวแปรที่ใช้หยุด
        self.stopped = False

    def start(self):
    # เริ่มอ่านเฟรมจาก stream วิดีโอ
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # วนไปเรื่อย ๆ จนกว่าจะหยุด
        while True:
            # ถ้ากล้องหยุด
            if self.stopped:
                # ปิดกล้องและคืน resources
                self.stream.release()
                return

            # จับเฟรมถัดไป
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # คืนค่าเฟรมล่าสุด
        return self.frame

    def stop(self):
    # หยุดทำงาน
        self.stopped = True

# กำหนด input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', default='test')# ที่อยู่โฟลเดอร
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')# file model
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')# file label
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')# file model edge tpu

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x') # display resolution Split แบบกำหนดจำนวนผลลัพธ์
imW, imH = int(resW), int(resH) #argument ขาดภาพ
use_TPU = args.edgetpu

# นำไลบราลี่ TensorFlow มาใช้งาน
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from normal tensorflow
# ถ้าใช้ Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# ถ้าใช้ Edge TPU จะกำหนดชื่อไฟล์ Edge TPU model
if use_TPU:
    # หากผู้ใช้ระบุชื่อไฟล์. tflite ให้ใช้ชื่อนั้นหรือใช้ค่าเริ่มต้น 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# ไปยังไดเร็กทอรีการทำงานปัจจุบัน
CWD_PATH = os.getcwd()

# path ไปยังไฟล์. tflite ซึ่งมีโมเดลที่ใช้สำหรับการตรวจจับอ็อบเจ็กต์
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# path ไปยังไฟล์ label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()] #วนอ่านแต่ละบรรทัด

# แก้ไข label map ถ้ามีการใช้ COCO "starter model"
# อันแรก ลบ label ที่ชื่อว่า '???'
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# ถ้าใช้ Edge TPU, ให้เรียกใช้ load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')]) #ใช้ edge tpu 1.0
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT) # ใช้โมเดลธรรมดา

interpreter.allocate_tensors() #โหลด model TFLite และจัดสรรเทนเซอร์

# Get model details
input_details = interpreter.get_input_details() # รับค่า weight
output_details = interpreter.get_output_details() #เก็บค่าเอาพุต weight
height = input_details[0]['shape'][1] #
width = input_details[0]['shape'][2] #
#คำนวณเป็น float ที่ได้จากการคำนวณค่า weight
floating_model = (input_details[0]['dtype'] == np.float32)
# ค่าเฉลี่ย
input_mean = 127.5
input_std = 127.5

# เริ่มต้นการคำนวณอัตราเฟรม
frame_rate_calc = 1
freq = cv2.getTickFrequency() #การวัดเวลาการดำเนินการ

# กำหนดตัวแปร videostream = ขนาดภาพ และ framerate ผ่านคลาส VideoStream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1) #ค้าง 1 วินาที

#วนทำงานใน frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:
    try:
        # เริ่ม (เริ่มคำนวณ frame rate)
        t1 = cv2.getTickCount()

        # รับภาพผ่านตัวแปร จากคลาส videostream.read()
        frame1 = videostream.read()

        # รับเฟรมและปรับขนาดเป็นรูปร่างที่คาดหวัง [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (หากแบบจำลองไม่ได้ตามขนาด)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # ทำการตรวจจับโดยเรียกใช้โมเดลที่มีรูปภาพเป็นอินพุต
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # กรอบล้อม objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # ชื่อวัตถุ
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # ความมั่นใจ
        #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

        # วนรอบการตรวจจับทั้งหมดและวาดกล่องตรวจจับหากมีความเชื่อมั่น
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # รับพิกัดกล่องขอบเขตและวาดกล่อง
                # สามารถส่งคืนพิกัดที่อยู่นอกขนาดภาพโดยต้องบังคับให้อยู่ภายในภาพโดยใช้ max () และ min ()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                #กำหนดขนาดสี่เหลี่ยม
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # วาด label
                object_name = labels[int(classes[i])] # ค้นหาชื่อวัตถุจากอาร์เรย์ "label" โดยใช้ index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # ตัวอย่าง: 'pistol: 99%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # ขนาด font 
                label_ymin = max(ymin, labelSize[1] + 10) 
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                sc='%d%%'%(int(scores[i]*100)) #ค่าน้ำหนัก
                if object_name == "pistol" and sc >= '90':
                    print("detection pistol")
                    cv2.imwrite("images/get.jpg",frame) # บันทึกภาพ
                    crop_img = frame[ymin:ymin+(ymax-ymin)-10, xmin:xmin+(xmax-xmin)-10] #ตัดภาพตามขนาด
                    #cv2.imshow("cropped", crop_img)
                    cv2.imwrite("images/gun.jpg",crop_img)
                    img=cv2.imread("images/gun.jpg",cv2.IMREAD_UNCHANGED)
                    width=300
                    height=300
                    dim=(width,height)
                    resized=cv2.resize(img,dim,interpolation=cv2.INTER_AREA) #แก้ขนาดภาพ
                    cv2.imwrite("images/gun2.jpg",resized) # บันทึกภาพ
                    os.system('python3 TFLite_detection_image.py') #เรียกใช้ไฟลการ classification
                    img2=cv2.imread("images/check.jpg",cv2.IMREAD_UNCHANGED) # อ่านภาพ
                    cv2.imshow("check", img2) #แสดงภาพ
                    #os.system('python3 classify_image.py')
                    #cv2.imwrite("images/gun2.jpg",frame)
                    os.system('python3 line2.py')#เรียกใช้ไฟล line
                    os.system('python3 sound.py')#เรียกใช้ไฟลเสียง

                    
        # วาดเลขที่ขึ้นบน fps
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA) 

        # show
        cv2.imshow('Object detector', frame)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    except:
        pass #cross
    

        
# Clean up
cv2.destroyAllWindows() # ทำลาย process
videostream.stop() # หยุด
