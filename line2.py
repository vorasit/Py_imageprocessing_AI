import requests #เรียกใช้ไลบราลี่เพื่อติดต่อ line
from datetime import datetime #เรียกใช้ไลบราลี่ datetime
from datetime import date #เรียกใช้ไลบราลี่ date
now = datetime.now() #เก็บค่าวันที่และเวลาปัจจุบันไว้ที่ตัวแปร now
date_time = now.strftime("%d/%m/%Y") #รูปแบบวัน เดือน ปี
hour = now.strftime("%H:%M:%S") #รูปแบบ ชั่วโมง นาที วินาที
text = "ตรวจพบอาวุธปืน" +' วันที่ :'+ date_time + '\n'+ ' เวลา :'+ hour #เก็บค่า
exp="%d-%m-%Y-%H:%M:%S" #เรียงลำดับรูปแบบ
LINE_ACCESS_TOKEN="1E8y9ZZvvjy9136Ev8JqqdijNXCQ8zM65Me9bINnvrS" #รหัส token
url = "https://notify-api.line.me/api/notify" #เว็บบริการ line notify
file = {'imageFile':open('image/gun.png','rb')} #เก็บที่อยู่ภาพ
data = ({
        'message': text
    }) #เก็บข้อมูลแบบ nosql
LINE_HEADERS = {"Authorization":"Bearer "+LINE_ACCESS_TOKEN} #เก็บค่ารหัส token
session = requests.Session() #ทำการรองขอเพื่อใช้บริการ
r=session.post(url, headers=LINE_HEADERS, files=file, data=data) #ทำการส่งช้อมูล
print(r.text) #แสดงข้อความจากผู้ให้บริการ