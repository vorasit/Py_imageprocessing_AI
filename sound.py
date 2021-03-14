
import pygame #นำไลบราลี่ชื่อ pygame มาใช้งาน
pygame.mixer.init() #เรียกใช้ฟังก์ชั่นหลัก
pygame.mixer.music.load("dangeralar.mp3") #โหลไฟลเสียงมาเตรียมใช้งาน
pygame.mixer.music.play() #เรียกใช้งานไฟล์เสียง
while pygame.mixer.music.get_busy() == True: #ขณะที่มีการทำงานแสดงเสียงจริงให้ทำตามเงื่อนไข
    continue #เล่นต่อไป

