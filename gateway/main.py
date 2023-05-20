import sys
from Adafruit_IO import MQTTClient
import time
import random
from simple_ai import *
from uart import *


AIO_FEED_ID = ["nutnhan1","nutnhan2"]
AIO_USERNAME = "van00972751"
AIO_KEY = "aio_rJsh148N4JVHIgIMj0Ddf1iuxdQ2"

def connected(client):
    print("Ket noi thanh cong ...")
    for topic in AIO_FEED_ID:
        client.subscribe(topic)

def subscribe(client , userdata , mid , granted_qos):
    print("Subscribe thanh cong ...")

def disconnected(client):
    print("Ngat ket noi ...")
    sys.exit(1)

def message(client , feed_id , payload):
    #nhan du lieu tu nút nhấn của dashborad (nutnhan1 và nutnhan2)
    print("Nhan du lieu: " + payload + " , feed id:" + feed_id)
    if feed_id == "nutnhan1":
        if payload == "0":
            # lien ket dieu khien cho thiet bi phan cung
            writeData("1")
            print(1)
        else:
            writeData("2")
            print(2)
    if feed_id == "nutnhan2":
        if payload == "3":
            print("capture...")
            ai_result = image_detector()
            client.publish("ai", ai_result[0])
            client.publish("image",ai_result[1]) 
            # writeData("3")  
            time.sleep(1)
            print("successful ...")
            
        else:
            # writeData("4")
            print(4)
            
client = MQTTClient(AIO_USERNAME , AIO_KEY)
client.on_connect = connected
client.on_disconnect = disconnected
client.on_message = message
client.on_subscribe = subscribe
client.connect()
client.loop_background()

counter = 6
sensor_type = 0
counter_ai=5
ai_result= ""
previous_result=""

avg_temp= 27.55
std_temp= 10
previous_temp=30

avg_humid= 79.5
std_humid= 20
previous_humid=50

while True:

    # counter = counter -1
    # if counter <=0:
    #     counter = 10
    #     #TODO
    # lab3
    # print("Random data is publishing ...")
    # if sensor_type ==0:
    #     print("Temprature...")
    #     temp = random.randint(10, 20)
    #     client.publish("cambien1", temp)
    #     sensor_type = 1
    # elif sensor_type ==1:
    #     print("Humidity...")
    #     humi = random.randint(50,70)
    #     client.publish("cambien2", humi)
    #     sensor_type = 2
    # elif sensor_type == 2:
    #     print("Light...")
    #     light = random.randint(100,500)
    #     client.publish("cambien3", light)
    #     sensor_type = 0

    counter = counter -1
    if counter <=0:
        counter = 6

        ############ Temprature read ################
        print("Temprature...")
        temp = random.randint(27, 36)   ##edit:

        temp_ok = False
        if (avg_temp-3*std_temp<=temp<=avg_temp+3*std_temp):
            if (previous_temp-5 <= temp <= previous_temp+5 ):
                previous_temp=temp
                client.publish("cambien1", temp)
                temp_ok=True
                client.publish("error1", "No Warning..")
        if (temp_ok==False):
            client.publish("error1", "Temprature is out of range..")
        
        ############## Humidity read #################
        print("Humidity...")
        humi = random.randint(40,70)

        humi_ok = False
        if (avg_humid-3*std_humid<=humi<=avg_humid+3*std_humid):
             if (previous_humid-6 <= humi <= previous_humid+6 ):
                previous_humid=humi
                client.publish("cambien2", humi)
                humi_ok=True
                client.publish("error2", "No Warning..")
        if (humi_ok== False):
            client.publish("error2","Humid is out of range...")



    ####### AI for 5s
    # counter_ai= counter_ai-1
    # if counter_ai <=0:
    #     counter_ai=5
    #     previous_result=ai_result
    #     ai_result = image_detector()
    #     print("AI output: ", ai_result)
    #     if previous_result!=ai_result:
    #         client.publish("ai", ai_result[0])
    #         client.publish("image",ai_result[1])
    ##############
    


    

    readSerial(client)
    time.sleep(1)
