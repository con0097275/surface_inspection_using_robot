import serial.tools.list_ports

def getPort():
    ports = serial.tools.list_ports.comports()
    N = len(ports)
    commPort = "None"
    for i in range(0, N):
        port = ports[i]
        strPort = str(port)
        if "USB Serial Device" in strPort:
            splitPort = strPort.split(" ")
            commPort = (splitPort[0])
    # return commPort
    return "COM4"

if getPort()!="None":
    ser = serial.Serial( port=getPort(), baudrate=115200)
    print(ser)

# def getPort():
#     ports = serial.tools.list_ports.comports()
#     N = len(ports)
#     commPort = "None"
#     for i in range(0, N):
#         port = ports[i]
#         strPort = str(port)
#         if "USB Serial Device" in strPort:
#             splitPort = strPort.split(" ")
#             commPort = (splitPort[0])
#     return commPort


# isMicrobitConnected = False
# if getPort() != "None":
#     ser = serial.Serial(port=getPort(), baudrate=115200)
#     isMicrobitConnected = True


# def processData(client, data):
#     data = data.replace("!", "")
#     data = data.replace("#", "")
#     splitData = data.split(":")
#     print(splitData)
#     if splitData[1] == "T":
#         client.publish("cambien1", splitData[2])
#     if splitData[1] == "H":
#         client.publish("cambien2", splitData[2])
avg_temp= 27.55
std_temp= 10
previous_temp=30

avg_humid= 79.5
std_humid= 20
previous_humid=50
def processData(client, data):
    data = data.replace("!", "")
    data = data.replace("#", "")
    splitData = data.split(":")
    print(splitData)
    if splitData[1] == "T":
        temp = float(splitData[2])
        
        if (avg_temp - 3*std_temp <= temp <= avg_temp + 3*std_temp):
            if (previous_temp - 5 <= temp <= previous_temp + 5 ):
                previous_temp = temp
                client.publish("cambien1", temp)
                temp_ok = True
                client.publish("error1", "No Warning..")
        if (temp_ok==False):
            client.publish("error1", "Temprature is out of range..")
        client.publish("cambien1", temp)
    elif splitData[1] == "H":
        print("Humidity...")
        humi = float(splitData[2])

        humi_ok = False
        if (avg_humid-3*std_humid<=humi<=avg_humid+3*std_humid):
             if (previous_humid-6 <= humi <= previous_humid+6 ):
                previous_humid=humi
                client.publish("cambien2", humi)
                humi_ok=True
                client.publish("error2", "No Warning..")
        if (humi_ok== False):
            client.publish("error2","Humid is out of range...")
        client.publish("cambien2", humi)


mess = ""
def readSerial(client):
    bytesToRead = ser.inWaiting()
    if (bytesToRead > 0):
        global mess
        mess = mess + ser.read(bytesToRead).decode("UTF-8")
        while ("#" in mess) and ("!" in mess):
            start = mess.find("!")
            end = mess.find("#")
            processData(client, mess[start:end + 1])
            if (end == len(mess)):
                mess = ""
            else:
                mess = mess[end+1:]




def writeData(data):
    ser.write(str(data).encode())
