import socket
import detect

service = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

service.bind(("0.0.0.0",9999))

service.listen()

c,address = service.accept()

while True:
    data = c.recv(10000)
    img = data.decode()
    result_img = detect.main(img)
    send_data = result_img.encode()
    c.send(send_data)
