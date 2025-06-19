# from ultralytics import YOLO

# model = YOLO("yolo11l-seg.pt")

# result = model("/home/shadoow/Downloads/Codes/ARCH Internship/archive/Training/glioma/Tr-gl_0010.jpg")
# result[0].show()

yolo task=detect model=train data=data.yaml model=yolo11n.pt epochs=100 imgsz=640
