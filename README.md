# 2. 读取图像
frame = cv2.imread("./images/car1.jpg")
# 视频的宽度和高度，即帧尺寸
(W, H) = (None, None)
if W is None or H is None:
    (H, W) = frame.shape[:2]
 
# 根据输入图像构造blob,利用OPenCV进行深度网路的计算时，一般将图像转换为blob形式，对图片进行预处理，包括缩放，减均值，通道交换等
# 还可以设置尺寸，一般设置为在进行网络训练时的图像的大小
blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
