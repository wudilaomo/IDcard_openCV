import cv2
import numpy as np


def ShowImage(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)  # 等待时间，0表示任意键退出
    cv2.destroyAllWindows()


def sort_contours(cnts, method="left-to-right"):
    # reverse = False 表示升序，若不指定reverse则默认升序
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True  # reverse = True 表示降序

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # 用一个最小的矩形，把找到的形状包起来，用x,y,h,w表示
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

    # zip函数用于打包可迭代数据，得到最终输出的cnts和boundingBoxes
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    return cnts, boundingBoxes


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]  # 获取图像的高度和宽度
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)  # 使用cv库的resize函数
    return resized


template = cv2.imread('./number.png')
ShowImage('template', template)

# 将图像转化为灰度图
image_Gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
ShowImage('gray', image_Gray)

# 转换为二值化图像,[1]表示返回二值化图像，[0]表示返回阈值177
image_Binary = cv2.threshold(image_Gray, 177, 255, cv2.THRESH_BINARY_INV)[1]
ShowImage('binary', image_Binary)

# 提取轮廓
refcnts, his = cv2.findContours(image_Binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(template, refcnts, -1, (0, 0, 255), 2)
ShowImage('contour', template)

refcnts = sort_contours(refcnts, method="left-to-right")[0]
digits = {}

# 遍历每个轮廓
for (i, c) in enumerate(refcnts):  # enumerate函数用于遍历序列中的元素以及它们的下标
    (x, y, w, h) = cv2.boundingRect(c)
    roi = image_Binary[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))

    digits[i] = roi

# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 读取图像，进行预处理
image = cv2.imread("./card.jpg")
ShowImage('card', image)

image = resize(image, width=300)
# 将图像转化为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ShowImage('card_gray', gray)

# 通过顶帽操作，突出更明亮的区域
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
ShowImage('tophat_card', tophat)

gradx = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
grady = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
gradx = np.absolute(gradx)
minVal = np.min(gradx)
maxVal = np.max(gradx)
# (minVal, maxVal) = (np.min(gradx), np.max(gradx))
# 保证值的范围在0-255之间
gradx = (255 * ((gradx - minVal) / (maxVal - minVal)))
gradx = gradx.astype("uint8")

print(np.array(gradx).shape)
ShowImage('gradx_card', gradx)

# 通过闭操作，先膨胀后腐蚀，将数字连接在一块
gradx = cv2.morphologyEx(gradx, cv2.MORPH_CLOSE, rectKernel)
ShowImage('gradx_card', gradx)

# THRESH_OTSU会自动寻找合适的阈值，适合双峰，需要把阈值设置为0
thresh = cv2.threshold(gradx, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
ShowImage('thresh_card', thresh)

# 再来一个闭合操作，填充白框内的黑色区域
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
ShowImage('thresh2_card', thresh)

# 计算轮廓
threshCnts, his = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 2)
ShowImage('contour_card', cur_img)

locs = []
# 遍历轮廓
for (i, c) in enumerate(cnts):  # 函数用于遍历序列中的元素以及它们的下标
    # 计算矩形
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    # 选择合适的区域，根据实际任务来，这里是四个数字为一组
    if ar > 2.5 and ar < 5.0:
        if (w > 40 and w < 85) and (h > 10 and h < 20):
            # 把符合的留下
            locs.append((x, y, w, h))

# 将符合的轮廓根据x的值，从左到右排序
locs = sorted(locs, key=lambda x: x[0])

output = []
# 遍历轮廓中的每一个数字
for (i, (gx, gy, gw, gh)) in enumerate(locs):
    # 初始化链表
    groupOutput = []
    # 根据坐标提取每一个组，往外多取一点，要不然看不清楚
    group = gray[gy - 5:gy + gh + 5, gx - 5:gx + gw + 5]
    ShowImage('group', group)
    # 预处理
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  # 二值化
    ShowImage('group', group)

    # 找到每一组的轮廓
    digitCnts, his = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # digitCnts = sortContours(digitCnts, method="LefttoRight")[0]
    # 对找到的轮廓进行排序
    digitCnts = sort_contours(digitCnts, method="left-to-right")[0]

    # 计算每一组中的每一个数值
    for c in digitCnts:
        # 找到当前数值的轮廓，resize成合适的大小
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        ShowImage('roi', roi)
        scores = []
        for (digit, digitROI) in digits.items():
            # 模板匹配
            #
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        # 得到最合适的数字
        groupOutput.append(str(np.argmax(scores)))

        # 画矩形和字体
        cv2.rectangle(image, (gx - 5, gy - 5), (gx + gw + 5, gy + gh + 5), (0, 0, 255), 1)
        cv2.putText(image, "".join(groupOutput), (gx, gy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        # 得到结果
        output.extend(groupOutput)

ShowImage('card_result', image)