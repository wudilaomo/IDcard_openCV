#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import myutils
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


template = cv2.imread('number.png')


# In[3]:


template1 = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)


# In[9]:


cv2.imshow('template',template1)
cv2.waitKey(0)
cv2.destoryALLWindows() 


# In[ ]:





# In[10]:


ref = cv2.threshold(template1, 25, 255, cv2.THRESH_BINARY_INV)[1]


# In[11]:


cv2.imshow('threshold',ref)
cv2.waitKey(0)
cv2.destoryALLWindows() 


# In[12]:


binary,contours = cv2.findContours(ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


# In[13]:


ref_Copy = template.copy()


# In[14]:


res = cv2.drawContours(ref_Copy, binary, -1, (0, 0, 255), 2)


# In[15]:


cv2.imshow('res',res)
cv2.waitKey(0)
cv2.destoryALLWindows() 


# In[ ]:





# In[16]:


print(np.array(binary).shape)

binary = myutils.sort.contours(binary, method="right-to-left")[0]
# In[17]:


digits={}


# In[ ]:





# In[18]:


for(i,c)in enumerate(binary):
    (x, y, w, h)=cv2.boundingRect(c)
    roi = ref[y:y+h, x:x+w]
    roi = cv2.resize(roi, (57, 88))
    digits[i]= roi


# In[ ]:





# In[19]:


rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(13,7))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))


# In[20]:


img = cv2.imread('card.jpg')


# In[21]:


cv2.imshow('res',img)
cv2.waitKey(0)
cv2.destoryALLWindows() 


# In[22]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[23]:


cv2.imshow('res',gray)
cv2.waitKey(0)
cv2.destoryALLWindows() 


# In[ ]:





# In[24]:


tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)


# In[ ]:





# In[25]:


gradX = cv2.Sobel(tophat, cv2.CV_32F, 1, 0, ksize=-1)


# In[ ]:





# In[26]:


gradX = np.absolute(gradX)


# In[27]:


(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")


# In[28]:


cv2.imshow('res',gradX)
cv2.waitKey(0)
cv2.destoryALLWindows() 


# In[29]:


rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(23,23))


# In[30]:


gradX1 = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)


# In[31]:


cv2.imshow('res',gradX1)
cv2.waitKey(0)
cv2.destoryALLWindows() 


# In[ ]:





# In[32]:


thresh = cv2.threshold(gradX1, 0, 255,
                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


# In[33]:


cv2.imshow('res',thresh)
cv2.waitKey(0)
cv2.destoryALLWindows() 


# In[ ]:





# In[34]:


binary2,contours2 = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
img_Copy = img.copy()
binary_Copy = binary2
card = cv2.drawContours(img_Copy, binary_Copy, -1, (0, 0, 255), 2)


# In[35]:


cv2.imshow('card',card)
cv2.waitKey(0)
cv2.destoryALLWindows() 


# In[ ]:





# In[37]:


locs = []


# In[38]:


for (i,c) in enumerate(binary_Copy):
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    
    if ar > 2.5 and ar<4.0:
        if (w>100 and w<150) and (h>37 and h<50):
            locs.append((x, y, w, h))
            


# In[42]:


print(locs)


# In[ ]:





# In[44]:


locs = sorted(locs, key=lambda x:x[0])
output = []
print(locs)


# In[69]:


for (i,(gx, gy, gw, gh)) in  enumerate(locs):
    # 初始化链表
    groupOutput = []
    # 根据坐标提取每一个组，往外多取一点，要不然看不清楚
    group = gray[gy-5:gy+gh+5,gx-5:gx+gw+5]
    # 预处理
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  # 二值化
 
    # 找到每一组的轮廓
    digitCnts, his = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # digitCnts = sortContours(digitCnts, method="LefttoRight")[0]
    # 对找到的轮廓进行排序
    digitCnts = sorted(digitCnts, key=lambda x: cv2.boundingRect(x)[0])
 
    # 计算每一组中的每一个数值
    for c in digitCnts:
        # 找到当前数值的轮廓，resize成合适的大小
        (x,y,w,h) = cv2.boundingRect(c)
        roi = group[y:y+h, x:x+w]
        roi = cv2.resize(roi, (57,88))
        scores = []
        for(digit, digitROI) in digits.items():
            # 模板匹配
            #
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        # 得到最合适的数字
        groupOutput.append(str(np.argmax(scores)))
 
        # 画矩形和字体
        cv2.rectangle(img, (gx - 5, gy - 5), (gx+gw+5, gy+gh+5), (0,0,255),1)
        cv2.putText(img, "".join(groupOutput), (gx, gy-15), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0,0,255),2)
        # 得到结果
        output.extend(groupOutput)


# In[ ]:





# In[70]:


print("Credit Card #: {}".format("".join(output)))


# In[71]:


cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destoryALLWindows() 


# In[ ]:




