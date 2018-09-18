# shape_based_matching  

try to implement halcon shape based matching, refer to machine vision algorithms and applications, page 317 3.11.5, written by halcon engineers  
We find that shape based matching is the same as linemod. [linemod pdf](Gradient%20Response%20Maps%20for%20Real-TimeDetection%20of%20Textureless%20Objects.pdf)  

## thoughts about the method

The key of shape based matching, or linemod, is using gradient orientation only. Though both edge and orientation are resistant to disturbance,
edge have only 1bit info(there is an edge or not), so it's hard to dig wanted shapes out if there are too many edges, but we have to have as many edges as possible if we want to find all the target shapes. It's quite a dilemma.  

However, gradient orientation has much more info than edge, so we can easily match shape orientation in the overwhelming img orientation by template matching across the img.  

Speed is also important. Thanks to the speeding up magic in linemod, we can handle 1000 templates in 20ms or so.  

[Chinese blog about the thoughts](https://www.zhihu.com/question/39513724/answer/441677905)  

## improvment

Comparing to opencv linemod src, we improve from 5 aspects:  

1. delete depth modality so we don't need virtual func, this may speed up  

2. opencv linemod can't use more than 63 features. Now wo can have up to 8191  

3. simple codes for rotating and scaling img for training. see test.cpp for examples  

4. nms for accurate edge selection  

5. one channel orientation extraction to save time, slightly faster for gray img

## some test

### Example for circle shape

![circle1](test/case0/1.jpg)
![circle1](test/case0/result/1.png)  

![circle2](test/case0/2.jpg)
![circle2](test/case0/result/2.png)  

![circle3](test/case0/3.png)
![circle3](test/case0/result/3.png)  

### circle template before and after nms  

#### before nms

![before](test/case0/features/no_nms_templ.png)

#### after nms

![after](test/case0/features/nms_templ.png)  

### Simple example for arbitary shape

Well, the example is too simple to show the robustness  
running time: 1024x1024, 60ms to construct response map, 7ms for 360 templates  

test img & templ features  
![test](./test/case1/result.png)  
![templ](test/case1/templ.png)  


### noise test  

![test2](test/case2/result/together.png)  
