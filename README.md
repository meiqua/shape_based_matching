# shape_based_matching  

try to implement halcon shape based matching, refer to machine vision algorithms and applications, page 317 3.11.5, written by halcon engineers  
We find that shape based matching is the same as linemod.  
[linemod pdf](Gradient%20Response%20Maps%20for%20Real-TimeDetection%20of%20Textureless%20Objects.pdf)  
Comparing to opencv linemod src, we improve from three aspects:  

1. delete depth modality so we don't need virtual func, this may speed up  

2. opencv linemod can't use more than 63 features. Now wo can have up to 8191  

3. simple codes for rotating and scaling img for training. see test.cpp for examples  

test img & templ features  
The example is too simple to explain the robustness  
running time: 1024x1024, 60ms to construct response map, 7ms for 360 templates  

![test](./test/case1/result.png)
![templ](test/case1/templ.png)
