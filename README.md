# Mind+& Google MediaPipe库引申demo汇总

## Introduction

本仓库基于Mind+(行空板)在Google MediaPipe开源框架[[文档]](https://ai.google.dev/edge/mediapipe/solutions/guide)[[GitHub]](https://github.com/google-ai-edge/mediapipe)的基础上进行开发的`unihiker-aidemo`[[Gitee]](https://gitee.com/unihiker/unihiker-aidemo)(下称原仓库)的基础上进行了二次开发。主要收纳了`demo1`, `demo2`, `demo3`等几个项目，本仓库同时收纳了部分不借助Google MediaPipe，而仅依赖于[Tensorflow](https://www.tensorflow.org/)和[Keras](https://keras.io/)的项目，仓库的详细介绍与文档如下。

**注意:** 环境配置和初始化请参考行空板文档和`unihiker-aidemo`仓库[[说明文档]](https://www.unihiker.com.cn/wiki/ai_project)[[Gitee]](https://gitee.com/unihiker/unihiker-aidemo)

## Overview

| 名称              | 介绍                                                         | 文档                       | 程序                                          |
| ----------------- | ------------------------------------------------------------ | -------------------------- | --------------------------------------------- |
| 手势识别+风扇启停 | 本案例基于原仓库的`4-hand.py` demo进行二次开发。主要实现了基于手指位置判断基础上的手势识别，并且可以根据手势控制风扇的启停 | [doc](#手势识别风扇启停)   | [code](demo/gesture_reco_fan/gesture_reco.py) |
| "三庭五眼"颜值打分    | 基于原仓库的`3-face_mesh.py`进行二次开发，原代码参考自[论坛博客](https://mc.dfrobot.com.cn/thread-313268-1-1.html)。主要实现了使用MediaPipe进行面部五眼进行锚点，然后计算其L2范数诱导的距离作为评判标准，给出颜值评分 | [doc](#三庭五眼颜值打分) |[code](demo/appearance_eval/face_eval.py)|
| 基于CNN的垃圾分类检测 | 论坛地址：https://mc.dfrobot.com.cn/thread-319039-1-1.html | [doc](#自行车智能安全辅助系统) |			|
| 自行车智能安全辅助系统 | 论坛地址：https://mc.dfrobot.com.cn/thread-313774-1-1.html | [doc](#基于CNN的垃圾分类检测) |	|
|  |  |	|	|



## 手势识别+风扇启停

### 效果展示

<img src="img/hand1.jpg" alt="hand1" style="zoom: 35%;" />
<img src="img/hand2.jpg" alt="hand2" style="zoom: 35%;" />





## "三庭五眼"颜值打分

源代码来自博客：https://mc.dfrobot.com.cn/thread-313268-1-1.html

> 五眼：指脸的宽度比例，以眼形长度为单位，把脸的宽度分成五个等分，从左侧发际至右侧发际，为五只眼形。两只眼睛之间有一只眼睛的间距，两眼外侧至侧发际各为一只眼睛的间距，各占比例的1/5。





## 基于CNN的垃圾分类检测

源代码论坛地址：https://mc.dfrobot.com.cn/thread-319039-1-1.html



## 自行车智能安全辅助系统

源代码论坛地址：https://mc.dfrobot.com.cn/thread-313774-1-1.html







**To Be Continue...**
