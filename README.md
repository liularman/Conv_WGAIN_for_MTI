# Conv_WGAIN_for_MTI

*Read this in [English](README_en.md).*

## 概览
本工具是基于[Conv_WGAIN](https://kns.cnki.net/kcms2/article/abstract?v=6Zsqnb4eDBVfRzgCdj1ce6Xy-LenR3oy2cLqVtA492kfQTvadKxo2XLmsrm9idYh9NGRE7A8PieyWnx5SfIPVFRQeAlJH08Ei9A0dM0xLrseEHssEGs_y2LTq6hAdzhzNHk-DQ9ihwc=&uniplatform=NZKPT&flag=copy)模型开发的多元时序数据填充工具，模型填充原理是生成器根据存在数据分布进行学习，然后生成数据填充缺失部分的数据，让判别器无法识别出真实数据和生成数据。该模型的论文已被中文核心期刊《计算机科学与工程》录用发表。本工具利用Flask框架编写了一个WEB端的填充工具，可以对任意的多元时序数据进行缺失值填充。用于填充数据的Conv_WGAIN模型不是预训练好的模型，因为该模型运行速度极快，这也让该工具可以广泛用于不同的数据集。

## 环境依赖
- Python 3.8.13
- Pandas 1.4.4
- Flask 2.3.1 
- NumPy 1.21.5
- PyTorch 1.12.1

## 用法
1. 进入项目目录下，运行下面命令来安装环境依赖

    ~~~shell
    cd path/to/Conv_WGAIN_for_MTI
    pip install -r requirements.txt
    ~~~

2. 运行flask_web.py脚本启动flask服务器

    ~~~shell
    python flask_web.py
    ~~~

3. 浏览器打开[http://127.0.0.1:5000/](http://127.0.0.1:5000/)启动填充工具。填充工具页面中心框用来输入一些必要参数信息。“原始数据”表示缺失原数据的位置，建议将数据复制到项目中‘data’文件夹下。“保存路径”表示数据填充完后保存的位置。“迭代次数”表示模型训练迭代的次数。“更新频率”表示判别器每多少批次数据训练完后进行更新，这个值的范围一般是1~5。为了提示用户，在文本框中，设置了相应的提示词。

    ![](./images/homepage.png)

4. 填写好相关参数后，点击“开始填充”按钮后，填充开始。在填充过程中，按钮下面会出现一个进度条来表示填充进度，同时为了防止用户再次点击按钮，取消按钮的可点击属性，另外在按钮悬停在按钮上时，鼠标变成禁用符合来提示用户。

    ![](./images/imputating.png)

5. 填充完成后，会跳转到结果页面。告知用户数据的保存路径，同时将部分数据展示给用户查看填充效果。

    ![](./images/result.png)


## 引用
如果你觉得这个工作有帮助的话，请考虑引用论文
~~~BibTex
@article{{0},
 author = {刘子建,丁维龙,邢梦达,李寒 &amp; 黄晔},
 title = {Conv-WGAIN:面向多元时序数据缺失的卷积生成对抗插补网络模型},
 journal = {计算机工程与科学},
 volume = {45},
 number = {931-939},
 year = {2023},
 issn = {1007-130X}
 }
~~~