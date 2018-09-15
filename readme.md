# repeated stage labeling tool
+ i: 标出视频中出现细胞放进培养皿的时间。
+ t: 标出tPNf时间节点，也就是原核出现的时间
+ 2、3、4、5: 分别代表t2、t3、t4、t4+节点，也就是细胞个数
+ l: 出现异常。只用标异常出现不用标结束。默认异常会在与它最接近的非异常帧前结束。
+ z: 撤销上一步。最多只能撤销到加载这个视频后加的一系列时间点。
+ c: 清除当前视频的全部标签，该步骤不可撤回。
+ 其他的和以前一样，q和e分别是前、后一张，a和d分别是前、后十二张，[和]分别是前、后一个项目。
+ <b>如果有所改动，请push到自己的分支。</b>

所有图像相关的在utils里，生成数据集相关的在generate_labels里。
utils里用于处理视频的函数包括读取视频,如果要resize，多加一个参数即可。
generate_labels里包含了生成标签、打包为npz、制作early fusion数据集等函数。 

# ellipse_label.py
+ a,d: 左右翻图片
+ i: 开始画图，图像中心会出现一个初始的椭圆
+ 鼠标拖动长轴短轴的端点，会对长短轴长度和角度进行调整
+ 鼠标拖动直线交叉中心点，会对椭圆的中心位置进行调整
+ s: 保存，将椭圆的参数以及一个01mask存到指定位置的相同文件名的npz文件里
+ z: 重新初始化椭圆

保存之后若要重新画图，就按i初始圆，再s保存，就可以覆盖之前的记录。保存后按z无效果。
