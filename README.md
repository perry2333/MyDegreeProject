# MyDegreeProject
upload all my code for my degree project

该项目主要通过matlab和python混合环境开展

使用的matlab版本是R2022b，对应的是python10

理论上只要matlab跟python的版本相适应就能成功运行

需要注意matlab调用python时版本的适配

还要安装好Pyroomacoustics和POT库，只要这两个库的示例代码能跑通，应该就能work

large_test.m用于大批次仿真测试

testing2.mlx用于画图

testing.mlx在中期报告及之前进行使用，保留着大量探索痕迹

easy_design.ipynb中有一些简单案例

self_design.ipynb中进行了对生成SRIR的探索

POT_testing.ipynb中对POT算法进行了探索

POT_calculate.py与R_SRIR_G.py是与matlab对接的接口函数

Pyroomacoustics中自带了is_inside函数，用于判断一个二维或三维坐标是否在房间内，可以直接使用

可以不需要替换源代码，只需要使用is_inside函数就可以了

我在R_SRIR_G.py便使用了is_inside函数判断了参考点之间的连线是否都在房间内，代码如下：

    if pair_check == 1:   #自己写的检测逻辑，运用了room库中自带的is_inside函数

        # 将列表或数组转换为 NumPy 数组
        
        pre_pos = np.array(pre_pos)
        
        mic_center = np.array(mic_center)
        
        try:
        
            #print('here')
            
            #check_list = [i for i in ]
            
            for i in range(1, 101):
            
                if not room.is_inside(i/100*pre_pos + (1-i)/100*mic_center):
                
                    raise ValueError("The pair is not suitable for evaluation.")
                    
        except ValueError as e:
        
            if str(e) == "The pair is not suitable for evaluation.":

                print("The pair is not suitable for evaluation.")
                
                return 2
                
# 主要参考文献
>1. Tervo S, Tynen J P, Kuusinen A, et al. Spatial Decomposition Method for Room Impulse Responses[J]. J. Audio Eng. Soc., 2013, 61(1).
>2. Geldert A. Room Impulse Response Interpolation via Optimal Transport[J].
