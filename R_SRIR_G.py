def Room_SRIR_Generater(source, mic_center, number='1', pair_check=0, pre_pos=0, mic_type='array', Max_Order = 5, room_type = 1):
    #该函数只有前两个参数是必需值，其它都有默认值
    #source是声源的三维坐标
    #mic_center是阵列麦克风的中心坐标
    #number用于确定导出数据的命名
    #pair_check在生成第二个麦克风的时候需要标记为1，用于确定第二个麦克风与第一个麦克风的连线是否在模型内
    #pre_pos即第一个麦克风阵列的位置，在生成第二个麦克风的时候填入
    #mic_type可选为'array'或'single'，single即为单麦克风，似乎没有经过调试（
    #Max_Order为镜像声源法的仿真阶数，默认为5阶（仿真速度较慢）
    #room_type = 1是ZJ425,2是矩形房间
    from tqdm import tqdm
    import time
    from scipy.io import wavfile
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import fftconvolve
    import IPython
    import pyroomacoustics as pra
    from scipy.io import savemat
    #加载自定义建筑的房间STL模型
    from mpl_toolkits import mplot3d

    try:
        from stl import mesh
    except ImportError as err: #报警处理
        print(
            "The numpy-stl package is required for this example. "
            "Install it with `pip install numpy-stl`"
        )
        raise err

    def stl_load(file_path="3D_model_file/zj425.stl",MaxOrder=5):
    # stl模型读取函数
        material = pra.Material(energy_absorption=0.2, scattering=0.1)

        # with numpy-stl
        the_mesh = mesh.Mesh.from_file(file_path)
        ntriang, nvec, npts = the_mesh.vectors.shape
        size_reduc_factor = 500.0  # to get a realistic room size (not 3km)

        # create one wall per triangle
        walls = []
        for w in range(ntriang):
            walls.append(
                pra.wall_factory(
                    the_mesh.vectors[w].T / size_reduc_factor,
                    material.energy_absorption["coeffs"],
                    material.scattering["coeffs"],
                )
            )

        room = pra.Room(
                walls,
                fs=16000,
                max_order=MaxOrder, #调整阶数
                ray_tracing=False,
                air_absorption=True,
            )
        return room

    def SDML7(d=0.1, r=[0, 0, 0]):
    # 阵列麦克风定义函数
        mic_7 = np.array([[d / 2, -d / 2, 0, 0, 0, 0, 0],
                          [0, 0, d / 2, -d / 2, 0, 0, 0],
                          [0, 0, 0, 0, d / 2, -d / 2, 0]])

        # 将 r 转换为数组
        r = np.array(r)
        
        # 将 r 添加到 mic_7 中的每个元素
        mic_7 = mic_7 + r[:, np.newaxis]
        
        return mic_7

    
    try:
        # 在这里手动设置文件路径
        if room_type == 1:
            room = stl_load(file_path="D:/MATLAB/Matlab_HaHa/Degree_project/3D_model_file/zj425.stl", MaxOrder = int(Max_Order))
        elif room_type == 2:
            corners = np.array([[-10,-12.5], [-10,0], [10,0], [10,-12.5]]).T 
            fs = 16000
            room = pra.Room.from_corners(corners, fs=fs, max_order=int(Max_Order), materials=pra.Material(0.2, 0.1), ray_tracing=False, air_absorption=True)
            room.extrude(5., materials=pra.Material(0.2, 0.1))
        #添加声源和麦克风
        fs=16000
        room.add_source(source)
    except ValueError as e:
        if str(e) == "The source must be added inside the room.":
            print("The source must be added inside the room.")
            return 0
        
    try:
        if mic_type == 'array':
            mic = SDML7(0.1, mic_center)
            room.add_microphone_array(mic) #源代码中的这个函数里添加了检测逻辑
        elif mic_type == 'single':
            mic = mic_center
            room.add_microphone(mic)
        else:
            raise ValueError("Unknown mic_type: {}".format(mic_type))
            print(str(ValueError))
            return 1
        
    except ValueError as e:
        if str(e) == "The microphone must be added inside the room.":
            print("The microphones must be added inside the room.")
            return 1
            
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

        
        
    # compute image sources
    room.image_source_model()
    
    room.compute_rir()
    
    
    # 导出 RIR 与 mic array 的坐标

    rir_name = 'rir'+str(number)
    mic_position_name = 'mic_position'+str(number)
    # 找到最小的数组长度
    min_length = min(len(arr[0]) for arr in room.rir)

    # 创建一个新的列表，其中包含所有数组的修剪版本
    trimmed_rir = [arr[0][:min_length] for arr in room.rir]

    # 现在所有的数组都具有相同的长度，可以保存到.mat文件中
    savemat(rir_name+'.mat', {rir_name: trimmed_rir})
    savemat(mic_position_name+'.mat', {mic_position_name: mic})
    return 3
 #   
#with open('smparameters.txt', 'r') as file:
  #  lines = file.readlines()

# 解析参数
#param1 = eval(lines[0])  # 第一个参数是一个列表
#param2 = eval(lines[1])  # 第二个参数是一个列表
#param3 = str(lines[2])   # 第三个参数是一个整数

#h = Room_SRIR_Generater(param1, param2, param3)