% 该函数能够进行大规模的仿真测试实验，注意所用python版本是python10
% 基本上对应Pyroomacoustics和POT库中的notebook代码可以跑通，该代码便可以运行


clear

%设置 Python 解释器的路径
pyenv('Version','D:\anacoda\envs\python10\python.exe')
%在指定环境中运行 Python 程序
cd('D:\MATLAB\Matlab_HaHa\Degree_project')

if count(py.sys.path,'D:\MATLAB\Matlab_HaHa\Degree_project') == 0
    insert(py.sys.path,int32(0),'D:\MATLAB\Matlab_HaHa\Degree_project');
end
py.importlib.import_module('numpy');%调用numpy
py.importlib.import_module('R_SRIR_G');%生成SRIR的代码封装在R_SRIR_G module中
py.sys.path;

fs = 16e3;
RT60 = 1500;%ms
epoch_num = 100;%确定测试的次数
K = 0.05:0.05:0.95;%确定插值点测试位置
Room_Type = 2;     %确定使用的虚拟环境模型，1是zj425，2是鞋盒
if Room_Type == 1
        range_source = [-20 20; -25 0; -6 0]; % 声源位置范围
        range_mic = [-20 20; -25 0; -6 0]; % 麦克风位置范围
elseif Room_Type == 2
        range_source = [-20 20; -25 0; 0 6]; % 声源位置范围
        range_mic = [-20 20; -25 0; 0 6]; % 麦克风位置范围
end
Image_source_order = 4; % 镜像源法仿真阶数，阶数小于等于4计算较快，阶数大于5以上计算会十分缓慢
Pair_Distance =4 %[0.125 0.25 0.5 1 2 4] ;    % 确定参考点之间的插值距离
pair_str = 4 %[0125;025;05;1;2;4 ];                % 仅用于确定画图时的名称
Special = '_鞋盒20db';                     %每次仿真前记得换一个名字；

SpError_list = zeros(epoch_num*3,length(K));    %储存频谱误差
SpError_list_W = zeros(epoch_num*3,length(K));  %储存通过窗函数处理后的频谱误差
AbError_list = zeros(epoch_num*3,length(K));    % 储存归一化时间平均误差
AbError_list_W = zeros(epoch_num*3,length(K)); % 储存通过窗函数的归一化频谱误差
Position_Log = zeros(epoch_num,9); %分别是[声源，参考点1，参考点2] % 储存每一次仿真的声源与参考点的位置
VOTE = zeros(epoch_num*length(K), length(Pair_Distance)); % 储存每次实验时三种方法中的最优方法，进行投票
pair_flag = 0; %记录当前位于第几个插值距离中
for pair_distance = Pair_Distance
    pair_flag = pair_flag + 1;
    testing_num = 1;
    for epoch = 1 : epoch_num
        epoch_str = 'epoch: ' + string(epoch);
        disp(epoch_str);
        %确定source与mic位置，获取SRIR
        
        %一个控制生成逻辑的模块
        check = 0;
        while check == 0
            h = py.int(0); % 初始化 h 为 0   %控制随机生成声源和麦克风位置时逻辑判断的变量，尽量不要改动
            % 错误代码：h=0源位置不在建筑内，h=1麦克风不在建筑内，h=2麦克风对连线不在建筑内
            % h=3 SUCCESS
            while h == 0
                disp('1') % 确定程序运行位置，下同
                % 在范围内随机生成声源和麦克风的位置
                source_position = rand(3, 1) .* (range_source(:, 2) - range_source(:, 1)) + range_source(:, 1);
                first_mic_position = rand(3, 1) .* (range_mic(:, 2) - range_mic(:, 1)) + range_mic(:, 1);

                % 调用 Room_SRIR_Generater 函数
                h = py.R_SRIR_G.Room_SRIR_Generater(source_position, first_mic_position, '1', mic_type='array', Max_Order = Image_source_order, room_type = Room_Type);

                % 如果 h 为 0，则继续循环
            end
            h =1; % 逻辑控制目的，尽量不要改动
            while h == 1
                disp('2')
                boom = 1; % 有时候第二个麦克风的生成会遇到死循环，使用boom控制死循环控制，循环过多时重新生成两个麦克风
                % 重新生成麦克风位置
                first_mic_position = rand(3, 1) .* (range_mic(:, 2) - range_mic(:, 1)) + range_mic(:, 1);

                % 调用 Room_SRIR_Generater 函数
                h = py.R_SRIR_G.Room_SRIR_Generater(source_position, first_mic_position, '1', mic_type='array', Max_Order = Image_source_order, room_type = Room_Type);
                    while h == 3
                        disp('3')
                        
                        second_mic_position = mic_pair(first_mic_position, pair_distance);
                        % 调用 Room_SRIR_Generater 函数
                        h = py.R_SRIR_G.Room_SRIR_Generater(source_position, second_mic_position, '2', pair_check = 1, pre_pos = first_mic_position, mic_type='array', Max_Order = Image_source_order, room_type = Room_Type);
                        if h == 3
                            break
                        elseif h==2 || h==1
                            disp('4')
                            boom = boom+1;
                            h = 3;
                            if boom == 10 %防锁死
                                h = 1;
                            end
                        end
                        % 如果 h 为 1或2，则继续循环
                    end
                    % 如果 h 为 1，则继续循环
            end

            load('rir1.mat');                                    %储存的文件供POT进行处理
            load('mic_position1.mat');
            load('rir2.mat');
            load('mic_position2.mat');
            if all(rir1(:) == 0) || all(rir2(:) == 0)      %有可能会出现都是零的情况，则全部重新仿真
                check = 0;
                %麦克风对有效性核验
            elseif h == 2
                check = 0;
            elseif h == 3
                check = 1;
            end
        end
        % 读取位置信息
        Position_Log(epoch_num,:) = [source_position' first_mic_position' second_mic_position'];
        Source_Position = source_position'
        Mic_Position1 = first_mic_position'
        Mic_Position2 = second_mic_position'

        %获取了SRIR1和SRIR2

        %进行SDM并导出点云
        PCld_1 = SRIR2PCLD(rir1, mic_position1, '1', fs);
        PCld_2 = SRIR2PCLD(rir2, mic_position2, '2', fs);
        %     h_ori_1 = PCld_rendering(PCld_1.PointCloud(:,1:3), PCld_1.PointCloud(:,4), first_mic_position', fs, RT60);
        %     h_ori_2 = PCld_rendering(PCld_2.PointCloud(:,1:3), PCld_2.PointCloud(:,4), second_mic_position', fs, RT60);
        %两个点云已导出

        %运行POT.py程序，获取传输矩阵和log记录
        py.importlib.import_module('POT_calculate'); % 涉及POT的处理放在POT_calculate模块中
        py.POT_calculate.POT_calculate();
        T = load("T.mat");
        log_emd = load("log_emd.mat"); % 这个数据没啥用
        x_s = PCld_1.PointCloud(:,4); % 源点云图和目标点云图质量信息
        x_t = PCld_2.PointCloud(:,4);
        %通过质量守恒计算dummy point的质量分布
        x_s_remain = abs(x_s/sum(x_s)) - sum(T.T,2);
        x_t_remain = abs(x_t/sum(x_t)) - sum(T.T,1)';

        u = x_s_remain;
        v = x_t_remain;
        %已获取传输矩阵

        %进行插值，rendering和误差比较
        %插值应当写在一个函数中，可以确定k_与插值点的仿真值
        for k = 1 : length(K)
            k_str = 'k: ' + string(K(k));
            disp(k_str);
            [PCld_interp, PCld_interp_z] = POT_interp(PCld_1, PCld_2, T.T, u, v, K(k)); % 完成插值点云图
            PCld_interp_actual_pos = (1-K(k))*first_mic_position + K(k)*second_mic_position;
            h = 0; % 初始化 h 为 0

            % 生成插值位置的实际仿真值
            while h == 0
                % 调用 Room_SRIR_Generater 函数
                h = py.R_SRIR_G.Room_SRIR_Generater(source_position, PCld_interp_actual_pos, '3', mic_type='array', Max_Order = Image_source_order, room_type = Room_Type);
                % 如果 h 为 0，则继续循环
            end

            load('mic_position3.mat');
            load('rir3.mat');
            PCld_3 = SRIR2PCLD(rir3, mic_position3, '3', fs);
            % h_TimeLinear_interp即简单线性方法
            h_TimeLinear_interp = PCld_rendering([PCld_1.PointCloud(:,1:3); PCld_2.PointCloud(:,1:3)], [(1-K(k))*PCld_1.PointCloud(:,4); K(k)*PCld_2.PointCloud(:,4)], PCld_interp_actual_pos', fs, RT60);%linearTime_interp(h_ori_1,h_ori_2,K(k));
            h_interp = PCld_rendering(PCld_interp_z, PCld_interp, PCld_interp_actual_pos', fs, RT60);
            h_actual = PCld_rendering(PCld_3.PointCloud(:,1:3), PCld_3.PointCloud(:,4), PCld_interp_actual_pos', fs, RT60);
            hann_length = 0.004;
            h_TimeLinear_interp_norm = Hann_filter(h_TimeLinear_interp, hann_length, fs, 1, 'POT time_interpolated');
            h_interp_norm = Hann_filter(h_interp, hann_length, fs, 1, 'POT interpolated');
            h_actual_norm = Hann_filter(h_actual, hann_length, fs, 1, 'actual position');
            % h_interp_linear为空间线性方法
            PCld_interp_linear = linearPCld_interp(PCld_1.PointCloud, PCld_2.PointCloud, K(k));
            h_interp_linear = PCld_rendering(PCld_interp_linear(:,1:3), PCld_interp_linear(:,4), PCld_interp_actual_pos', fs, RT60);
            h_interp_linear_norm = Hann_filter(h_interp_linear, hann_length, fs, 1, 'linear interpolated');
            
            %计算一系列误差
            Spectral_Error_POT_W = spectral_error(h_interp_norm, h_actual_norm, fs)
            Spectral_Error_Linear_W = spectral_error(h_interp_linear_norm, h_actual_norm, fs)
            Spectral_Error_TimeLinear_W = spectral_error(h_TimeLinear_interp_norm, h_actual_norm, fs)

            Spectral_Error_POT = spectral_error(h_interp, h_actual, fs)
            Spectral_Error_Linear = spectral_error(h_interp_linear, h_actual, fs)
            Spectral_Error_TimeLinear = spectral_error(h_TimeLinear_interp, h_actual, fs)

            Ab_Error_POT_W = calculate_absolute_error(h_interp_norm, h_actual_norm)
            Ab_Error_Linear_W = calculate_absolute_error(h_interp_linear_norm, h_actual_norm)
            Ab_Error_TimeLinear_W = calculate_absolute_error(h_TimeLinear_interp_norm, h_actual_norm)

            Ab_Error_POT = calculate_absolute_error(h_interp, h_actual)
            Ab_Error_Linear = calculate_absolute_error(h_interp_linear, h_actual)
            Ab_Error_TimeLinear = calculate_absolute_error(h_TimeLinear_interp, h_actual)
            %已完成误差比较

            %记录当前epoch所用的source和mic位置与插值效果
            SpError_list(epoch, k) = Spectral_Error_POT;
            SpError_list(epoch + epoch_num, k) = Spectral_Error_Linear;
            SpError_list(epoch + epoch_num*2, k) = Spectral_Error_TimeLinear;

            SpError_list_W(epoch, k) = Spectral_Error_POT_W;
            SpError_list_W(epoch + epoch_num, k) = Spectral_Error_Linear_W;
            SpError_list_W(epoch + epoch_num*2, k) = Spectral_Error_TimeLinear_W;

            AbError_list(epoch, k) = Ab_Error_POT;
            AbError_list(epoch + epoch_num, k) = Ab_Error_Linear;
            AbError_list(epoch + epoch_num*2, k) = Ab_Error_TimeLinear;

            AbError_list_W(epoch, k) = Ab_Error_POT_W;
            AbError_list_W(epoch + epoch_num, k) = Ab_Error_Linear_W;
            AbError_list_W(epoch + epoch_num*2, k) = Ab_Error_TimeLinear_W;
            
            Ab_Error_SpatialLinear_W = Ab_Error_Linear_W;
            Ab_Error_SimpleLinear_W = Ab_Error_TimeLinear_W;

            [~, order] = min([Ab_Error_POT_W, Ab_Error_SpatialLinear_W, Ab_Error_SimpleLinear_W]);

            VOTE(testing_num,pair_flag) = order;
            testing_num = testing_num+1;
        end
%         sum_POT = sum(AbError_list_W(epoch, :),2);
%         sum_SpatialLinear = sum(AbError_list_W(epoch + epoch_num, :),2);
%         sum_SimpleLinear = sum(AbError_list_W(epoch + epoch_num*2, :),2);
        
        

    end
%     % 构造文件名
%     filename = 'SquareError_list_W_'+ string(pair_str(pair_flag))+ Special+ '.mat';
% 
%     % 保存数据到文件中
%     save(filename, 'AbError_list_W');
     % 构造文件名
    filename = 'SpError_list_W_'+ string(pair_str(pair_flag))+ Special+ '.mat';

    % 保存数据到文件中
    save(filename, 'SpError_list_W');
   % 调用新函数来绘制图表
%     plot_error_curves(SpError_list, SpError_list_W, AbError_list, AbError_list_W, ...
%         'Spectral Error POT vs. Linear vs.TimeLinear', 'Spectral Error POT vs. Linear (W) vs. TimeLinear', ...
%         'Absolute Error POT vs. Linear vs. TimeLinear', 'Absolute Error POT vs. Linear (W) vs. TimeLinear', K, epoch_num);
end

%% 函数部分
function second_mic = mic_pair(first_mic, distance)
% 生成随机角度
theta = rand * 2 * pi; % 在 [0, 2*pi] 范围内生成随机角度
phi = rand * pi; % 在 [0, pi] 范围内生成随机角度

% 计算球面坐标系中的位置
x = distance * sin(phi) * cos(theta);
y = distance * sin(phi) * sin(theta);
z = distance * cos(phi);

% 将球面坐标系转换为直角坐标系
second_mic = first_mic + [x; y; z];
end

function PointCloud = SRIR2PCLD(SRIR, mic_position, order, fs)
% 从SRIR直接生成点云图

% 处理RIR数组
if iscell(SRIR)
    % 获取cell数组的大小
    %[num_cells, ~, Len] = size(rir);
    [num_cells, ~] = size(SRIR);
    % 初始化一个空矩阵来存储转换后的数据
    rir_length = [];

    % 循环遍历每个单元格
    for i = 1:num_cells
        Length = length(SRIR{i});
        rir_length = [rir_length; Length];
    end
    limit_len = min(rir_length);
    rir_data = zeros(num_cells,limit_len);

    for i = 1:num_cells
        rir_data(i,:) = SRIR{i}(1:limit_len);
    end
else
    rir_data = SRIR;%reshape(rir, [7, l]); % 转换矩阵尺寸为 [7, N]
end
% 处理RIR数组完毕

% 进行SDM
p.micLocs = mic_position';
p.fs = fs;
p.c = 345;
p.winLen = 0;
p.parFrames = 8192;
p.showArray = 0;
DOA = SDMPar(rir_data',p);
% SDM完成

% 处理并生成点云
% 生成一个逻辑向量，指示哪些值是非 NaN 值
valid_indices = ~isnan(DOA);

% 使用逻辑索引选取非 NaN 值
valid_DOA = DOA(valid_indices);
valid_DOA = reshape(valid_DOA, [], 3);

valid_amp_indice = valid_indices(:,1);
ori_amp = rir_data(7,:);
valid_amp = ori_amp(valid_amp_indice);

%尝试剔除幅值过小的点云
%利用直方图的统计剔除点云
% 计算直方图
[counts, edges] = histcounts(valid_amp.^2);
% 找到计数最多的组别的索引
[~, maxIndex] = max(counts);
% 获取具有最多计数的组别的范围
maxRange = [edges(maxIndex), edges(maxIndex + 1)];
pure_amp = [];
pure_DOA = [];
for i = 1:length(valid_amp)
    if abs(valid_amp(i)^2) >= abs(maxRange(2))/3*2 %计数最高的组别的2/3以外被保留
        pure_amp = [pure_amp, valid_amp(i)];
        pure_DOA = [pure_DOA; valid_DOA(i,:)];
    end
end
% scaleFactor = 2e3;
%pointcloud_plot(pure_amp, pure_DOA, scaleFactor)
%导出剔除后的点云图
PointCloud_export(pure_DOA, pure_amp', ['pure_square' order]);
PointCloud = load(['PointCloud_pure_square' order '.mat']);
end

function pointcloud_plot(amp, DOA, scaleFactor, Axis)
% 绘制点云图，将幅值映射为图像源半径，需要调整scaleFactor参数调整比例大小
% Axis具有默认输入
if nargin == 3
    Axis = [-60,60,-60,60,-30,30];
end
% 可以根据具体需求设计一个合适的映射关系，这里简单地将幅度信息乘以一个系数作为半径
% 将幅度信息映射到球的半径上
radius = abs(amp * scaleFactor); % scaleFactor 是一个缩放系数，用于调整球的大小
% 绘制散点图
%figure;
scatter3(DOA(:,1), DOA(:,2), DOA(:,3), radius', 'filled');
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Three-dimensional scatter plot with amplitude-dependent spheres');
colorbar; % 添加颜色条，以显示幅度信息
hold on
% 读取STL文件
[vertices, ~, ~, ~] = stlread('D:\MATLAB\Matlab_HaHa\Degree_project\3D_model_file\zj425.stl');
room = triangulation(vertices.ConnectivityList, vertices.Points./1000);
% 显示STL模型
%trisurf(room, 'FaceColor', [0.8 0.8 0.8], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
axis(Axis)
hold off;
end

function PointCloud_export(DOA, amp, str)
% 导出点云图数据
PointCloud = [DOA amp];
% 将数字转换为字符串
%ad = num2str(abs(mic_position(2,7)));

% 构造文件名
filename = ['PointCloud_', str,'.mat'];

% 保存数据到文件中
save(filename, 'PointCloud');
end

function h = PCld_rendering(loca, pres, reci_loca, fs, RT60)
% 将点云图渲染成单声道RIR
fly_dist = sum((loca-repmat(reci_loca,length(loca),1)).^2,2).^0.5;
arr_time = fly_dist/345; %距离除以声速得到接收时的时间
[arr_time_sorted, sorted_order] = sort(arr_time);
t = zeros(ceil(RT60/1000*fs),1);
for k = 1:length(arr_time_sorted)
    t(floor(arr_time_sorted(k)*fs)) = t(floor(arr_time_sorted(k)*fs))+abs(pres(sorted_order(k)));
end
h = t;
end

function [r,z] = POT_interp(PCld_1, PCld_2, T, u, v, k_)
% 核心算法，所设计的POT插值算法
pure_T = T;
k = 1;%插入点的标号
%pure_T(abs(pure_T)<2e-3)=0;

T_ = pure_T;

for i = 1:length(u)
    if u(i) > 0
        if sum(pure_T(i,:)) == 0
            r(k) = (1-k_)*u(i);
            z(k,:) = PCld_1.PointCloud(i,1:3);
            k = k + 1;
        else
            T_(i,:) = T_(i,:) + (1-k_)*u(i)*pure_T(i,:)/sum(pure_T(i,:));
        end
    end
end
for j = 1:length(v)
    if v(j) > 0
        if sum(pure_T(:,j)) == 0
            r(k) = k_*v(j);
            z(k,:) = PCld_2.PointCloud(j,1:3);
            k = k + 1;
        else
            T_(:,j) = T_(:,j) + k_*v(j)*pure_T(:,j)/sum(pure_T(:,j));
        end
    end
end

[n,m] = size(T_);
for i = 1:n
    for j = 1:m
        if T_(i,j) > 0
            r(k) = T_(i,j);
            z(k,:) = (1-k_)*PCld_1.PointCloud(i,1:3)...
                + k_*PCld_2.PointCloud(j,1:3);
            k = k + 1;
        end
    end
end
end

function H_output_norm = Hann_filter(input, hann_length, fs, varargin)
% 将信号通过汉宁窗，具有绘图功能，需要调试（
% Parse optional input arguments
if nargin < 4 || isempty(varargin)
    plot_check = 0;
    Title = '';
else
    plot_check = varargin{1};
    if numel(varargin) > 1
        Title = varargin{2};
    else
        Title = '';
    end
end
% Apply Hann window
W_hann = hann(hann_length*fs);

% Convolve input with Hann window
H_output = conv(W_hann, input);

% Time vector for plotting
t_output = (0:length(H_output)-1) / fs;

% Normalize the output
H_output_norm = H_output / sum(H_output);

% Plot the output if plot_check is true
if plot_check == 1
    %     figure
    %     plot(t_output, H_output_norm);
    %     axis([0 max(t_output) 0 max(H_output_norm)])
    %     xlabel('Time (s)');
    %     ylabel('Normalized Amplitude');
    %     title(Title);

end
end
function error = spectral_error(signal1, signal2, fs)
% 计算信号的功率谱密度（PSD）
[pxx1, freq1] = pwelch(signal1, [], [], [], fs);
[pxx2, freq2] = pwelch(signal2, [], [], [], fs);

% 如果两个信号的频率轴不同，则进行插值使它们具有相同的频率轴
if ~isequal(freq1, freq2)
    freq_interp = min(freq1(1), freq2(1)):min(freq1(end), freq2(end));
    pxx1_interp = interp1(freq1, pxx1, freq_interp, 'linear', 'extrap');
    pxx2_interp = interp1(freq2, pxx2, freq_interp, 'linear', 'extrap');
else
    freq_interp = freq1;
    pxx1_interp = pxx1;
    pxx2_interp = pxx2;
end

% 计算频谱误差
error = (pxx1_interp - pxx2_interp).^2;
% figure
% plot(freq_interp,20*log10(error))
error = sum(error)/sum(pxx2_interp.^2);
end
function PCld = linearPCld_interp(PCld1,PCld2,k)
% 进行点云图之间的空间线性插值
L1 = length(PCld1);
L2 = length(PCld2);
[~,order] = min([L1,L2]);
if order == 1
    PCld = (1-k)*PCld1 + k*PCld2(1:L1,:);
elseif order == 2
    PCld = (1-k)*PCld1(1:L2,:) + k*PCld2;
end

end

function h = linearTime_interp(h1,h2,k)
% 从时域进行线性插值，未使用
L1 = length(h1);
L2 = length(h2);
[~,order] = min([L1,L2]);
if order == 1
    h = abs((1-k)*h1 + k*h2(1:L1));
elseif order == 2
    h = abs((1-k)*h1(1:L2) + k*h2);
end

end

function rel_error = calculate_absolute_error(signal1, signal2)
% 计算两个信号的归一化时间平方误差
% signal1 和 signal2 分别是两个信号的向量

% 确保两个信号长度相同
if length(signal1) ~= length(signal2)
    error('信号长度不一致');
end

% 计算平方误差
square_diff = (signal1 - signal2).^2;
rel_error = sum(square_diff)/sum(signal2.^2);
end

function plot_error_curves(Error_list1, Error_list2, Error_list3, Error_list4, title1, title2, title3, title4, K, epoch_num)
% 进行大量绘图，Plot error curves for four error matrices with titles

x = repmat(K, epoch_num, 1);
% Plot first error matrix
figure;
plot(x', Error_list1(1:epoch_num,:)', 'Color','b','LineWidth',0.1);
hold on
plot(x', Error_list1(epoch_num+1: epoch_num*2,:)', 'Color','r','LineWidth',0.1);
hold on
plot(x', Error_list1(epoch_num*2+1: epoch_num*3,:)', 'Color','g','LineWidth',0.1);
title(title1);
xlabel('Sample');
ylabel('Error');
fprintf('Sum of %s: %f\n', title1, sum(Error_list1, 2));
hold off

% Plot second error matrix
figure;
plot(x', Error_list2(1:epoch_num,:)', 'Color','b','LineWidth',0.1);
hold on
plot(x', Error_list2(epoch_num+1: epoch_num*2,:)', 'Color','r','LineWidth',0.1);
hold on
plot(x', Error_list2(epoch_num*2+1: epoch_num*3,:)', 'Color','g','LineWidth',0.1);
title(title2);
xlabel('Sample');
ylabel('Error');
fprintf('Sum of %s: %f\n', title2, sum(Error_list2, 2));
hold off

% Plot third error matrix
figure;
plot(x', Error_list3(1:epoch_num,:)', 'Color','b','LineWidth',0.1);
hold on
plot(x', Error_list3(epoch_num+1: epoch_num*2,:)', 'Color','r','LineWidth',0.1);
hold on
plot(x', Error_list3(epoch_num*2+1: epoch_num*3,:)', 'Color','g','LineWidth',0.1);
title(title3);
xlabel('Sample');
ylabel('Error');
fprintf('Sum of %s: %f\n', title3, sum(Error_list3, 2));
hold off

% Plot fourth error matrix
figure;
plot(x', Error_list4(1:epoch_num,:)', 'Color','b','LineWidth',0.1);legend('POT');
hold on
plot(x', Error_list4(epoch_num+1: epoch_num*2,:)', 'Color','r','LineWidth',0.1);legend('Spatial Linear');
hold on
plot(x', Error_list4(epoch_num*2+1: epoch_num*3,:)', 'Color','g','LineWidth',0.1);legend('Simple Linear')
title(title4);
xlabel('Sample');
ylabel('Error');
fprintf('Sum of %s: %f\n', title4, sum(Error_list4, 2));
end
