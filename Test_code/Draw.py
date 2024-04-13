import os
import numpy as np
import matplotlib.pyplot as plt

basepath = "../Weight_models"
save_basepath = "../LossPic"
data_list = ['Canny', 'Dilate', 'Noise']
file_list = ['loss1_history.npy', 'loss2_history.npy', 'loss_criticA_history.npy', 'loss_criticB_history.npy']
code_list = ['QRGAN', 'wcyclegan', 'ONE2ONE2', 'REVgan']
numbers = ['0', '1', '2','3', '4','5', '6','7']

ifcritic = False
# ifcritic = True
code_index = 0
code = code_list[code_index]

    
for data in data_list:
    for number in numbers:
        if ifcritic:
            inter = 'critic_'
            label_index = 2
            ylim = (-0.5, 1)
            if code_index == 0:
                ylim = (-20, 10)
            elif code_index == 1:
                ylim = (-100, 100)
            elif code_index == 2:
                ylim = (-30, 30)
        else:
            inter = ''
            label_index = 0
            ylim = (-1, 15)
            if code_index == 0:
                ylim = (-200, 100)
            elif code_index == 1:
                ylim = (-40, 50)
            elif code_index == 2:
                ylim = (-10, 30)
            
        data1 = np.load(os.path.join(basepath, code, data+number, 'Loss', file_list[label_index]))
        data2 = np.load(os.path.join(basepath, code, data+number, 'Loss', file_list[label_index+1]))

        # 截取相同长度
        data1 = data1[:1000]
        data2 = data2[:1000]
        
        plt.plot(data1[::1], label=file_list[label_index][:-4])
        plt.plot(data2[::1], label=file_list[label_index+1][:-4])

        # 添加标签和标题
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(code+'_'+inter+data+number)

        # 设置纵坐标范围
        plt.ylim(ylim)
        plt.xlim(0, 800)

        # 添加图例
        plt.legend()

        # 保存图片
        plt.savefig(os.path.join(save_basepath, code, inter+data+number+'.png'))

        # 显示图形（可选）
        # plt.show()
        
        plt.clf()
