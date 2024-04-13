import os
import numpy as np
import matplotlib.pyplot as plt

basepath = "./Weight_models"
save_basepath = "./LossPic2"
# data_list = ['Canny', 'Dilate', 'Noise']
data_list = ['Dilate']

file_list = ['loss1_history.npy', 'loss2_history.npy', 'loss_criticA_history.npy', 'loss_criticB_history.npy']
# code_list = ['QRGAN', 'wcyclegan', 'ONE2ONE2', 'REVgan']
code_list = ['QRGAN']
# code_list = ['QRGAN', 'wcyclegan', 'ONE2ONE2', 'REVgan']
numbers = ['1']

ifcritic = False
# ifcritic = True
code_index = 0
code = code_list[code_index]


for data in data_list:
    for number in numbers:
        data1 = np.load(os.path.join(basepath, code, data+number, 'Loss', file_list[0]))
        data2 = np.load(os.path.join(basepath, code, data+number, 'Loss', file_list[1]))
        data3 = np.load(os.path.join(basepath, code, data+number, 'Loss', file_list[2]))
        data4 = np.load(os.path.join(basepath, code, data+number, 'Loss', file_list[3]))
        
        # 截取相同长度
        data1 = data1[:1000]
        data2 = data2[:1000]
        data3 = data3[:1000]
        data4 = data4[:1000]
        
        # 间隔采样
        sample_interval = 10
        data1 = data1[::sample_interval]
        data2 = data2[::sample_interval]
        data3 = data3[::sample_interval]
        data4 = data4[::sample_interval]
        
        # 创建画布和子图
        fig, ax1 = plt.subplots()
        # 设置全局字体大小
        plt.rcParams.update({'font.size': 14})  # 设置字体大小为14
        # 第一个y轴，与data1和data2绑定
        color = 'tab:blue'
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('Generator_loss', color='black')
        ax1.plot(np.arange(len(data1)) * sample_interval, data1, color=color, label="G")
        ax1.plot(np.arange(len(data2)) * sample_interval, data2, linestyle='dashed', color=color, label="F")
        ax1.tick_params(axis='y', labelcolor='black')

        # 第二个y轴，与data3和data4绑定
        ax2 = ax1.twinx()
        color = 'tab:purple'
        ax2.set_ylabel('Critic_loss', color='black')
        ax2.plot(np.arange(len(data3)) * sample_interval, data3, color=color, label="D1")
        ax2.plot(np.arange(len(data4)) * sample_interval, data4, linestyle='dashed', color=color, label="D2")
        ax2.tick_params(axis='y', labelcolor='black')

        # # 添加标签和标题
        # plt.title(data+number)
        # plt.title('Digit'+number,x=0.5,y=-0.4) #

        # 添加图例
        fig.tight_layout()
        # fig.legend(fontsize='xx-small', loc='upper right',borderaxespad=2)
        fig.legend(loc=(0.66, 0.69))
        # plt.legend(bbox_to_anchor=(1.13,1.25))#显示图例

        # 保存图片
        save_dir = os.path.join(save_basepath, code)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_basepath, code, data+number+'.png'))
        
        # 清除画布
        plt.close(fig)
