import os
import wcyclegan_test
import QRGAN_test
import ONE2ONE2_test
import JiaoYannoise
import RMSE
# import RevGAN.models.RevGAN2_test


data_basepath = "./Data"
weight_basepath = "./Weight_models"
result_basepath = "./TestResult"
denoise_basepath = "./TestDenoise"
metric_basepath = "./Metric_Result"

code_list = ['QRGAN', 'wcyclegan', 'ONE2ONE2', 'REVgan', 'Origin']
# data_list = ['Canny', 'Dilate', 'Noise']
data_list = ['Noise']
number_list = ['7']
code_index =0
code = code_list[code_index]


def Runtest(data, number, data_path, weight_path, result_path, mestric_path, denoise_path=None):
    
    real_path1 = os.path.join(result_path, 'result_B_real')
    real_path2 = os.path.join(result_path, 'result_A_real')
    metric_save_path1 = os.path.join(mestric_path, data+number+'_A2B.csv')
    metric_save_path2 = os.path.join(mestric_path, data+number+'_B2A.csv')
            
    if code_index == 0:
        # QRGAN_test.test(data_path, weight_path, result_path)
        
        # JiaoYannoise
        input_folder = os.path.join(result_path, "result_A2B")
        output_folder = os.path.join(denoise_path, "A2B_denoise")
        input_folder2 = os.path.join(result_path, "result_B2A")
        output_folder2 = os.path.join(denoise_path, "B2A_denoise")
        JiaoYannoise.process_and_save_images(input_folder, output_folder)
        JiaoYannoise.process_and_save_images(input_folder2, output_folder2)
        
        # RMSE 
        gen_path1 = os.path.join(denoise_path, 'A2B_denoise')
        gen_path2 = os.path.join(denoise_path, 'B2A_denoise')
        if data == 'Noise':
            RMSE.Metric_Calculate(real_path2,gen_path1, metric_save_path1)
        else:
            RMSE.Metric_Calculate(gen_path1, real_path1, metric_save_path1)
            RMSE.Metric_Calculate(gen_path2, real_path2, metric_save_path2)     
        
    elif code_index == 1:
        wcyclegan_test.test(data_path, weight_path, result_path)
        
        # RMSE
        gen_path1 = os.path.join(result_path, 'result_A2B')
        gen_path2 = os.path.join(result_path, 'result_B2A')
        if data == 'Noise':
            # RMSE.Metric_Calculate(gen_path1, real_path1, metric_save_path1)
            RMSE.Metric_Calculate(real_path2,gen_path1, metric_save_path1)
        else:
            RMSE.Metric_Calculate(gen_path1, real_path1, metric_save_path1)
            RMSE.Metric_Calculate(gen_path2, real_path2, metric_save_path2)
        
    elif code_index == 2:
        # ONE2ONE2_test.test(data_path, weight_path, result_path)
        
        # RMSE
        gen_path1 = os.path.join(result_path, 'result_A2B')
        gen_path2 = os.path.join(result_path, 'result_B2A')
        if data == 'Noise':
           RMSE.Metric_Calculate(real_path2,gen_path1, metric_save_path1)
        else:
            RMSE.Metric_Calculate(gen_path1, real_path1, metric_save_path1)
            RMSE.Metric_Calculate(gen_path2, real_path2, metric_save_path2)
        
    elif code_index == 3:
        # RevGAN2_test.UnpairedRevGANModel(data_path, weight_path, result_path)
        
        # RMSE
        gen_path1 = os.path.join(result_path, 'result_A2B')
        gen_path2 = os.path.join(result_path, 'result_B2A')
        if data == 'Noise':
            RMSE.Metric_Calculate(gen_path1, real_path1, metric_save_path1)
        else:
            RMSE.Metric_Calculate(gen_path1, real_path1, metric_save_path1)
            RMSE.Metric_Calculate(gen_path2, real_path2, metric_save_path2)
    
    elif code_index == 4:
        A = os.path.join(data_basepath, data, number, 'A')
        B = os.path.join(data_basepath, data, number, 'B')
        metric_save_path = os.path.join(mestric_path, data+number+'.csv')
        RMSE.Metric_Calculate(A, B, metric_save_path)
        

if __name__ == '__main__':
    print("code: "+code)
    for data in data_list:
        for number in number_list:
            #这段代码使用 os.path.join() 函数来组合多个路径部分，生成一个完整的文件路径。
            data_path = os.path.join(data_basepath, data, number)
            weight_path = os.path.join(weight_basepath, code, data+number)
            result_path = os.path.join(result_basepath, code+'_test', data, number)
            metric_path = os.path.join(metric_basepath, code+'_test')
            denoise_path = os.path.join(denoise_basepath, data, number)
            # print("\ndata path: "+data_path)
            # print("weight path: "+weight_path)
            # print("result path: "+result_path)
            Runtest(data, number, data_path, weight_path, result_path, metric_path, denoise_path)
    

# data_basepath = "../Data"
# weight_basepath = "../Weight_models"
# result_basepath = "../TestResult"
# denoise_basepath = "../TestDenoise"
# metric_basepath = "../Metric_Result"

# code_list = ['QRGAN', 'wcyclegan', 'ONE2ONE2', 'REVgan', 'Origin']
# data_list = ['Canny', 'Dilate', 'Noise']
# number_list = ['0', '1', '7']

# code_index = 2
# code = code_list[code_index]
