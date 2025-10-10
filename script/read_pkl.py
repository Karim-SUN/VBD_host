import pickle

# 定义读取pkl文件的函数
def read_pkl_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

# 文件路径
# pkl_file_path_1 = '/home/huang/Mount_Disk/shy/VBD/script/testing_results/test_diffusion/0/dd944f718adce5ce_sim_new.pkl'
pkl_file_path_2 = '/home/huang/Mount_Disk/shy/VBD/script/testing_results/test_diffusion/0/dd944f718adce5ce_sim.pkl'


# 读取pkl文件
# data_1 = read_pkl_file(pkl_file_path_1)
data_2 = read_pkl_file(pkl_file_path_2)


# 打印读取的数据
if data_2 is not None:
    print(data_2)