import os

def find_missing_files(directory):
    # 定义文件名格式
    file_format = "cc12m-train-{:04d}.tar"
    # 定义文件编号范围
    start_num = 0
    end_num = 2175

    # 获取文件夹中的所有文件名
    existing_files = set(os.listdir(directory))

    # 初始化缺失文件列表
    missing_files = []

    # 检查每个编号的文件是否存在
    for i in range(start_num, end_num + 1):
        file_name = file_format.format(i)
        if file_name not in existing_files:
            missing_files.append(file_name)

    return missing_files

# 使用示例
directory_path = "/home/mobile/cc12m"  # 替换为你的文件夹路径
missing_files = find_missing_files(directory_path)

if missing_files:
    print("缺失的文件有：")
    for file in missing_files:
        print(file)
else:
    print("没有缺失的文件。")