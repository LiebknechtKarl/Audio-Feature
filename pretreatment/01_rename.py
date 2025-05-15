import os

def rename_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)

        # 跳过文件夹，只处理文件
        if not os.path.isfile(old_path):
            continue

        # 拆分文件名为主名和扩展名
        parts = filename.split('.')
        if len(parts) < 2:
            continue  # 跳过没有扩展名的文件

        # 除了最后一部分，其他部分用 _ 连接
        new_name = '_'.join(parts[:-1]) + '.' + parts[-1]
        new_path = os.path.join(folder_path, new_name)

        # 重命名文件
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")


if __name__ == "__main__":

    # 替换为你的文件夹路径
    folder = "CMLRdataset/audio"
    rename_files_in_folder(folder)


    folder = "CMLRdataset/video"
    rename_files_in_folder(folder)