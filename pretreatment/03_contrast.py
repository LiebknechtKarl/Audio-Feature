import os

def find_matching_files(folder1, folder2):
    # 获取第一个文件夹中的所有文件名（不含后缀）
    files1 = {os.path.splitext(file)[0] for file in os.listdir(folder1) if file.endswith('.wav')}
    
    # 获取第二个文件夹中的所有文件名（不含后缀）
    files2 = {os.path.splitext(file)[0] for file in os.listdir(folder2) if file.endswith('.mp4')}
    
    # 找出两个文件夹中都存在的文件名
    matching_files = files1.intersection(files2)
    
    return matching_files

if __name__ == "__main__":
    # 替换为你的两个文件夹路径
    # folder_wav = '/path/to/wav/folder'
    # folder_mp4 = '/path/to/mp4/folder'

    folder_wav = 'CMLRdataset/audio'
    folder_mp4 = 'CMLRdataset/video'


    matching_files = find_matching_files(folder_wav, folder_mp4)
    
    if matching_files:
        print("以下文件名在两个文件夹中一一对应:")
        for file_name in matching_files:
            print(file_name)
    else:
        print("没有找到在两个文件夹中一一对应的文件名")