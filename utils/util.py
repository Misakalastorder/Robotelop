import os

def create_folder(folder):
	if not os.path.exists(folder):  # 判断path对应文件或目录是否存在，返回True或False
		os.makedirs(folder)
