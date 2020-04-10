import os

def deleteBigFiles(max_size):
    file_list = [f for f in os.listdir() if os.path.isfile(f)]
    for f in file_list:
        filesize = os.stat(f).st_size
        if filesize > max_size:
            os.remove(f)
            # print(f'filename: {f} \n file size: {filesize} \n current working directory: {os.getcwd()} \n\n')
    directory_list = [d for d in os.listdir() if os.path.isdir(d)]
    for d in directory_list:
        os.chdir(d)
        deleteBigFiles(max_size)
        os.chdir("..")

def deleteBigFilesFor1000experiment():
    max_size = 10000000
    os.chdir(os.getcwd())
    os.chdir("results/logs")
    for folder in [folder for folder in os.listdir()]:
        os.chdir(folder)
        deleteBigFiles(max_size)
        os.chdir("..")
