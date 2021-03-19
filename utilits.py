import os 

def getAllName(dir):
    fileList = list(map(lambda fileName: os.path.join(dir, fileName), os.listdir(dir)))
    return fileList
