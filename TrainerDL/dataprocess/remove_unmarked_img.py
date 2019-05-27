import os, shutil
import glob


file_list = glob.glob("/media/cobot/30b0f4a0-3376-4f8f-b458-9c6857504361/Dataset/TUBE/*.png")

dst_dir = '/media/cobot/30b0f4a0-3376-4f8f-b458-9c6857504361/Dataset/tube-test/'

for file in file_list:
    jsonname = file.replace('.png', '.json')
    if not os.path.exists(jsonname):
        shutil.move(file, dst_dir)