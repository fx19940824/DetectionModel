import glob
import json
from tqdm import tqdm
import os
def main(dir_source):
    if dir_source[-1]!='/':
        dir_source += '/'
    imglist = []
    for index, filename in enumerate(glob.glob(dir_source + '*.png')):
        imgname = filename.split('/')[-1]
        imgname = imgname.split('.png')[0]
        filepath = filename.split('.png')[0]
        imglist.append((filepath, imgname))

    num = 0
    total = len(imglist)
    with tqdm(total=len(imglist)) as bar:
        for filepath, imgname in imglist:
            try:
                f = open(filepath + ".json", encoding='utf-8')
                myjson = json.load(f)
                for a in myjson['shapes']:
                    if a['label']!='a' and a['label']!='b' and a['label']!='c':
                        print(filepath)
                        break
                f.close()

                # os.remove(filepath + '.json')
                # os.remove(filepath + '.png')
                num += 1

                bar.update(1)
            except Exception as e:
                print(filepath)

    print("finished")

if __name__=='__main__':
    dir_source = '/media/cobot/30b0f4a0-3376-4f8f-b458-9c6857504361/Dataset/TUBE'

    main(dir_source)
