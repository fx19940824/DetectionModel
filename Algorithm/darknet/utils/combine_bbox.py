import os
from Utils.parsers import parse_cfg

def parse_box_file(input_path, width, height):

    box_name = open(input_path).readlines()
    name_list = []

    for line in box_name:
        data = line.split(' ')
        class_id = data[0]
        x = float(data[1]) * width
        y = float(data[2]) * height
        w = float(data[3]) * width
        h = float(data[4]) * height
        x_min = int(x - w/2)
        y_min = int(y - h/2)
        x_max = int(x + w/2)
        y_max = int(y + h/2)
        bbox = '{},{},{},{},{}'.format(x_min, y_min, x_max, y_max, class_id)
        name_list.append(bbox)

    return name_list


def parse_from_config(cfgfile):
    # train_yolo.cfg
    cfgs = parse_cfg(cfgfile)
    n_cfg = parse_cfg(cfgs["cfgpath"])
    width, height = n_cfg["width"], n_cfg["height"]
    file_list = []
    for file in os.listdir(cfgs["train_dir"]):

        name, suffix = os.path.splitext(file)
        if suffix == '.txt' and name != "classes":
            file_path = os.path.join(cfgs["train_dir"], file)
            img_path = os.path.join(cfgs["train_dir"], name + suffix)
            bbox = parse_box_file(file_path, width, height)
            file_list.append(img_path + ' ' + ' '.join(bbox))

    return file_list
