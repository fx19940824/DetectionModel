from __future__ import print_function
from __future__ import division

from classifications.utils.model_factory import *
import torch.onnx


def startTesting(model_name, num_classes, param_path, data_transforms, img_path, output_dir):
    model_ft = initialize_model(model_name, num_classes, False, use_pretrained=False)
    print(model_ft)

    model_ft.load_state_dict(torch.load(param_path))
    image_datasets = datasets.ImageFolder(img_path, data_transforms)
    # Create training and validation dataloaders
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=10, shuffle=False, num_workers=1)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    model_ft.eval()

    running_corrects = 0
    i = 0
    # Iterate over data.
    for inputs, labels in dataloaders:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)

        # statistics
        running_corrects += torch.sum(preds == labels.data)

    epoch_acc = running_corrects.double() / len(dataloaders.dataset)
    print('Test path {}: totalNum:{} Acc: {:.4f}'.format(img_path, len(dataloaders.dataset), epoch_acc))

def startTestOneImage(model_name, num_classes, param_path, data_transforms, img_path):
    model_ft = initialize_model(model_name, num_classes, True, use_pretrained=False)

    model_ft.load_state_dict(torch.load(param_path))

    input = data_transforms(Image.open(img_path))
    input = input.view(-1, input.size()[0], input.size()[1], input.size()[2])

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input = input.to(device)

    model_ft = model_ft.to(device)
    model_ft.eval()
    t = time.time()
    # with torch.no_grad():
    outputs = model_ft(input)
    print("test time: {}".format((time.time()-t)*1000))
    # print(outputs)
    return outputs

def startTestingFromListFile(model_name, num_classes, param_path, data_transforms, img_path, output_dir):
    if output_dir:
        if os.path.exists(output_dir):
            print('the output dir is already exist')
            return
        else:
            os.makedirs(output_dir)

    model_ft = initialize_model(model_name, num_classes, False, use_pretrained=False)
    print(model_ft)

    model_ft.load_state_dict(torch.load(param_path))

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    model_ft.eval()

    fopen = open(img_path, 'r')
    lines = fopen.readlines()
    running_corrects = 0
    # Iterate over data.
    for line in lines:
        line = line.split(' ')
        input = data_transforms(Image.open(line[0]))
        input = input.view(-1, input.size()[0], input.size()[1], input.size()[2])
        print(input.size())
        input = input.to(device)
        label = int(line[1])

        outputs = model_ft(input)
        softmax = F.softmax(outputs[0])


        # _, preds = torch.max(outputs, 1)
        # pred = preds.item()
        # success = (pred == label)

        if softmax[label].item() > 0.9:
            success = True
        else:
            success = False

        if success:
            running_corrects += 1
        else:
            if output_dir:
                srcfile = line[0]
                fpath, fname = os.path.split(srcfile)
                f, b = os.path.splitext(fname)
                dstfile = output_dir+'/'+f+'_'+str(label)+'_'+str(softmax[label].item())+b;
                print(dstfile)

                shutil.copyfile(srcfile, dstfile)


    epoch_acc = float(running_corrects) / len(lines)

    print('Test path {}: totalNum:{} Acc: {:.4f}'.format(img_path, len(lines), epoch_acc))

def startTestingFromFloder(model_name, num_classes, param_path, data_transforms, img_path, output_dir):
    if output_dir:
        if os.path.exists(output_dir):
            print('the output dir is already exist')
            return
        else:
            os.makedirs(output_dir)

    model_ft = initialize_model(model_name, num_classes, False, use_pretrained=False)
    print(model_ft)

    model_ft.load_state_dict(torch.load(param_path))

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    model_ft.eval()

    running_corrects = 0
    lens = 0;
    for dirpath, dirnames, filenames in os.walk(img_path):
        lens = len(filenames)
        if lens == 0:
            return

        for file in filenames:
            img_path = dirpath + "/" + file;
            img = Image.open(img_path)
            width, height = img.size;
            w = width / 3
            margin = w / 6

            box_list = [(0, 0, w + margin, w + margin), (w - margin/2, 0, 2*w + margin/2, w + margin), (2*w - margin, 0, width, w + margin),
                        (0, w - margin/2, w+margin, 2*w + margin/2), (w - margin/2, w - margin/2, 2*w + margin/2, 2*w + margin/2),
                        (2*w - margin, w - margin/2, width, 2*w+margin/2), (0, 2*w - margin, w + margin, width),
                        (w - margin/2, 2*w - margin, 2*w + margin/2, width), (2*w - margin, 2*w -margin, width, width)]

            image_list = [img.crop(box) for box in box_list]
            image_list.append(img)

            success = False
            for i in image_list:
                input = data_transforms(i)
                input = input.view(-1, input.size()[0], input.size()[1], input.size()[2])
                input = input.to(device)

                outputs = model_ft(input)
                softmax = F.softmax(outputs[0])

                prob = softmax[0].item()
                if prob > 0.5:
                    success = True
                    # i.save(output_dir+'/'+ str(prob) + "_" +file)
                # else:
                #     success = False

            if success:
                running_corrects += 1
            else:
                if output_dir:
                    shutil.copyfile(img_path, output_dir+'/'+file)

    epoch_acc = float(running_corrects) / lens

    print('totalNum:{} corrent:{} Acc: {:.4f}'.format(lens, running_corrects, epoch_acc))

def showOneImageTransform(img_path, data_transforms):
    img = Image.open(img_path)
    img_transform = data_transforms(img)
    plt.figure("data tranform")
    plt.imshow(img_transform)
    plt.show()

def showImageListTransform(imglist_path, data_transforms, output_dir):
    if os.path.exists(output_dir):
        print('the output dir is already exist')
        return

    os.makedirs(output_dir)
    fopen = open(imglist_path, 'r')
    lines = fopen.readlines()
    # Iterate over data.
    for line in lines:
        line = line.split(' ')
        srcfile = line[0]
        input = data_transforms(Image.open(srcfile))
        fpath, fname = os.path.split(srcfile)
        dstfile = output_dir+'/'+fname
        print(dstfile)
        input.save(dstfile)

def traceModelJit(src_model_name, src_num_classes, src_param_path, dst_model_name, input_rows=224, input_cols=224, input_channels=3, input_batch=1):
    model_ft = initialize_model(src_model_name, src_num_classes, False, use_pretrained=False)
    model_ft.load_state_dict(torch.load(src_param_path))

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    model_ft.eval()

    # An example input you would normally provide to your model's forward() method.
    example = torch.rand(input_batch, input_channels, input_rows, input_cols)
    example = example.to(device)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model_ft, example)
    traced_script_module.save(dst_model_name)


def traceModelONNX(src_model_name, src_num_classes, src_param_path, dst_model_name, input_rows=224, input_cols=224, input_channels=3, input_batch=1):
    model_ft = initialize_model(src_model_name, src_num_classes, False, use_pretrained=False)
    model_ft.load_state_dict(torch.load(src_param_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    model_ft.eval()

    # An example input you would normally provide to your model's forward() method.
    example = torch.randn(input_batch, input_channels, input_rows, input_cols, requires_grad=True)
    example = example.to(device)

    torch.onnx._export(model_ft,  # model being run
                       example,  # model input (or a tuple for multiple inputs)
                       dst_model_name, # where to save the model (can be a file or file-like object)
                       export_params=True)
