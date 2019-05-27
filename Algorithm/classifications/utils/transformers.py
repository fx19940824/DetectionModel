from torchvision import transforms as T
from torch import Tensor

def default_transformer(input_size):
    data_transforms = {
        'train': T.Compose([
            T.Resize(int(input_size * 1.1)),
            # transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomCrop(input_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # T.Lambda(lambda x:x.half()),
        ]),
        'val': T.Compose([
            T.Resize(int(input_size * 1.1)),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # T.Lambda(lambda x:x.half()),
        ]),
    }

    return data_transforms


def bomo_transformer(input_size):
    data_transforms = {
        'train': T.Compose([
            T.Resize((int(input_size*1.05), int(input_size*1.05))),
            T.ColorJitter(0.1, 0.1, 0.1, 0.05),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomCrop(input_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    return data_transforms
