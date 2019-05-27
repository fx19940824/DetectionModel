from torchvision import transforms as T


def build_transform_yolo(obj, scale):
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(scale),
            T.ToTensor(),
            T.Lambda(lambda x: x[[2, 0, 1]]),
            T.Lambda(lambda x: x.float().unsqueeze(0))
        ]
    )
    return transform


def build_transform_maskrcnn(obj, scale):
    if obj._cfg.INPUT.TO_BGR255:
        to_bgr_transform = T.Lambda(lambda x: x * 255)
    else:
        to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

    normalize_transform = T.Normalize(
        mean=obj._cfg.INPUT.PIXEL_MEAN, std=obj._cfg.INPUT.PIXEL_STD
    )

    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(scale),
            T.ToTensor(),
            to_bgr_transform,
            normalize_transform,
        ]
    )
    return transform


def build_transform_ganomaly(obj):
    opt = obj.args
    mean = opt.mean if opt.__contains__("mean") else 0.5
    std = opt.std if opt.__contains__("std") else 0.5
    if not isinstance(mean, tuple):
        mean = (mean,)
    if not isinstance(std, tuple):
        std = (std,)
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(opt.img_size),
            T.CenterCrop(opt.img_size),
            T.ToTensor(),
            T.Normalize(mean, std),
            T.Lambda(lambda x: x.float().unsqueeze(0))
        ]
    )
    return transform


def build_transform_cls(obj):
    opt = obj.args
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(opt.img_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.Lambda(lambda x: x.float().unsqueeze(0))
        # T.Lambda(lambda x:x.half()),
    ])
    return transform


def fn(x):
    test = x.numpy()
    return x


def build_transform_seg(obj):
    # opt = obj.args
    transform = T.Compose([
        T.ToPILImage(),
        # T.Resize(opt.img_size),
        T.ToTensor(),
        T.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        # T.Lambda(fn),
        T.Lambda(lambda x: x.float().unsqueeze(0))
    ])
    return transform