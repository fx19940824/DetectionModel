import torch
import cv2
import numpy as np

def keepOneMask(top_predictions):
    masks = top_predictions.get_field("mask").numpy()
    for i in range(len(masks)):
        thresh = masks[i][0, :, :, None]
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 1:
            continue
        contour = contours[0]
        for ct in contours:
            contour = ct if len(ct) > len(contour) else contour
        width,height,channel=thresh.shape
        mask = np.zeros((width,height),dtype="uint8")
        cv2.fillPoly(mask,[contour],1)
        masks[i][0] = mask
    top_predictions.add_field("mask", torch.from_numpy(masks))
    return top_predictions

def removeSharedBox(original_size, top_predictions, threshold_sharedRatio=0.4):
    height, width = original_size
    boxes = np.zeros((len(top_predictions.bbox), 6), dtype=int)
    for idx, box in enumerate(top_predictions.bbox):
        boxes[idx] = (idx, box[0], box[1], box[2], box[3], (box[3] - box[1]) * (box[2] - box[0]))
    # boxes = boxes[boxes[:, 5].argsort()]
    visited = np.zeros((height, width), dtype=np.bool)
    indx = []
    for box in boxes:
        acculate = np.sum(visited[box[2]:box[4], box[1]:box[3]])
        if acculate < box[5] * threshold_sharedRatio:
            indx.append(box[0])
            visited[box[2]:box[4], box[1]:box[3]] = 1

    return top_predictions[indx]

def removeOutlier(top_predictions):
    masks = top_predictions.get_field("mask").numpy()

    boxes = np.zeros((len(top_predictions.bbox), 3), dtype=float)
    for idx, mask in enumerate(masks):
        thresh = mask[0, :, :, None]
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        points = []
        for pt in contours[0]:
            points.append([pt[0][0], pt[0][1]])

        rect = cv2.minAreaRect(np.mat(points))
        boxes[idx] = (idx, max(rect[1][0], rect[1][1]), min(rect[1][0], rect[1][1]))

    (mean_w, std_w) = cv2.meanStdDev(boxes[:, 1])
    (mean_h, std_h) = cv2.meanStdDev(boxes[:, 2])
    l1 = mean_w - 1.5 * std_w
    r1 = mean_w + 1.5 * std_w
    l2 = mean_h - 1.5 * std_h
    r2 = mean_h + 1.5 * std_h

    index = []
    for box in boxes:
        if box[1] > l1 and box[1] < r1 and box[2] > l2 and box[2] < r2:
            index.append(box[0])

    return top_predictions[index] if len(index) != 0 else top_predictions

def removeLowRectangleRatio(top_prediction, threshold_rectangleRatio):

    masks = top_prediction.get_field("mask").numpy()

    indx = []
    for i in range(len(masks)):
        thresh = masks[i][0, :, :, None]
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        points = []
        for pt in contours[0]:
            points.append([pt[0][0], pt[0][1]])

        rect = cv2.minAreaRect(np.mat(points))
        if masks[i].sum() / (rect[1][0] * rect[1][1]) > threshold_rectangleRatio:
            indx.append(i)
    return top_prediction[indx]

def reSortPrediction(predictions):
    if len(predictions):
        # 得到label的tensor
        labels = predictions.get_field("labels")
        # 消除重复label，并排序，最后转成list，实际上得到class列表
        label_list = torch.unique(labels, sorted=True).tolist()

        res_list = []
        for label in label_list:
            # 按class优先级排序，eq得到该等于该label的元素掩码，nonzero得到掩码索引
            res = torch.nonzero(torch.eq(labels, label))
            # squeeze去除多余维度
            res = torch.reshape(res, (-1,))
            # 将索引tensor加入列表
            if res.size():
                res_list.append(res)

        # 将各label的索引连接起来，最终得到先按label排序，每个label中再按置信度排序的顺序
        res_tensor = torch.cat(res_list, dim=0)

        return predictions[res_tensor]
    else:
        return predictions

def build_postprocess_ganomaly(obj):
    def postprocess_ganomaly(result):
        thresh = obj.args.threshold
        m_score = obj.args.max_score
        M_score = obj.args.min_score
        latent_i, latent_o = result[1:]
        error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1)
        an_score = error.reshape(error.size(0)).squeeze()
        an_score = (an_score - m_score) / (M_score - m_score)
        return an_score > thresh

    return postprocess_ganomaly

def build_generalized_postprocess(obj):
    def generalized_postprocess(prediction, original_size, masker=None):
        height, width = original_size
        prediction = prediction.resize((width, height))

        segms = None
        if masker is None:
            return prediction, segms

        if prediction.has_field("mask"):
            if isinstance(prediction.get_field("mask"), torch.Tensor):
                masks = prediction.get_field("mask")
                masks = masker([masks], [prediction])[0]
                prediction.add_field("mask", masks)
                segms = torch.zeros((height, width), dtype=masks.dtype)
                # scores = prediction.get_field("scores")
                # idx = torch.argmax(scores)
                for i, mask in enumerate(masks):
                    mask = mask.squeeze()
                    tempmask = mask * segms
                    mask[tempmask != 0] = 0
                    segms += (i + 1) * mask
        return prediction, segms
    return generalized_postprocess

def build_postprocess_plgdetection(obj):
    def postprocess_plgdetection(prediction, original_size, masker=None):
        segms = None
        if masker is None:
            return prediction, segms

        prediction = reSortPrediction(prediction)

        #移除重叠区域的阈值，当重叠区域大于该阈值时会移除将要叠加上去的检测框
        threshold_sharedRatio = 0.5

        #矩形度阈值，当检测矩形时若检测结果的矩形度低于该阈值将移除
        threshold_rectangleRatio = 0.5

        height, width = original_size
        prediction = prediction.resize((300, 300))

        if prediction.has_field("mask"):
            if isinstance(prediction.get_field("mask"), torch.Tensor):
                masks = prediction.get_field("mask")
                masks = masker([masks], [prediction])[0]
                prediction.add_field("mask", masks)

                # 若存在多个连通域，则只保留最大的一个
                prediction = keepOneMask(prediction)

                # 移除重叠部分
                prediction = removeSharedBox(original_size, prediction, threshold_sharedRatio)

                # 根据矩形度阈值去除非矩形提取
                prediction = removeLowRectangleRatio(prediction, threshold_rectangleRatio)

                masks = prediction.get_field("mask")
                segms = torch.zeros(masks.shape[2:], dtype=masks.dtype)
                for i, mask in enumerate(masks):
                    mask = mask.squeeze()
                    tempmask = mask * segms
                    mask[tempmask != 0] = 0
                    segms += (i + 1) * mask
        return prediction, segms

    return postprocess_plgdetection

def build_postprocess_bagdetection(obj):
    def postprocess_daizidetection(prediction, original_size, masker=None):
        segms = None
        if masker is None:
            return prediction, segms

        #移除重叠区域的阈值，当重叠区域大于该阈值时会移除将要叠加上去的检测框
        threshold_sharedRatio = 0.4
        #使用离群点检测的阈值，当检测数量大于该阈值时使用离群点检测
        threshold_removeOutlier = 3
        #矩形度阈值，当检测矩形时若检测结果的矩形度低于该阈值将移除
        threshold_rectangleRatio = 0.8

        # 当挑出多于15个候选框时去除多余候选框
        if len(prediction) > 15:
            prediction = prediction[:15]

        height, width = original_size
        prediction = prediction.resize((300, 300))

        if prediction.has_field("mask"):
            if isinstance(prediction.get_field("mask"), torch.Tensor):
                masks = prediction.get_field("mask")
                masks = masker([masks], [prediction])[0]
                prediction.add_field("mask", masks)

                # 若存在多个连通域，则只保留最大的一个
                prediction = keepOneMask(prediction)

                # 离群点检测
                if len(prediction) > threshold_removeOutlier:
                    prediction = removeOutlier(prediction)

                # 当挑出多于8个候选框时去除多于候选框
                # if len(prediction) > 7:
                #     prediction = prediction[:7]

                # 移除重叠部分
                prediction = removeSharedBox(original_size, prediction, threshold_sharedRatio)

                # 根据矩形度阈值去除非矩形提取
                prediction = removeLowRectangleRatio(prediction, threshold_rectangleRatio)

                masks = prediction.get_field("mask")
                segms = torch.zeros(masks.shape[2:], dtype=masks.dtype)
                for i, mask in enumerate(masks):
                    mask = mask.squeeze()
                    tempmask = mask * segms
                    mask[tempmask != 0] = 0
                    segms += (i + 1) * mask

        return prediction, segms

    return postprocess_daizidetection

def build_postprocess_cls(obj):
    def postprocess_cls(result):
        cls = torch.argmax(result)
        return cls
    return postprocess_cls


def build_postprocess_seg(obj):
    def postprocess_seg(prediction, original_size):
        # kernel = np.ones((5, 5), dtype=np.uint8)
        prediction = prediction.data.cpu().numpy().squeeze()
        prediction = np.argmax(prediction, axis=0).astype(np.uint8)
        prediction = cv2.resize(prediction, original_size, cv2.INTER_NEAREST)
        prediction = cv2.medianBlur(prediction, 9)
        # prediction *= 255
        # cv2.imshow("test", prediction)
        # cv2.waitKey(0)
        return prediction
    return postprocess_seg