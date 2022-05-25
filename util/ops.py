import numpy as np
import torch
import random
import os
import matplotlib.cm as cmx
import cv2
import matplotlib as mpl
mpl.use('Agg')

from torch.autograd import Variable
from PIL import Image, ImageDraw, ImageFont
from skimage.transform import resize
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']

def np_kld(p,q):
    eps = 1e-7
    kld = (p * np.log((p + eps) / (q + eps))) + ((1 - p) * np.log((1 - p + eps) / (1 - q + eps)))

    return kld.mean()


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

def enable_eval_dropout(model):
    for module in model.modules():
        if 'Dropout' in type(module).__name__:
            module.train()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def computeAUROC(dataGT, dataPRED, classCount):
    outAUROC = []

    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()

    for i in range(classCount):
        outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))

    return outAUROC

def numpy_to_torch(img, device, requires_grad=True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output).to(device)

    output.unsqueeze_(0)
    v = Variable(output, requires_grad=requires_grad)
    return v

def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1 , :] - img[1 :, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[: , :-1] - img[: , 1 :])).pow(tv_beta))
    return row_grad + col_grad

def preprocess_image(img, device):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))


    preprocessed_img_tensor = torch.from_numpy(preprocessed_img).to(device)

    preprocessed_img_tensor.unsqueeze_(0)

    return Variable(preprocessed_img_tensor, requires_grad=False)

def box_matching(idx, target_idx, bbox):
    bnd_idx = idx[:,1]

    box_label = []
    matched_box = []

    for i in range(len(bnd_idx)):
        if bnd_idx[i].item() == target_idx.item():
            box_label.append(i)

    for j in box_label:
        matched_box.append(bbox[j].unsqueeze(0))

    return torch.cat(matched_box)

def assign_random_seed(v_seed):
    torch.manual_seed(v_seed)
    torch.cuda.manual_seed(v_seed)
    np.random.seed(v_seed)
    torch.random.manual_seed(v_seed)
    random.seed(v_seed)

def visualization(attr_map, root, bBox, pred_idx, idx):
    selectedFont = ImageFont.truetype(os.path.join('usr/share/fonts/', 'NanumGothic.ttf'), size=15)
    box_width = 3

    attr_min = np.min(attr_map)
    attr_max = np.max(attr_map)

    ori_img = cv2.imread(root)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

    h, w, c = ori_img.shape
    attr_map = resize(attr_map, (h, w), anti_aliasing=True)

    attr_map = np.uint8(cmx.jet((attr_map - attr_min) / (attr_max - attr_min)) * 255)[:, :, 0:3]

    blen_img = ori_img * 0.4 + attr_map * 0.6
    gcam = np.uint8(blen_img)
    gcam = Image.fromarray(gcam)

    bbox_draw = ImageDraw.Draw(gcam)
    for i in range(len(bBox)):
        bbox_draw.rectangle([(round(bBox[i][0].item()), round(bBox[i][1].item())),
                             (round((bBox[i][0] + bBox[i][2]).item()), round((bBox[i][1] + bBox[i][3]).item()))],
                            outline="red", width=box_width)

    bg_img = Image.new("RGB", (w+120, h+120), color=(255, 255, 255))
    bg_img.paste(gcam, ((120) // 2, (120) // 2))

    draw = ImageDraw.Draw(bg_img)

    draw.text((20, 30), 'Prediction  : ', fill='red', font=selectedFont)
    draw.text((20, 10), 'Ground Truth: ', fill='black', font=selectedFont)

    uniq_p_idx = torch.unique(pred_idx[:,1])
    for j in range(len(uniq_p_idx)):
        draw.text((130 * (j + 1) + 20, 30), object_categories[uniq_p_idx[j].item()], fill='red',
                  font=selectedFont)

    uniq_idx = torch.unique(idx[:, 1])
    for k in range(len(uniq_idx)):
        draw.text((130 * (k + 1) + 20, 10), object_categories[uniq_idx[k].item()], fill='black', font=selectedFont)

    del ori_img

    return bg_img


def encode_labels(target):
    """
    Encode multiple labels using 1/0 encoding

    Args:
        target: xml tree file
    Returns:
        torch tensor encoding labels as 1/0 vector
    """

    ls = target['annotation']['object']

    j = []
    if type(ls) == dict:
        if int(ls['difficult']) == 0:
            j.append(object_categories.index(ls['name']))

    else:
        for i in range(len(ls)):
            if int(ls[i]['difficult']) == 0:
                j.append(object_categories.index(ls[i]['name']))

    k = np.zeros(len(object_categories))
    k[j] = 1

    return torch.from_numpy(k)


def get_ap_score(y_true, y_scores):
    """
    Get average precision score between 2 1-d numpy arrays

    Args:
        y_true: batch of true labels
        y_scores: batch of confidence scores
=
    Returns:
        sum of batch average precision
    """
    scores = 0.0

    for i in range(y_true.shape[0]):
        scores += average_precision_score(y_true=y_true[i], y_score=y_scores[i])

    return scores

def bounded_kl(p, q):
    return (torch.log2(torch.abs(p-q) + 1)).mean()

def diff_log_odds(original, perturbed):
    log_ori = torch.log(original / (1-original + 1e-07))
    log_prt = torch.log(perturbed / (1-perturbed + 1e-07))

    return log_ori - log_prt