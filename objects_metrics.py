import numpy as np
from scipy.special import comb
from skimage import measure
import torch


def compute_ari(table):
    """
    Compute ari, given the index table
    :param table: (r, s)
    :return:
    """
    
    # (r,)
    a = table.sum(axis=1)
    # (s,)
    b = table.sum(axis=0)
    n = a.sum()
    
    comb_a = comb(a, 2).sum()
    comb_b = comb(b, 2).sum()
    comb_n = comb(n, 2)
    comb_table = comb(table, 2).sum()
    
    if (comb_b == comb_a == comb_n == comb_table):
        # the perfect case
        ari = 1.0
    else:
        ari = (
            (comb_table - comb_a * comb_b / comb_n) /
            (0.5 * (comb_a + comb_b) - (comb_a * comb_b) / comb_n)
        )
    
    return ari
    
    
def compute_mask_ari(mask0, mask1):
    """
    Given two sets of masks, compute ari
    :param mask0: ground truth mask, (N0, H, W)
    :param mask1: predicted mask, (N1, H, W)
    :return:
    """
    
    # will first need to compute a table of shape (N0, N1)
    # (N0, 1, H, W)
    mask0 = mask0[:, None].byte()
    # (1, N1, H, W)
    mask1 = mask1[None, :].byte()
    # (N0, N1, H, W)
    agree = mask0 & mask1
    # (N0, N1)
    table = agree.sum(dim=-1).sum(dim=-1)
    
    return compute_ari(table.numpy())


def _segmentation_cover(pred, gt):
    regionsGT = []
    regionsPred = []
    total_gt = 0

    cntR = 0
    sumR = 0
    cntP = 0
    sumP = 0

    propsGT = measure.regionprops(gt)
    for prop in propsGT:
        regionsGT.append(prop.area)
    regionsGT = np.array(regionsGT).reshape(-1, 1)
    total_gt = total_gt + np.max(gt)

    best_matchesGT = np.zeros((1, total_gt))

    matches = _match_segmentation(pred, gt)

    matchesPred = np.max(matches, axis=1).reshape(-1, 1)
    matchesGT = np.max(matches, axis=0).reshape(1, -1)

    propsPred = measure.regionprops(pred)
    for prop in propsPred:
        regionsPred.append(prop.area)
    regionsPred = np.array(regionsPred).reshape(-1, 1)

    for r in range(regionsPred.shape[0]):
        cntP += regionsPred[r] * matchesPred[r]
        sumP += regionsPred[r]

    for r in range(regionsGT.shape[0]):
        cntR += regionsGT[r] * matchesGT[:, r]
        sumR += regionsGT[r]

    best_matchesGT = np.maximum(best_matchesGT, matchesGT)

    R = cntR / (sumR + (sumR == 0))
    P = cntP / (sumP + (sumP == 0))

    return R[0], P[0]
def iou_binary(mask_A, mask_B):
    assert mask_A.shape == mask_B.shape
    assert mask_A.dtype == torch.bool
    assert mask_B.dtype == torch.bool
    intersection = (mask_A * mask_B).sum((1, 2, 3))
    union = (mask_A + mask_B).sum((1, 2, 3))
    # Return -100 if union is zero, else return IOU
    return torch.where(union == 0, torch.tensor(-100.0).cuda(),
                       intersection.float() / union.float())


def average_segcover(segA, segB, ignore_background=False):
    """
    Covering of segA by segB
    segA.shape = [batch size, 1, img_dim1, img_dim2]
    segB.shape = [batch size, 1, img_dim1, img_dim2]
    scale: If true, take weighted mean over IOU values proportional to the
           the number of pixels of the mask being covered.
    Assumes labels in segA and segB are non-negative integers.
    Negative labels will be ignored.
    """

    assert segA.shape == segB.shape, f"{segA.shape} - {segB.shape}"
    assert segA.shape[1] == 1 and segB.shape[1] == 1
    bsz = segA.shape[0]
    nonignore = (segA >= 0)

    mean_scores = torch.tensor(bsz*[0.0]).cuda()
    N = torch.tensor(bsz*[0]).cuda()
    scaled_scores = torch.tensor(bsz*[0.0]).cuda()
    scaling_sum = torch.tensor(bsz*[0]).cuda()

    # Find unique label indices to iterate over
    if ignore_background:
        iter_segA = torch.unique(segA[segA > 0]).tolist()
    else:
        iter_segA = torch.unique(segA[segA >= 0]).tolist()
    iter_segB = torch.unique(segB[segB >= 0]).tolist()
    # Loop over segA
    for i in iter_segA:
        binaryA = segA == i
        if not binaryA.any():
            continue
        max_iou = torch.tensor(bsz*[0.0]).cuda()
        # Loop over segB to find max IOU
        for j in iter_segB:
            # Do not penalise pixels that are in ignore regions
            binaryB = (segB == j) * nonignore
            if not binaryB.any():
                continue
            iou = iou_binary(binaryA, binaryB)
            max_iou = torch.where(iou > max_iou, iou, max_iou)
        # Accumulate scores
        mean_scores += max_iou
        N = torch.where(binaryA.sum((1, 2, 3)) > 0, N+1, N)
        scaled_scores += binaryA.sum((1, 2, 3)).float() * max_iou
        scaling_sum += binaryA.sum((1, 2, 3))

    # Compute coverage
    mean_sc = mean_scores / torch.max(N, torch.tensor(1)).float()
    scaled_sc = scaled_scores / torch.max(scaling_sum, torch.tensor(1)).float()

    # Sanity check
    assert (mean_sc >= 0).all() and (mean_sc <= 1).all(), mean_sc
    assert (scaled_sc >= 0).all() and (scaled_sc <= 1).all(), scaled_sc
    assert (mean_scores[N == 0] == 0).all()
    assert (mean_scores[nonignore.sum((1, 2, 3)) == 0] == 0).all()
    assert (scaled_scores[N == 0] == 0).all()
    assert (scaled_scores[nonignore.sum((1, 2, 3)) == 0] == 0).all()

    # Return mean over batch dimension 
    return mean_sc.mean(0), scaled_sc.mean(0)


def _match_segmentation(pred, gt):
    total_gt = np.max(gt)
    cnt = 0
    matches = np.zeros((total_gt, np.max(pred)))

    num1 = np.max(gt) + 1
    num2 = np.max(pred) + 1
    confcounts = np.zeros((num1, num2))

    # joint histogram
    sumim = 1 + gt + pred * num1

    hs, _ = np.histogram(sumim.flatten(), bins=np.linspace(1, num1*num2+1, num=num1*num2+1))
    hs = hs.reshape(confcounts.shape[1], confcounts.shape[0]).T

    confcounts = confcounts + hs
    accuracies = np.zeros((num1, num2))

    for j in range(0, num1):
        for i in range(0, num2):
            gtj = np.sum(confcounts[j, :])
            resj = np.sum(confcounts[:, i])
            gtjresj = confcounts[j, i]
            if gtj + resj - gtjresj:
                value = gtjresj / (gtj + resj - gtjresj)
                accuracies[j, i] = value
    matches[cnt:cnt + np.max(gt), :] = accuracies[1:, 1:]

    return matches.T