import numpy as np


def constraint(w, h, disp_max, patch_size, neg_high):
    off_w = 1 + patch_size // 2 + neg_high
    off_h = 1 + patch_size // 2
    return off_w + disp_max, w - off_w, off_h, h - off_h


def make_patch(left, right, K, idx_L, idx_RP, idx_RQ):
    """
    :param left:left image 3D
    :param right: right image 3D
    :param K: input_patch_size
    :param idx_L: tuple of indexes
    :param idx_RP:
    :return: ndarray
    """
    off = K // 2
    upp_L, low_L = idx_L + 1 + off, idx_L - off
    upp_L, low_L = upp_L, low_L
    patch_left = left[int(low_L[0]):int(upp_L[0]), int(low_L[1]):int(upp_L[1]), :]

    upp_RP, low_RP = idx_RP + 1 + off, idx_RP - off
    patch_right_P = right[int(low_RP[0]):int(upp_RP[0]), int(low_RP[1]):int(upp_RP[1]), :]

    upp_RQ, low_RQ = idx_RQ + 1 + off, idx_RQ - off
    patch_right_Q = right[int(low_RQ[0]):int(upp_RQ[0]), int(low_RQ[1]):int(upp_RQ[1]), :]

    assert (patch_left.size == patch_right_P.size == patch_right_Q.size) and (
            patch_left.size == K ** 2 * left.shape[-1]), str((idx_L, idx_RP, idx_RQ))
    patch = np.array(
        [patch_left.transpose(2, 0, 1), patch_right_P.transpose(2, 0, 1), patch_right_Q.transpose(2, 0, 1)])
    return patch


def generate_examples(example, *, patch_size=9, pos_off=1, neg_off_l=4, neg_off_h=14):
    left, right, disparity = example
    disparity = np.uint8(disparity)
    wc_L, wc_H, hc_L, hc_H = constraint(disparity.shape[1], disparity.shape[0], disparity.max(),
                                        patch_size, neg_off_h)
    # some disparities set as 0 are not valid
    nonzero_disp = np.nonzero(disparity)
    nonzero_disp = (
        nonzero_disp[0][(nonzero_disp[0] >= hc_L) & (nonzero_disp[0] <= hc_H) & (nonzero_disp[1] >= wc_L) & (
                nonzero_disp[1] <= wc_H)],
        nonzero_disp[1][(nonzero_disp[0] >= hc_L) & (nonzero_disp[0] <= hc_H) & (nonzero_disp[1] >= wc_L) & (
                nonzero_disp[1] <= wc_H)]
    )
    # prepare left indexes for valid disparities
    valid_indexes_left = np.transpose(nonzero_disp)
    # prepare left indexes for valid disparities: offset
    valid_indexes_right = np.array(
        [valid_indexes_left[:, 0], valid_indexes_left[:, 1] - disparity[nonzero_disp]]).T

    # offset for positive examples
    opos = np.random.randint(-pos_off, pos_off, valid_indexes_right[:, 1].size) if pos_off != 0 else np.zeros(
        valid_indexes_right[:, 1].size)
    # offset for negative examples
    oneg_choices = list(range(neg_off_l, neg_off_h))
    oneg_choices.extend(list(range(-neg_off_h, -neg_off_l)))
    oneg = np.random.choice(oneg_choices, valid_indexes_right[:, 1].size)
    # finally, indexes for all positive and all negative examples in an image
    valid_indexes_right_P = np.array([valid_indexes_right[:, 0], valid_indexes_right[:, 1] + opos]).T
    valid_indexes_right_Q = np.array([valid_indexes_right[:, 0], valid_indexes_right[:, 1] + oneg]).T

    patches = [make_patch(left, right, patch_size, idx_L, idx_RP, idx_RQ)
               for idx_L, idx_RP, idx_RQ in zip(valid_indexes_left, valid_indexes_right_P, valid_indexes_right_Q)]
    return np.array(patches)
