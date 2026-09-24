"""Text Image Augmentation (TIA): distort / stretch / perspective via WarpMLS.

Ported from PaddleOCR `ppocr/data/imaug/text_image_aug` (Apache-2.0).
"""

from __future__ import annotations

import numpy as np


class WarpMLS:
    def __init__(self, src, src_pts, dst_pts, dst_w, dst_h, trans_ratio=1.0):
        self.src = src
        self.src_pts = src_pts
        self.dst_pts = dst_pts
        self.pt_count = len(self.dst_pts)
        self.dst_w = dst_w
        self.dst_h = dst_h
        self.trans_ratio = trans_ratio
        self.grid_size = 100
        self.rdx = np.zeros((self.dst_h, self.dst_w))
        self.rdy = np.zeros((self.dst_h, self.dst_w))

    @staticmethod
    def _bilinear_interp(x, y, v11, v12, v21, v22):
        return (v11 * (1 - y) + v12 * y) * (1 - x) + (v21 * (1 - y) + v22 * y) * x

    def generate(self):
        self._calc_delta()
        return self._gen_img()

    def _calc_delta(self):
        w = np.zeros(self.pt_count, dtype=np.float32)
        if self.pt_count < 2:
            return
        i = 0
        while True:
            if self.dst_w <= i < self.dst_w + self.grid_size - 1:
                i = self.dst_w - 1
            elif i >= self.dst_w:
                break
            j = 0
            while True:
                if self.dst_h <= j < self.dst_h + self.grid_size - 1:
                    j = self.dst_h - 1
                elif j >= self.dst_h:
                    break
                sw = 0.0
                swp = np.zeros(2, dtype=np.float32)
                swq = np.zeros(2, dtype=np.float32)
                new_pt = np.zeros(2, dtype=np.float32)
                cur_pt = np.array([i, j], dtype=np.float32)
                k = 0
                for k in range(self.pt_count):
                    if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                        break
                    w[k] = 1.0 / (
                        (i - self.dst_pts[k][0]) ** 2 + (j - self.dst_pts[k][1]) ** 2
                    )
                    sw += w[k]
                    swp = swp + w[k] * np.array(self.dst_pts[k])
                    swq = swq + w[k] * np.array(self.src_pts[k])
                if k == self.pt_count - 1:
                    pstar = (1 / sw) * swp
                    qstar = (1 / sw) * swq
                    miu_s = 0.0
                    for k in range(self.pt_count):
                        if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                            continue
                        pt_i = self.dst_pts[k] - pstar
                        miu_s += w[k] * np.sum(pt_i * pt_i)
                    cur_pt = cur_pt - pstar
                    cur_pt_j = np.array([-cur_pt[1], cur_pt[0]])
                    for k in range(self.pt_count):
                        if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                            continue
                        pt_i = self.dst_pts[k] - pstar
                        pt_j = np.array([-pt_i[1], pt_i[0]])
                        tmp_pt = np.zeros(2, dtype=np.float32)
                        tmp_pt[0] = np.sum(pt_i * cur_pt) * self.src_pts[k][0] - np.sum(pt_j * cur_pt) * self.src_pts[k][1]
                        tmp_pt[1] = -np.sum(pt_i * cur_pt_j) * self.src_pts[k][0] + np.sum(pt_j * cur_pt_j) * self.src_pts[k][1]
                        tmp_pt *= w[k] / max(miu_s, 1e-6)
                        new_pt += tmp_pt
                    new_pt += qstar
                else:
                    new_pt = np.array(self.src_pts[k], dtype=np.float32)
                self.rdx[j, i] = new_pt[0] - i
                self.rdy[j, i] = new_pt[1] - j
                j += self.grid_size
            i += self.grid_size

    def _gen_img(self):
        src_h, src_w = self.src.shape[:2]
        dst = np.zeros_like(self.src, dtype=np.float32)
        for i in np.arange(0, self.dst_h, self.grid_size):
            for j in np.arange(0, self.dst_w, self.grid_size):
                ni = i + self.grid_size
                nj = j + self.grid_size
                w = h = self.grid_size
                if ni >= self.dst_h:
                    ni = self.dst_h - 1
                    h = ni - i + 1
                if nj >= self.dst_w:
                    nj = self.dst_w - 1
                    w = nj - j + 1
                di = np.reshape(np.arange(h), (-1, 1))
                dj = np.reshape(np.arange(w), (1, -1))
                delta_x = self._bilinear_interp(
                    di / h, dj / w, self.rdx[i, j], self.rdx[i, nj], self.rdx[ni, j], self.rdx[ni, nj]
                )
                delta_y = self._bilinear_interp(
                    di / h, dj / w, self.rdy[i, j], self.rdy[i, nj], self.rdy[ni, j], self.rdy[ni, nj]
                )
                nx = np.clip(j + dj + delta_x * self.trans_ratio, 0, src_w - 1)
                ny = np.clip(i + di + delta_y * self.trans_ratio, 0, src_h - 1)
                nxi = np.floor(nx).astype(np.int32)
                nyi = np.floor(ny).astype(np.int32)
                nxi1 = np.ceil(nx).astype(np.int32)
                nyi1 = np.ceil(ny).astype(np.int32)
                if self.src.ndim == 3:
                    x = np.tile(np.expand_dims(ny - nyi, -1), (1, 1, 3))
                    y = np.tile(np.expand_dims(nx - nxi, -1), (1, 1, 3))
                else:
                    x = ny - nyi
                    y = nx - nxi
                dst[i : i + h, j : j + w] = self._bilinear_interp(
                    x, y, self.src[nyi, nxi], self.src[nyi, nxi1], self.src[nyi1, nxi], self.src[nyi1, nxi1]
                )
        return np.clip(dst, 0, 255).astype(np.uint8)


def tia_distort(src, segment=4):
    img_h, img_w = src.shape[:2]
    cut = img_w // segment
    thresh = max(cut // 3, 1)
    src_pts = [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]]
    dst_pts = [
        [np.random.randint(thresh), np.random.randint(thresh)],
        [img_w - np.random.randint(thresh), np.random.randint(thresh)],
        [img_w - np.random.randint(thresh), img_h - np.random.randint(thresh)],
        [np.random.randint(thresh), img_h - np.random.randint(thresh)],
    ]
    half = thresh * 0.5
    for cut_idx in range(1, segment):
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append(
            [cut * cut_idx + np.random.randint(thresh) - half, np.random.randint(thresh) - half]
        )
        dst_pts.append(
            [cut * cut_idx + np.random.randint(thresh) - half, img_h + np.random.randint(thresh) - half]
        )
    return WarpMLS(src, src_pts, dst_pts, img_w, img_h).generate()


def tia_stretch(src, segment=4):
    img_h, img_w = src.shape[:2]
    cut = img_w // segment
    thresh = max(cut * 4 // 5, 1)
    src_pts = [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]]
    dst_pts = [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]]
    half = thresh * 0.5
    for cut_idx in range(1, segment):
        move = np.random.randint(thresh) - half
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append([cut * cut_idx + move, 0])
        dst_pts.append([cut * cut_idx + move, img_h])
    return WarpMLS(src, src_pts, dst_pts, img_w, img_h).generate()


def tia_perspective(src):
    img_h, img_w = src.shape[:2]
    thresh = max(img_h // 2, 1)
    src_pts = [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]]
    dst_pts = [
        [0, np.random.randint(thresh)],
        [img_w, np.random.randint(thresh)],
        [img_w, img_h - np.random.randint(thresh)],
        [0, img_h - np.random.randint(thresh)],
    ]
    return WarpMLS(src, src_pts, dst_pts, img_w, img_h).generate()
