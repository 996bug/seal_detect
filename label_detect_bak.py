# -*- coding: utf-8 -*-
# ☯ Date  : 2023/2/24 10:05
import os
import cv2
import time
import numpy as np
import onnxruntime
from numpy import *
from PIL import Image
from glob import glob


class UnionFind(object):
    """Union-find disjoint sets datastructure.
    Union-find is a data structure that maintains disjoint set
    (called connected components or components in short) membership,
    and makes it easier to merge (union) two components, and to find
    if two elements are connected (i.e., belong to the same
    component).
    This implements the "weighted-quick-union-with-path-compression"
    union-find algorithm.  Only works if elements are immutable
    objects.
    Worst case for union and find: :math:`(N + M \log^* N)`, with
    :math:`N` elements and :math:`M` unions. The function
    :math:`\log^*` is the number of times needed to take :math:`\log`
    of a number until reaching 1. In practice, the amortized cost of
    each operation is nearly linear [1]_.
    Terms
    -----
    Component
        Elements belonging to the same disjoint set
    Connected
        Two elements are connected if they belong to the same component.
    Union
        The operation where two components are merged into one.
    Root
        An internal representative of a disjoint set.
    Find
        The operation to find the root of a disjoint set.
    Parameters
    ----------
    elements : NoneType or container, optional, default: None
        The initial list of elements.
    Attributes
    ----------
    n_elts : int
        Number of elements.
    n_comps : int
        Number of distjoint sets or components.
    Implements
    ----------
    __len__
        Calling ``len(uf)`` (where ``uf`` is an instance of ``UnionFind``)
        returns the number of elements.
    __contains__
        For ``uf`` an instance of ``UnionFind`` and ``x`` an immutable object,
        ``x in uf`` returns ``True`` if ``x`` is an element in ``uf``.
    __getitem__
        For ``uf`` an instance of ``UnionFind`` and ``i`` an integer,
        ``res = uf[i]`` returns the element stored in the ``i``-th index.
        If ``i`` is not a valid index an ``IndexError`` is raised.
    __setitem__
        For ``uf`` and instance of ``UnionFind``, ``i`` an integer and ``x``
        an immutable object, ``uf[i] = x`` changes the element stored at the
        ``i``-th index. If ``i`` is not a valid index an ``IndexError`` is
        raised.
    .. [1] http://algs4.cs.princeton.edu/lectures/
    """

    def __init__(self, elements=None):
        self.n_elts = 0  # current num of elements
        self.n_comps = 0  # the number of disjoint sets or components
        self._next = 0  # next available id
        self._elts = []  # the elements
        self._indx = {}  # dict mapping elt -> index in _elts
        self._par = []  # parent: for the internal tree structure
        self._siz = []  # size of the component - correct only for roots

        if elements is None:
            elements = []
        for elt in elements:
            self.add(elt)

    def __repr__(self):
        return (
            '<UnionFind:\n\telts={},\n\tsiz={},\n\tpar={},\nn_elts={},n_comps={}>'
                .format(
                self._elts,
                self._siz,
                self._par,
                self.n_elts,
                self.n_comps,
            ))

    def __len__(self):
        return self.n_elts

    def __contains__(self, x):
        return x in self._indx

    def __getitem__(self, index):
        if index < 0 or index >= self._next:
            raise IndexError('index {} is out of bound'.format(index))
        return self._elts[index]

    def __setitem__(self, index, x):
        if index < 0 or index >= self._next:
            raise IndexError('index {} is out of bound'.format(index))
        self._elts[index] = x

    def add(self, x):
        """Add a single disjoint element.
        Parameters
        ----------
        x : immutable object
        Returns
        -------
        None
        """
        if x in self:
            return
        self._elts.append(x)
        self._indx[x] = self._next
        self._par.append(self._next)
        self._siz.append(1)
        self._next += 1
        self.n_elts += 1
        self.n_comps += 1

    def find(self, x):
        """Find the root of the disjoint set containing the given element.
        Parameters
        ----------
        x : immutable object
        Returns
        -------
        int
            The (index of the) root.
        Raises
        ------
        ValueError
            If the given element is not found.
        """
        if x not in self._indx:
            raise ValueError('{} is not an element'.format(x))

        p = self._indx[x]
        while p != self._par[p]:
            # path compression
            q = self._par[p]
            self._par[p] = self._par[q]
            p = q
        return p

    def connected(self, x, y):
        """Return whether the two given elements belong to the same component.
        Parameters
        ----------
        x : immutable object
        y : immutable object
        Returns
        -------
        bool
            True if x and y are connected, false otherwise.
        """
        return self.find(x) == self.find(y)

    def union(self, x, y):
        """Merge the components of the two given elements into one.
        Parameters
        ----------
        x : immutable object
        y : immutable object
        Returns
        -------
        None
        """
        # Initialize if they are not already in the collection
        for elt in [x, y]:
            if elt not in self:
                self.add(elt)

        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return
        if self._siz[xroot] < self._siz[yroot]:
            self._par[xroot] = yroot
            self._siz[yroot] += self._siz[xroot]
        else:
            self._par[yroot] = xroot
            self._siz[xroot] += self._siz[yroot]
        self.n_comps -= 1

    def component(self, x):
        """Find the connected component containing the given element.
        Parameters
        ----------
        x : immutable object
        Returns
        -------
        set
        Raises
        ------
        ValueError
            If the given element is not found.
        """
        if x not in self:
            raise ValueError('{} is not an element'.format(x))
        elts = np.array(self._elts)
        vfind = np.vectorize(self.find)
        roots = vfind(elts)
        return set(elts[roots == self.find(x)])

    def components(self):
        """Return the list of connected components.
        Returns
        -------
        list
            A list of sets.
        """
        elts = np.array(self._elts)
        vfind = np.vectorize(self.find)
        roots = vfind(elts)
        distinct_roots = set(roots)
        return [set(elts[roots == root]) for root in distinct_roots]
        # comps = []
        # for root in distinct_roots:
        #     mask = (roots == root)
        #     comp = set(elts[mask])
        #     comps.append(comp)
        # return comps

    def component_mapping(self):
        """Return a dict mapping elements to their components.
        The returned dict has the following semantics:
            `elt -> component containing elt`
        If x, y belong to the same component, the comp(x) and comp(y)
        are the same objects (i.e., share the same reference). Changing
        comp(x) will reflect in comp(y).  This is done to reduce
        memory.
        But this behaviour should not be relied on.  There may be
        inconsitency arising from such assumptions or lack thereof.
        If you want to do any operation on these sets, use caution.
        For example, instead of
        ::
            s = uf.component_mapping()[item]
            s.add(stuff)
            # This will have side effect in other sets
        do
        ::
            s = set(uf.component_mapping()[item]) # or
            s = uf.component_mapping()[item].copy()
            s.add(stuff)
        or
        ::
            s = uf.component_mapping()[item]
            s = s | {stuff}  # Now s is different
        Returns
        -------
        dict
            A dict with the semantics: `elt -> component contianing elt`.
        """
        elts = np.array(self._elts)
        vfind = np.vectorize(self.find)
        roots = vfind(elts)
        distinct_roots = set(roots)
        comps = {}
        for root in distinct_roots:
            mask = (roots == root)
            comp = set(elts[mask])
            comps.update({x: comp for x in comp})
            # Change ^this^, if you want a different behaviour:
            # If you don't want to share the same set to different keys:
            # comps.update({x: set(comp) for x in comp})
        return comps


class LabelDetector:
    def __init__(self, mode=False):
        self.session_a = onnxruntime.InferenceSession(fr'models\label\{r"big.onnx" if mode is True else r"small.onnx"}')
        self.session_b = onnxruntime.InferenceSession(fr'models\angle\{r"big.onnx" if mode is True else r"small.onnx"}')
        self.imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # rgb

    @staticmethod
    def bb_intersection_over_union(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(min(boxAArea, boxBArea))

        # return the intersection over union value
        return iou

    def custom_nms(self, boxes):
        valid_boxes = []
        n = len(boxes)
        uf = UnionFind(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                iou = self.bb_intersection_over_union(boxes[i], boxes[j])
                if iou > 0.9:
                    uf.union(i, j)
        for idx in uf.components():
            idx = list(idx)
            if len(idx) == 1:
                valid_boxes.append(boxes[idx[0]])
            else:
                idx = np.array(idx)
                union_boxes = boxes[idx]
                xmin = np.min(union_boxes[:, 0])
                ymin = np.min(union_boxes[:, 1])
                xmax = np.max(union_boxes[:, 2])
                ymax = np.max(union_boxes[:, 3])
                valid_boxes.append([xmin, ymin, xmax, ymax])

        return valid_boxes

    @staticmethod
    def resize(x):
        h, w = x.shape[:2]
        norm_h = 640
        if h > w:
            ratio = norm_h / h
            x = cv2.resize(x, (0, 0), fx=ratio, fy=ratio)
            xx = np.zeros((norm_h, norm_h, 3))
            xx[:x.shape[0], :x.shape[1], :] = x
            x = xx
        else:
            ratio = norm_h / w
            x = cv2.resize(x, (0, 0), fx=ratio, fy=ratio)
            xx = np.zeros((norm_h, norm_h, 3))
            xx[:x.shape[0], :x.shape[1], :] = x
            x = xx
        return x

    def read_image(self, file):
        image = np.array(Image.open(file).convert("RGB"))
        resize_short = 640
        src_image0 = image.copy()
        img_h, img_w = src_image0.shape[:2]
        percent = float(resize_short) / min(img_w, img_h)
        if percent < 2:  # 正常图像
            w = int(round(img_w * percent))
            h = int(round(img_h * percent))
            src_image0 = cv2.resize(src_image0, (w, h))
        else:  # 小图像
            top, bottom, left, right = 0, 0, 0, 0
            if resize_short > img_h:
                top = (resize_short - img_h) // 2
                bottom = resize_short - top - img_h
            if resize_short > img_w:
                left = (resize_short - img_w) // 2
                right = resize_short - img_w - left
            src_image0 = cv2.copyMakeBorder(src_image0, top, bottom, left, right, cv2.BORDER_CONSTANT)  # 中心外扩
        w, h = 512, 512
        img_h, img_w = src_image0.shape[:2]
        if img_h < h or img_w < w:
            return image
        w_start = (img_w - w) // 2
        h_start = (img_h - h) // 2

        w_end = w_start + w
        h_end = h_start + h
        src_image0 = src_image0[h_start:h_end, w_start:w_end, :]

        src_image0 = src_image0[:, :, ::-1]  # bgr2rgb

        src_image0 = src_image0 / 255.0
        src_image0 = src_image0 - [0.485, 0.456, 0.406]
        src_image0 = src_image0 / [0.229, 0.224, 0.225]

        src_image0 = src_image0.astype(np.float32)
        image_input = np.expand_dims(src_image0, axis=0)
        image_input = np.transpose(image_input, (0, 3, 1, 2))
        base_angle_ort_inputs = {self.session_b.get_inputs()[0].name: image_input}
        base_angle_out = self.session_b.run(None, base_angle_ort_inputs)[0]
        # 根据大角度旋转文本图像到0度，0度情况下检测效果最佳
        if np.argmax(base_angle_out) == 1:
            image = np.rot90(image, 1)
        if np.argmax(base_angle_out) == 2:
            image = np.rot90(image, 2)
        if np.argmax(base_angle_out) == 3:
            image = np.rot90(image, 3)
        return image

    def _detect(self, img):
        # img = np.array(Image.open(file).convert("RGB"))
        img_ratio = [[640 / img.shape[0], 640 / img.shape[1]]]
        img = cv2.resize(img, (640, 640))
        img = img / 255.0 - self.imagenet_stats[0]
        img = img / self.imagenet_stats[1]
        img = np.expand_dims(img, axis=0)
        img = np.transpose(img, (0, 3, 1, 2))

        ort_inputs = {
            self.session_a.get_inputs()[0].name: np.float32(img),
            self.session_a.get_inputs()[1].name: np.float32(np.asarray(img_ratio))
        }
        det_out, num_out = self.session_a.run(None, ort_inputs)
        ind = det_out[:, 1] > 0.2
        det_out = det_out[ind]

        shouxie_out = det_out[det_out[:, 0] == 0]
        shouyin_out = det_out[det_out[:, 0] == 1]
        yinzhang_out = det_out[det_out[:, 0] == 2]

        items = list()
        shouxie_out = shouxie_out[:, 2:]
        if len(shouxie_out) > 0:
            for det in self.custom_nms(shouxie_out):
                items.append({"label": "手写", "box": (int(det[0]), int(det[1]), int(det[2]), int(det[3]))})

        shouyin_out = shouyin_out[:, 2:]
        if len(shouyin_out) > 0:
            for det in shouyin_out:
                items.append({"label": "手印", "box": (int(det[0]), int(det[1]), int(det[2]), int(det[3]))})

        yinzhang_out = yinzhang_out[:, 2:]
        if len(yinzhang_out) > 0:
            for det in self.custom_nms(yinzhang_out):
                items.append({"label": "印章", "box": (int(det[0]), int(det[1]), int(det[2]), int(det[3]))})

        return items

    def detect(self, file, debug=True):
        # img = np.array(Image.open(file).convert("RGB"))
        img = self.read_image(file)
        h1, w1, _ = img.shape
        if max(h1, w1) / min(h1, w1) > 2.5:
            return
        items = self._detect(img)
        # if debug and len(items) > 0:
        #     for item in items:
        #         box = item['box']
        #         cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        # show_img = cv2.resize(img, (int(w1 * (800 / h1)), 800))
        # cv2.imshow("test", show_img)
        # cv2.waitKey(0)
        return items

    def run(self):
        task_path = r"E:\签字合同_图片资源"
        useful_path = r"E:\签字合同_图片资源\有效资源"
        useless_path = r"E:\签字合同_图片资源\无效资源"
        files = glob(fr'{task_path}\*.jpg')
        for file in files[0:]:
            filename = os.path.basename(file)
            new_fl = os.path.join(useless_path, filename)
            try:
                items = self.detect(file)
                if items is None or len(items) == 0:
                    new_fl = os.path.join(useless_path, filename)
                else:
                    flag = '_'.join(list(set([item['label'] for item in items])))
                    new_fl = os.path.join(useful_path, filename.split('.')[0] + "_" + flag + ".jpg")
            except Exception as e:
                _ = e
            finally:
                print(file, new_fl)
                try:
                    os.rename(file, new_fl)
                except:
                    pass
                # break


if __name__ == '__main__':
    start = LabelDetector()
    start.run()
