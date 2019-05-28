import cv2
import numpy as np

class AnchorMaker(object):
    """docstring for AnchorMaker"""
    def __init__(self, target_file, label_path, cluster_num, colab=False):
        super(AnchorMaker, self).__init__()
        self.target_file = target_file
        self.label_path = label_path
        self.cluster_num = cluster_num
        self.seed = 1
        self.dist = np.median
        self.colab = colab

    def parse_anno(self):
        anno = open(self.label_path, 'r')
        result = []
        for line in anno:
            s = line.strip().split(' ')
            if self.colab:
                image = cv2.imread(str(s[0]+" "+s[1]))
            else:
                image = cv2.imread(s[0])
            image_h, image_w = image.shape[:2]

            if self.colab:
                s = s[2:]
            else:
                s = s[1:]
            box_cnt = len(s) // 5
            for i in range(box_cnt):
                x_min, y_min, x_max, y_max = float(s[i*5+0]), float(s[i*5+1]), float(s[i*5+2]), float(s[i*5+3])
                width  = (x_max - x_min) / image_w
                height = (y_max - y_min) / image_h
                result.append([width, height])
        result = np.asarray(result)
        return result


    def iou(self, box, clusters):
        """
        Calculates the Intersection over Union (IoU) between a box and k clusters.
        param:
            box: tuple or array, shifted to the origin (i. e. width and height)
            clusters: numpy array of shape (k, 2) where k is the number of clusters
        return:
            numpy array of shape (k, 0) where k is the number of clusters
        """
        x = np.minimum(clusters[:, 0], box[0])
        y = np.minimum(clusters[:, 1], box[1])
        if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
            raise ValueError("Box has no area")

        intersection = x * y
        box_area = box[0] * box[1]
        cluster_area = clusters[:, 0] * clusters[:, 1]

        iou_ = intersection / (box_area + cluster_area - intersection)

        return iou_


    def kmeans(self):
        rows = self.boxes.shape[0]
        distances     = np.empty((rows, self.cluster_num)) ## N row x N cluster
        last_clusters = np.zeros((rows,))

        np.random.seed(self.seed)

        # initialize the cluster centers to be k items
        clusters = self.boxes[np.random.choice(rows, self.cluster_num, replace=False)]

        while True:
            # Step 1: allocate each item to the closest cluster centers
            for icluster in range(self.cluster_num): # I made change to lars76's code here to make the code faster
                distances[:,icluster] = 1 - self.iou(clusters[icluster], self.boxes)

            nearest_clusters = np.argmin(distances, axis=1)

            if (last_clusters == nearest_clusters).all():
                break

            # Step 2: calculate the cluster centers as mean (or median) of all the cases in the clusters.
            for cluster in range(self.cluster_num):
                clusters[cluster] = self.dist(self.boxes[nearest_clusters == cluster], axis=0)
            last_clusters = nearest_clusters

        return clusters, nearest_clusters, distances

    def make_anchors(self):

        self.boxes = self.parse_anno()
        clusters, nearest_clusters, distances = self.kmeans()

        # sorted by area
        area = clusters[:, 0] * clusters[:, 1]
        indice = np.argsort(area)
        clusters = clusters[indice]
        with open(self.target_file, "w") as f:
            for i in range(self.cluster_num):
                width, height = clusters[i]
                f.writelines(str(width) + " " + str(height) + " ")

if __name__ == '__main__':
    anchormaker = AnchorMaker(target_file="../../data/anchors.txt",
                              label_path="../../data/labels-train.txt",
                              cluster_num=4)
    anchormaker.make_anchors()
