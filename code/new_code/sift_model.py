import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from math import sqrt


class sift_model:
    def __init__(self):
        self.img = None
        self.gray_image = None
        self.key_points = None
        self.descriptors = None
        self.color = (0, 0, 255)
        self.distance = cv2.NORM_L2
        self.k = 3

    # loading image
    def load_image(self, image):
        self.img = cv2.imread(image)
        self.gray_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.h, self.w = self.gray_image.shape
        return

    # function to extract picture data (keypoint and descriptor)
    def featureExtractor(self):
        self.sift = cv2.SIFT_create(nfeatures=100000, nOctaveLayers=7, sigma=1.6)
        self.key_points, self.descriptors = self.sift.detectAndCompute(
            self.gray_image, None
        )
        return self.key_points, self.descriptors

    # function to matching data using euclidean distance
    def featureMatching(self):

        # create lowe bib algorithm
        self.bf = cv2.BFMatcher(self.distance, crossCheck=False)

        # setup ratio with value from papers
        ratio = 0.5

        # matching process
        matches = self.bf.knnMatch(self.descriptors, self.descriptors, k=self.k)

        # remove self matching points
        first_grade_matches = []
        for a, b, c in matches:
            if a.trainIdx == a.queryIdx:
                first_grade_matches.append([b, c])
            elif b.trainIdx == b.queryIdx:
                first_grade_matches.append([a, c])
            elif c.trainIdx == c.queryIdx:
                first_grade_matches.append([a, b])

        # parsing by distance
        good_matches = []
        for m, n in first_grade_matches:
            if m.distance < ratio * n.distance:
                good_matches.append(m)

        # affine transform
        MIN_MATCH_COUNT = 3
        if len(good_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([self.key_points[m.trainIdx].pt for m in good_matches])
            dst_pts = np.float32([self.key_points[m.queryIdx].pt for m in good_matches])

            retval, inliers = cv2.estimateAffine2D(
                src_pts,
                dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=3,
                maxIters=100,
                confidence=0.99,
            )

            matchesMask = inliers.ravel().tolist()

        # Filter with ransac
        final_matches = []
        for i in range(len(good_matches)):
            if matchesMask[i] == 1:
                final_matches.append(good_matches[i])

        return good_matches, final_matches, retval

    def show_result(self, final_matches, keypoints):
        list_point1 = []
        list_point2 = []
        img_RGB = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2RGB)
        for j in final_matches:

            # Get the matching keypoints for each of the images
            point1 = j.trainIdx
            point2 = j.queryIdx

            # Get the coordinates, x - columns, y - rows
            (x1, y1) = keypoints[point1].pt
            (x2, y2) = keypoints[point2].pt

            # Append to each list
            list_point1.append((int(x1), int(y1)))
            list_point2.append((int(x2), int(y2)))

            # Draw a small circle at both co-ordinates: radius 4, colour green, thickness = 1
            # copy keypoints circles
            cv2.circle(img_RGB, (int(x1), int(y1)), 4, (0, 255, 0), 1)
            # original keypoints circles
            cv2.circle(img_RGB, (int(x2), int(y2)), 4, (0, 255, 0), 1)

            # Draw a line in between the two points, thickness = 1, colour green
            cv2.line(img_RGB, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

        return cv2.imwrite("point_location.png", img_RGB)

    def compute_correlation_region(self, retval):
        wrapAffine_img = cv2.warpAffine(self.gray_image, retval, (self.w, self.h))
        cv2.imwrite("wrapAffine_img.png", wrapAffine_img)

        # Black image filled up with zeros
        blank_image = np.zeros((self.h, self.w, 1), np.uint8)

        for y in range(0, self.h - 4):
            for x in range(0, self.w - 4):
                window1 = self.gray_image[y : y + 5, x : x + 5]
                window2 = wrapAffine_img[y : y + 5, x : x + 5]
                a1 = window1[3][3]
                a2 = window2[3][3]
                mean1 = cv2.mean(window1)
                mean2 = cv2.mean(window2)
                mean1_num = mean1[0]
                mean2_num = mean2[0]
                b1 = a1 - mean1_num
                b2 = a2 - mean2_num
                top = b1 * b2
                bottom = math.sqrt((b1 * b1) * (b2 * b2))

                if bottom > 0:
                    print(top / bottom)
                    intensity = top / bottom
                    blank_image[y + 2, x + 2] = intensity
                elif bottom <= 0:
                    intensity = 0
                    blank_image[y + 2, x + 2] = intensity

        return cv2.imwrite("blank_image.png", blank_image)

    # function to show loaded image
    def show_image(self):
        cv2.imshow("Image", self.gray_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # function to create keypoint image
    def show_keypoint(self, keypoints):
        return cv2.drawKeypoints(self.gray_image, keypoints, self.img)

    # function to draw match point
    def show_matches(self, keypoint_1, keypoint_2, matches):
        return cv2.drawMatchesKnn(
            self.gray_image,
            keypoint_1,
            self.gray_image,
            keypoint_2,
            matches,
            None,
            flags=2,
        )

    # function to save image
    def save_image(self, image_name, image):
        return cv2.imwrite(image_name, image)


def main():
    img_path = "dataset_example_blur.png"

    # create model
    model = sift_model()

    # loading image
    model.load_image(img_path)

    # show image to make sure image loaded
    model.show_image()

    # extract keypoint and descriptor
    keypoint, descriptor = model.featureExtractor()

    # save keypoint image
    keypoint_image = model.show_keypoint(keypoint)
    model.save_image("keypoint_location.png", keypoint_image)

    # keypoint selection and affine transform matching
    good_matches, final_matches, retval = model.featureMatching()
    model.show_result(final_matches, keypoint)

    #Region Corellation
    model


if __name__ == "__main__":
    main()

