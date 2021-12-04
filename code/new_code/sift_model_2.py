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
        self.sift = cv2.SIFT_create()
        self.key_points, self.descriptors = self.sift.detectAndCompute(
            self.gray_image, None
        )
        return self.key_points, self.descriptors

    # remove self matching point
    def remove_self_match(self, matches):
        better_matches = []
        for a, b, c in matches:
            if a.trainIdx == a.queryIdx:
                better_matches.append([b, c])
            elif b.trainIdx == b.queryIdx:
                better_matches.append([a, c])
            elif c.trainIdx == c.queryIdx:
                better_matches.append([a, b])
        return better_matches

    # function to matching data using euclidean distance
    def featureMatching(self):

        # create lowe bib algorithm
        self.bf = cv2.BFMatcher(self.distance, crossCheck=False)

        # setup ratio with value from papers
        ratio = 0.5

        # matching process
        matches = self.bf.knnMatch(self.descriptors, self.descriptors, k=self.k)
        better_matches = self.remove_self_match(matches)

        # good_match_1 = list()
        # good_match_2 = list()
        good_list = list()

        for m, n in better_matches:
            if m.distance < ratio * n.distance:
                good_list.append(m)

        # for i in range(1, j):
        #     if (
        #         pdist(
        #             np.array(
        #                 [
        #                     self.key_points[better_matches[i].queryIdx].pt,
        #                     self.key_points[better_matches[i].trainIdx].pt,
        #                 ]
        #             ),
        #             "euclidean",
        #         )
        #         > 3
        #     ):
        #         good_match_1.append(self.key_points[better_matches[i].queryIdx])
        #         good_match_2.append(self.key_points[better_matches[i].trainIdx])

        # return good_match_1, good_match_2, good_list

        return good_list

    # calculate affine transform and robust using ransac
    def affine_ransac(self, good_list):
        # making one inliers list to save value
        inliers = []

        # affine transform and ransac
        p1 = np.float32([self.key_points[m.trainIdx].pt for m in good_list])
        p2 = np.float32([self.key_points[m.queryIdx].pt for m in good_list])

        # getting inliers using ransac
        retval, inliers = cv2.estimateAffine2D(
            p1,
            p2,
            method=cv2.RANSAC,
            ransacReprojThreshold=3,
            maxIters=100,
            confidence=0.99,
        )

        matchesMask = inliers.ravel().tolist()

        # filter with ransac
        final_matches = []
        for i in range(len(good_list)):
            if matchesMask[i] == 1:
                final_matches.append(good_list[i])

        # good_match = list()
        # for i, m in enumerate(match_p1):

        #     col = np.ones((3, 1), dtype=np.float64)
        #     col[0:2, 0] = m.pt
        #     col = np.dot(transformation_rigid_matrix, col)
        #     col /= col[1, 0]

        #     distance = sqrt(
        #         pow(col[0, 0] - match_p2[i].pt[0], 2)
        #         + pow(col[1, 0] - match_p2[i].pt[1], 2)
        #     )

        #     if distance < threshold:
        #         good_matches.append(cv2.DMatch(len(inliers1), len(inliers2), 0))
        #         inliers_1.append(match_p1[i])
        #         inliers_2.append(match_p2[i])

        # good_points1 = np.float32([kp1.pt for kp1 in inliers_1])
        # good_points2 = np.float32([kp2.pt for kp2 in inliers_2])

        # return good_points1, good_points2

        return final_matches, retval

    def compute_correlation_region(self, final_matches, retval):
        wrapAffine_img = cv2.warpAffine(self.gray_image, retval, (self.w, self.h))
        cv2.imwrite("wrapAffine_img.png", wrapAffine_img)

        # correlation region
        blank_image = np.zeros((self.h, self.w, 1), np.uint8)

        for y in range(0, self.h - 3):
            for x in range(0, self.w - 3):
                window1 = self.gray_image[y : y + 3, x : x + 3]
                window2 = wrapAffine_img[y : y + 3, x : x + 3]

                a1 = window1[1][1]
                a2 = window2[1][1]

                mean1 = cv2.mean(window1)
                mean2 = cv2.mean(window2)

                mean1_num = mean1[0]
                mean2_num = mean2[0]
                b1 = a1 - mean1_num
                b2 = a2 - mean2_num
                top = b1 * b2
                bottom = sqrt((b1 * b1) * (b2 * b2))

                if bottom > 0:
                    intensity = top / bottom
                    blank_image[y + 1, x + 1] = intensity
                elif bottom <= 0:
                    intensity = 0
                    blank_image[y + 1, x + 1] = intensity

        cv2.imwrite("blank_image.png", blank_image)

        return final_matches

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
            #cv2.circle(img_RGB, (int(x2), int(y2)), 4, (0, 255, 0), 1)

            # Draw a line in between the two points, thickness = 1, colour green
            #cv2.line(img_RGB, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

        return cv2.imwrite("point_location.png", img_RGB)

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
        return cv2.drawMatches(
            self.gray_image,
            keypoint_1,
            self.gray_image,
            keypoint_2,
            matches,
            self.gray_image,
            flags=2,
        )

    # function to save image
    def save_image(self, image_name, image):
        return cv2.imwrite(image_name, image)

    def rotate(image, angle, center=None, scale=1.0):
        (self.h, self.w) = image.shape[:2]

        if center is None:
            center = (self.w / 2, self.h / 2)

        # Perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))

        return rotated


def main():
    img_path = "car.jpeg"

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

    # feature matching
    matches = model.featureMatching()
    img_feature = model.show_matches(keypoint, keypoint, matches)
    model.save_image("feature_matching.png", img_feature)

    # Affine Transform
    final_matches, retval = model.affine_ransac(matches)

    # print(final_matches)

    # Region Corellation
    last_matches = model.compute_correlation_region(final_matches, retval)
    model.show_result(final_matches, keypoint)


if __name__ == "__main__":
    main()

