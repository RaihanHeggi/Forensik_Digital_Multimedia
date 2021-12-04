import cv2
import matplotlib.pyplot as plt
import numpy as np


class sift_model:
    def __init__(self, img_1, img_2):
        self.img_1_path = img_1
        self.img_2_path = img_2
        self.MIN_MATCH_COUNT = 3
        self.sift = cv2.SIFT_create()
        # builtin features matcher function

        # brute force matcher
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # flann matcher
        index_params = dict(algorithm=3, trees=10)
        search_params = dict(checks=50)

        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Loading data to opencv and  preprocessing to BnW
    def load_image(self):
        # load image pixel
        self.img_1 = cv2.imread(self.img_1_path)
        self.img_2 = cv2.imread(self.img_2_path)

        # change to black and white
        self.img_1 = cv2.cvtColor(self.img_1, cv2.COLOR_BGR2GRAY)
        self.img_2 = cv2.cvtColor(self.img_2, cv2.COLOR_BGR2GRAY)

        # load size
        self.h, self.w = self.img_2.shape

        return self.img_1, self.img_2

    # getting keypoint value using SIFT algorithm
    def getKeypointDescriptor(self):
        keypoints_1, descriptors_1 = self.sift.detectAndCompute(self.img_2, None)
        keypoints_2, descriptors_2 = self.sift.detectAndCompute(self.img_2, None)
        return keypoints_1, descriptors_1, keypoints_2, descriptors_2

    def remove_best_self(self, matches):
        better_matches = list()
        for a, b, c in matches:
            if a.trainIdx == a.queryIdx:
                better_matches.append([b, c])
            elif b.trainIdx == b.queryIdx:
                better_matches.append([a, c])
            elif c.trainIdx == c.queryIdx:
                better_matches.append([a, b])
        return better_matches

    # features matching
    def features_matching(self, des1, des2):
        matches = self.flann.knnMatch(des1, des2, k=2)

        matchesMask = [[0, 0] for i in range(len(matches))]
        best_match = list()
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            # threshold value 0.5 from paper
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [0, 1]
                best_match.append(m)

        draws_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            matchesMask=matchesMask,
            flags=cv2.DrawMatchesFlags_DEFAULT,
        )

        ################ old value #####################
        # better_matches = self.remove_best_self(matches)
        # best_match = list()
        # for m, n in better_matches:
        #     # threshold value 0.5 from paper
        #     if m.distance < 0.5 * n.distance:
        #         best_match.append(n)
        return matches, matchesMask, draws_params, best_match

    # calculate affine transform using ransac
    def affine_ransac(self, kp1, kp2, best_match):
        final_matches = list()
        if len(best_match) > self.MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in best_match])
            dst_pts = np.float32([kp1[m.trainIdx].pt for m in best_match])
            
            retval, inliers = cv2.estimateAffine2D(
                src_pts,
                dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=3,
                maxIters=100,
                confidence=0.99,
            )

            dst = inliers.ravel().tolist()

            for i in range(len(best_match)):
                if dst[i] == 1:
                    final_matches.append(best_match[i])

        else:
            print(
                "Not enough matches are found - %d/%d"
                % (len(best_match), self.MIN_MATCH_COUNT)
            )
            matchesMask = None
            final_matches = None
        return final_matches, retval

    # print image
    def show_image(self, img_1, img_2):
        figure, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].imshow(img_1, cmap="gray")
        ax[1].imshow(img_2, cmap="gray")
        plt.show()
        return

    # print_keypoint location
    def show_keypoint(self, keypoints_1, keypoints_2):
        # colored
        colored_img_1 = cv2.imread(self.img_1_path)
        colored_img_2 = cv2.imread(self.img_2_path)

        # black and white
        img_1 = cv2.drawKeypoints(self.img_1, keypoints_1, colored_img_1)
        img_2 = cv2.drawKeypoints(self.img_2, keypoints_2, colored_img_2)

        figure, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].imshow(img_1, cmap="gray")
        ax[1].imshow(img_2, cmap="gray")

        plt.show()
        return

    def show_matches(self, keypoints_1, keypoints_2, matches, draw_params):
        img_3 = cv2.drawMatchesKnn(
            self.img_1,
            keypoints_1,
            self.img_2,
            keypoints_2,
            matches,
            None,
            **draw_params,
        )
        plt.imshow(img_3)
        plt.show()
        return

    def show_final_matches(self, keypoints, final_matches, retval):
        img_RGB = cv2.cvtColor(self.img_2, cv2.COLOR_GRAY2RGB)

        list_point1 = []
        list_point2 = []
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
            cv2.circle(img_RGB, (int(x1), int(y1)), 4, (255, 0, 0), 1)
            # original keypoints circles
            cv2.circle(img_RGB, (int(x2), int(y2)), 4, (0, 255, 0), 1)

            # Draw a line in between the two points, thickness = 1, colour green
            # cv2.line(img_RGB, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

        # 6) Computing region correlation map
        # 6.1) WrapAffine
        wrapAffine_img = cv2.warpAffine(self.img_2, retval, (self.w, self.h))
        cv2.imwrite("res.png", wrapAffine_img)

        # 3) Output
        # cv2.imshow("result", img_RGB)
        cv2.imwrite("final_matches.png", img_RGB)
        # cv2.imshow("corelation", img_correlation_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return


def main():
    img_path_1 = "dataset_example.png"
    img_path_2 = "dataset_example_blur.png"

    model = sift_model(img_path_1, img_path_2)
    img_1, img_2 = model.load_image()

    # getting black and white picture,  un-comment to see result
    # model.show_image(img_1, img_2)

    # getting keypoint
    (
        keypoints_1,
        descriptors_1,
        keypoints_2,
        descriptors_2,
    ) = model.getKeypointDescriptor()

    # getting keypoint picture position,  un-comment to see result
    # model.show_keypoint(keypoints_1, keypoints_2)

    # calculate best match
    best_match_value, matches_mask, draw_params, good_match = model.features_matching(
        descriptors_1, descriptors_2
    )

    # getting bestmatch location, un-comment to see result
    # model.show_matches(keypoints_1, keypoints_2, best_match_value, draw_params)

    # affine transform
    final_matches, retval = model.affine_ransac(keypoints_1, keypoints_2, good_match)

    # getting final_matches location, un-comment to see result
    model.show_final_matches(keypoints_2, final_matches, retval)


if __name__ == "__main__":
    main()

