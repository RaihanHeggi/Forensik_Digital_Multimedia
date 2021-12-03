import cv2
import matplotlib.pyplot as plt


class sift_model:
    def __init__(self, img_1, img_2):
        self.img_1_path = img_1
        self.img_2_path = img_2
        self.sift = cv2.SIFT_create()
        # builtin features matcher function

        # brute force matcher
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # flann matcher
        index_params = dict(algorithm=1, trees=5)
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

        return self.img_1, self.img_2

    # getting keypoint value using SIFT algorithm
    def getKeypointDescriptor(self):
        keypoints_1, descriptors_1 = self.sift.detectAndCompute(self.img_1, None)
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
        matches = self.flann.knnMatch(des1, des2, k=3)
        better_matches = self.remove_best_self(matches)
        best_match = list()
        for m, n in better_matches:
            # threshold value 0.5 from paper
            if m.distance < 0.5 * n.distance:
                best_match.append([m, n])
        return best_match

    # print image
    def show_image(self, img_1, img_2):
        figure, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].imshow(img_1, cmap="gray")
        ax[1].imshow(img_2, cmap="gray")
        plt.show()
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

    # calculate best match
    best_match_value = model.features_matching(descriptors_1, descriptors_2)

    print(best_match_value)


if __name__ == "__main__":
    main()

