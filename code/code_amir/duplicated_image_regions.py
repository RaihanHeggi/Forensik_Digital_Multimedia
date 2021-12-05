from skimage.metrics import structural_similarity

import imutils
import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1007, 652)
        Dialog.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.filename = QtWidgets.QLineEdit(Dialog)
        self.filename.setGeometry(QtCore.QRect(20, 130, 511, 22))
        self.filename.setObjectName("filename")
        self.browse = QtWidgets.QPushButton(Dialog)
        self.browse.setGeometry(QtCore.QRect(550, 130, 150, 28))
        self.browse.setObjectName("browse")
        self.filename2 = QtWidgets.QLineEdit(Dialog)
        self.filename2.setGeometry(QtCore.QRect(20, 170, 511, 21))
        self.filename2.setObjectName("filename2")
        self.browse2 = QtWidgets.QPushButton(Dialog)
        self.browse2.setGeometry(QtCore.QRect(550, 170, 150, 28))
        self.browse2.setObjectName("browse2")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(30, 230, 181, 41))
        self.pushButton.setObjectName("pushButton")

        self.browse.clicked.connect(self.onclick_Pilih)
        self.browse2.clicked.connect(self.onclick_Pilih2)
        self.pushButton.clicked.connect(self.onclick_Proses)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Duplicated Image Regions"))
        self.browse.setText(_translate("Dialog", "Browse File Original"))
        self.browse2.setText(_translate("Dialog", "Browse File Fake"))
        self.pushButton.setText(_translate("Dialog", "Lihat Hasil"))

    def onclick_Pilih(self):
        image = QFileDialog.getOpenFileName(None, 'OpenFile', '', "Image file(*.jpg)")
        #         imagePath = image[0]
        #         pixmap = QPixmap(imagePath)
        #         self.fotoAsli.setPixmap(pixmap)

        self.filename.setText(image[0])

    def onclick_Pilih2(self):
        image2 = QFileDialog.getOpenFileName(None, 'OpenFile', '', "Image file(*.jpg)")
        self.filename2.setText(image2[0])

    def onclick_Proses(self):
        imageA = cv2.imread(self.filename.text())
        imageB = cv2.imread(self.filename2.text())
        dim = (500, 500)

        # convertion en gris
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

        # Calcule de 'Structural Similarity '(SSIM)
        # images, assure le retour
        (score, diff) = structural_similarity(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        print("indice de simularité SSIM: {}".format(score))

        # detection de contour
        # obtain the regions of the two input images that differ
        thresh = cv2.threshold(diff, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        for c in cnts:
            # dessiner un rectangle rouge autour
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # afficher les images
        cv2.imshow("Original", imageA)
        cv2.imshow("Modified", imageB)
        cv2.imshow("Diff", diff)
        cv2.imshow("Thresh", thresh)
        cv2.waitKey(0)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
