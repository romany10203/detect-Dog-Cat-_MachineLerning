
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.utils import shuffle
import os
from PIL import Image
import glob
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure



categorys = []
categorys_test = []
hog_imgs = []
hog_img_test = []



for filename in glob.glob('train/*'):
    img=Image.open(filename)
    img = img.resize((128,64))
    fd,hog_imgs = hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=True)
    hog_imgs.append(fd)

for filename in glob.glob('test/*'):
    img=Image.open(filename)
    img = img.resize((128,64))
    fd,hog_imgs = hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=True)
    hog_img_test.append(fd)


filenames = os.listdir('train')
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'cat':
        categorys.append(str(0))
    else:
        categorys.append(str(1))


filenames = os.listdir('test')
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'cat':
        categorys_test.append(str(0))
    else:
        categorys_test.append(str(1))

hog_imgs, categorys = shuffle(hog_imgs, categorys)
hog_img_test, categorys_test = shuffle(hog_img_test, categorys_test)

C = 0.001  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(hog_imgs, categorys)
lin_svc = svm.LinearSVC(C=C).fit(hog_imgs, categorys)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.8, C=C).fit(hog_imgs, categorys)
poly_svc = svm.SVC(kernel='poly', degree=8, C=C).fit(hog_imgs, categorys)

for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    predictions = clf.predict(hog_imgs)
    accuracy = np.mean(predictions == categorys)
    print(accuracy)


for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    predictions = clf.predict(hog_img_test)
    accuracy = np.mean(predictions == categorys_test)
    print(accuracy)



