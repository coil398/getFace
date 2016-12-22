import cv2
import os


def initialize(path):
    if os.path.exists(path):
        return
    else:
        os.mkdir(path)


def getFilesInDir():
    path = './shaped'
    return os.listdir(path)


def resizer(image):
    print(image)
    if image is None:
        return
    else:
        resized = cv2.resize(image, (112, 112))
        return resized


def saveImage(num, dirPath, image):
    path = dirPath + '/' + str(num) + '.jpg'
    cv2.imwrite(path, image)


if __name__ == '__main__':
    dirForResized = './cvted'
    initialize(dirForResized)
    files = getFilesInDir()
    i = 0
    for f in files:
        f = './shaped/' + f
        print('f: ' + f)
        image = cv2.imread(f)
        if image is None:
            continue
        else:
            resized = resizer(image)
            grayResized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            saveImage(i, dirForResized, grayResized)
        i += 1
