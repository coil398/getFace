import cv2
import os


class Image:

    def __init__(self, path):
        self.image = cv2.imread(path)

    def copyImage(self):
        images = list()
        images.append(self.image)
        for i in range(0, 3):
            images.append(self.image.copy())
        return images


class ShapeFace:

    def __init__(self):
        self.haarcascadesDir = './haarcascades'
        self.prepareHaarcascades()
        self.imageObjs = self.prepareImages()

    def prepareHaarcascades(self):
        cascade_files = self.haarcascadesAppender()
        self.face_cascades = list()
        for f in cascade_files:
            self.face_cascades.append(cv2.CascadeClassifier(f))

    def haarcascadesAppender(self):
        haarcascades = list()
        haarcascades.append(self.haarcascadesDir +
                            '/haarcascade_frontalface_default.xml')
        haarcascades.append(self.haarcascadesDir +
                            '/haarcascade_frontalface_alt.xml')
        haarcascades.append(self.haarcascadesDir +
                            '/haarcascade_frontalface_alt2.xml')
        haarcascades.append(self.haarcascadesDir +
                            '/haarcascade_frontalface_alt_tree.xml')
        return haarcascades

    def prepareImages(self):
        paths = self.getImgPaths()
        imageObjs = list()
        for path in paths:
            imageObjs.append(Image(path))
        return imageObjs

    def getImgPaths(self):
        paths = list()
        files = os.listdir('./talentData')
        for f in files:
            paths.append('./talentData/' + f)
        return paths

    def getFaces(self):
        for imageObj in self.imageObjs:
            verticies = self.useHaarcascade(imageObj)
            # details = getDetailOfVerticies(verticies)
            self.writeVerticies(verticies)

    def useHaarcascade(self, imageObj):
        images = imageObj.copyImage()
        i = 0
        verticies = list()
        for image in images:
            verticies.append(self.applyHaarcascade(image, i))
            i += 1
        return verticies

    def applyHaarcascade(self, image, num):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = self.face_cascades[num]
        faces = face_cascade.detectMultiScale(gray)
        if len(faces) == 1:
            return faces[0][0], faces[0][1], faces[0][0] + faces[0][2], faces[0][1] + faces[0][3]
        else:
            return False

    def getDetailOfVerticies(self, verticies):
        pass

    def writeVerticies(self, verticies):
        with open('verticies.log', 'a') as f:
            for data in verticies:
                print(data)
                f.write(str(data) + '\n')
            f.write('\n')


if __name__ == '__main__':
    shaper = ShapeFace()
    shaper.getFaces()
