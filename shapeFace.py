import cv2
import os
import time
import collections


class Image:

    def __init__(self, path):
        self.id = path.split('-')[0].split('/')[2]
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
        self.createListForVerticies()
        self.setDirPath()

    def setDirPath(self):
        self.dirPath = './shaped'
        if os.path.isdir(self.dirPath):
            return
        else:
            os.mkdir(self.dirPath)
            return

    def createListForVerticies(self):
        self.listForVerticies = list()
        for i in range(0, 4):
            self.listForVerticies.append(list())

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

    def saveCutFaces(self):
        i = 0
        for imageObj in self.imageObjs:
            verticies = self.useHaarcascade(imageObj)
            self.writeVerticies(verticies)
            for vertex in verticies:
                if vertex is False:
                    continue
                elif len(vertex) == 1:
                    continue
                else:
                    self.saveFace(i, imageObj.image[
                                  vertex[1] - 30:vertex[3] + 30, vertex[0] - 30:vertex[2] + 30])
                    break
            i += 1
        self.writeStatistics()

    def saveFace(self, num, image):
        path = self.dirPath + '/' + str(num) + '.jpg'
        print(path)
        cv2.imwrite(path, image)

    def useHaarcascade(self, imageObj):
        images = imageObj.copyImage()
        i = 0
        verticies = list()
        for image in images:
            verticies.append(self.applyHaarcascade(image, i))
            i += 1
        count_dict = collections.Counter(verticies)
        if count_dict[False] == 4:
            self.writeError(imageObj.id)
            return imageObj.id
        else:
            return verticies

    def writeError(self, num):
        with open('error.log', 'a') as f:
            f.write(str(num) + '\n')

    def applyHaarcascade(self, image, num):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = self.face_cascades[num]
        faces = face_cascade.detectMultiScale(gray)
        if len(faces) == 1:
            x, y, w, h = faces[0]
            print(x, y, w, h)
            if w * h > 8000:
                w += x
                h += y
                self.listForVerticies[num].append((x, y, w, h))
                return x, y, w, h
            else:
                return False
        else:
            return False

    def writeVerticies(self, verticies):
        with open('verticies.log', 'a') as f:
            for data in verticies:
                f.write(str(data) + '\n')
            f.write('\n')

    def displaySummation(self):
        print('----------------')
        time.sleep(10)
        for lines in self.listForVerticies:
            print('-----------------')
            for line in lines:
                print(line)

    def writeStatistics(self):
        with open('verticies.log', 'a') as f:
            for lines in self.listForVerticies:
                ave = self.getAverage(lines)
                print('The Number Of Valid Data: ' + str(len(lines)) + '\n')
                print('The Average Of Each Vertex: ' + str(ave) + '\n')
                f.write('The Number Of Valid Data: ' + str(len(lines)) + '\n')
                f.write('The Average Of Each Vertex: ' + str(ave) + '\n')
                f.write('--------------------------------------------')

    def getAverage(self, lines):
        sumX = sumY = sumW = sumH = 0
        for line in lines:
            sumX += line[0]
            sumY += line[1]
            sumW += line[2]
            sumH += line[3]
        aveX = sumX / len(lines)
        aveY = sumY / len(lines)
        aveW = sumW / len(lines)
        aveH = sumH / len(lines)
        print(aveX, aveY, aveW, aveH)

        return aveX, aveY, aveW, aveH


if __name__ == '__main__':
    shaper = ShapeFace()
    shaper.saveCutFaces()
