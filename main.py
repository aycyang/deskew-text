import cv2 as cv
import numpy as np
import argparse
import math

def drawGrid(img, hRes = 32, vRes = 32, stroke = 1):
    rows, cols = img.shape
    for i in range(vRes):
        y = rows // vRes * i
        cv.line(img, (0, y), (cols, y), (160, 0, 0), stroke)
    for i in range(hRes):
        x = cols // hRes * i
        cv.line(img, (x, 0), (x, rows), (160, 0, 0), stroke)

def calculateStraightness(img):
    preprocessed = img
    #preprocessed = cv.blur(preprocessed, (3, 3))

    #gradX = cv.Sobel(preprocessed, cv.CV_64F, 1, 0, ksize=1)
    #gradY = cv.Sobel(preprocessed, cv.CV_64F, 0, 1, ksize=1)
    gradX = cv.Scharr(preprocessed, cv.CV_64F, 1, 0)
    gradY = cv.Scharr(preprocessed, cv.CV_64F, 0, 1)
    #gradX = cv.Laplacian(preprocessed, cv.CV_64F)
    #gradY = cv.Laplacian(preprocessed, cv.CV_64F)

    # Sum of change along X and Y axes
    dX = np.sum(np.abs(gradX), axis=0)
    dY = np.sum(np.abs(gradY), axis=1)

    # Sum of squares of dX and dY
    # Intuitively, this should peak when there are a few rows/cols that change a lot
    # If the text lines up to the grid perfectly, we should see a spike in these values
    varX = np.sum(np.pow(dX, 2))
    varY = np.sum(np.pow(dY, 2))

    return varX, varY

def rotateImg(img, angle):
    rows, cols = img.shape
    center = ((cols-1)/2.0, (rows-1)/2.0)
    M = cv.getRotationMatrix2D(center, angle, 1)
    return cv.warpAffine(img, M, (cols, rows), borderValue=(255, 255, 255))

def cropImg(img):
    _, img = cv.threshold(img, 162, 255, 0)
    img = cv.bitwise_not(img)
    whitePoints = np.argwhere(img)
    x, y, w, h = cv.boundingRect(whitePoints)
    img = img[x:x+w, y:y+h]
    img = cv.bitwise_not(img)
    return img

def main():
    parser = argparse.ArgumentParser(
        prog="deskew-text",
        description="Straighten and clean up scanned images of text")

    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)

    args = parser.parse_args()

    img = cv.imread(args.input, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read"

    # calculate straightness scores for a range of rotation adjustments
    angles = []
    maxScore = -math.inf
    bestAngle = 0
    for angle in np.arange(-10, 10, 0.1):
        rotated = rotateImg(img, angle)
        varX, varY = calculateStraightness(rotated)
        score = varX + varY
        angles.append((score, angle))

    # find angles with top N scores and use the median of these angles
    angles.sort(reverse=True)
    bestAngles = []
    N = 7
    for _, angle in angles[:N]:
        bestAngles.append(angle)
    bestAngles.sort()
    bestAngle = bestAngles[N//2]

    rotated = rotateImg(img, bestAngle)

    cropped = cropImg(rotated)

    cv.imwrite(args.output, cropped)

if __name__ == '__main__':
    main()
