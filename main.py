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

def coarseDeskew(img):
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

    return rotated

def loadCharMap():
    img = cv.imread("charset.png", cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_NEAREST)
    imgHeight, imgWidth = img.shape
    charWidth = imgWidth // 16
    charHeight = imgHeight // 6
    charMap = {}
    for i in range(0x20, 0x80):
        x = (i & 0xf) * charWidth
        y = ((i >> 4) - 2) * charHeight
        char = img[y:y+charHeight, x:x+charWidth]
        centroid, avgMag = calculateCentroidAndAverageMagnitudes(char)
        if avgMag == 0:
            continue
        charMap[chr(i)] = (char, centroid, avgMag)
    return charMap

def calculateCentroidAndAverageMagnitudes(img):
    whitePoints = np.argwhere(img)
    n, _ = whitePoints.shape
    if n == 0:
        return np.array([0, 0]), 0
    centroid = np.sum(whitePoints, axis=0) / n
    mags = np.sqrt(np.sum(np.pow(whitePoints - centroid, 2), axis=1))
    avgMag = np.sum(mags) / n
    return centroid, avgMag

def avgImg(a, b, tx = 0, ty = 0):
    aw, ah = a.shape
    bw, bh = b.shape
    atx = 0
    aty = 0
    btx = 0
    bty = 0
    if tx < 0:
        atx = -tx
    else:
        btx = tx
    if ty < 0:
        aty = -ty
    else:
        bty = ty
    c = np.zeros((max(aw + atx, bw + btx), max(ah + aty, bh + bty)), dtype=np.uint8)
    c[atx:atx+aw, aty:aty+ah] += a // 2
    c[btx:btx+bw, bty:bty+bh] += b // 2
    return c

def main():
    parser = argparse.ArgumentParser(
        prog="deskew-text",
        description="Straighten and clean up scanned images of text")

    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)

    args = parser.parse_args()

    charMap = loadCharMap()

    orig = cv.imread(args.input, cv.IMREAD_GRAYSCALE)
    assert orig is not None, "file could not be read"

    deskew1 = coarseDeskew(orig)
    deskew1_color = cv.cvtColor(deskew1, cv.COLOR_GRAY2RGB)

    _, deskew1_thresh = cv.threshold(deskew1, 180, 255, 0)
    deskew1_thresh = cv.bitwise_not(deskew1_thresh)
    N, markers = cv.connectedComponents(deskew1_thresh, ltype=cv.CV_16U)
    classifications = [None] * N
    for i in range(1, N):
        points = np.argwhere(markers == i)
        x, y, w, h = cv.boundingRect(points)
        patch_grayscale = deskew1[x:x+w, y:y+h]
        patch_thresh = deskew1_thresh[x:x+w, y:y+h]
        centroid, avgMag = calculateCentroidAndAverageMagnitudes(patch_thresh)
        if avgMag == 0:
            continue
        charMatches = []
        for charLabel, (char, charCentroid, charMag) in charMap.items():
            scaleFactor = charMag / avgMag
            if scaleFactor > 2 or scaleFactor < 1:
                continue
            resized_patch = cv.resize(patch_grayscale, None, fx=scaleFactor, fy=scaleFactor, interpolation=cv.INTER_LINEAR)
            _, resized_patch_thresh = cv.threshold(resized_patch, 180, 255, 0)
            resized_patch_thresh = cv.bitwise_not(resized_patch_thresh)
            tx, ty = charCentroid - scaleFactor * centroid
            avg = avgImg(char, resized_patch_thresh, int(tx), int(ty))
            hist, _ = np.histogram(avg, bins=3)
            _, numUnmatchedPixels, numMatchedPixels = hist
            matchPct = 2 * numMatchedPixels / (2 * numMatchedPixels + numUnmatchedPixels)
            if matchPct > 0.7:
                charMatches.append((matchPct, scaleFactor, charLabel, x - tx, y - ty))

        sortedCharMatches = list(reversed(sorted(charMatches)))
        if len(sortedCharMatches) > 0:
            bestMatch = sortedCharMatches[0]
            charLabel, cx, cy = bestMatch[2:]
            classifications[i] = bestMatch
            cv.putText(deskew1_color, str(charLabel), (int(cy), int(cx)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv.rectangle(deskew1_color, (y, x), (y+h, x+w), (0, 0, 255), 2)

    #drawGrid(img)
    cv.imshow("debug", deskew1_color)
    cv.waitKey()
    return

if __name__ == '__main__':
    main()
