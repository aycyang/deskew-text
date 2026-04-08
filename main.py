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

def avgImg(a, b, ty = 0, tx = 0):
    ah, aw = a.shape
    bh, bw = b.shape
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
    c = np.zeros((max(ah + aty, bh + bty), max(aw + atx, bw + btx)), dtype=np.uint8)
    c[aty:aty+ah, atx:atx+aw] += a // 2
    c[bty:bty+bh, btx:btx+bw] += b // 2
    return c

def invertAndThreshold(img, threshold = 90):
    img = cv.bitwise_not(img)
    _, img = cv.threshold(img, threshold, 255, 0)
    return img

def boxConnectedComponents(img):
    N, markers = cv.connectedComponents(img, ltype=cv.CV_16U)
    boxes = []
    for i in range(1, N):
        points = np.argwhere(markers == i)
        boxes.append(cv.boundingRect(points))
    return boxes

def recognizeChar(img, box, charMap):
    y, x, h, w = box
    patch_grayscale = img[y:y+h, x:x+w]
    patch_thresh = invertAndThreshold(patch_grayscale)
    centroid, avgMag = calculateCentroidAndAverageMagnitudes(patch_thresh)
    if avgMag == 0:
        return None
    charMatches = []
    for charLabel, (char, charCentroid, charMag) in charMap.items():
        scaleFactor = charMag / avgMag
        if scaleFactor < 1 or 2 < scaleFactor:
            continue
        resized_patch = cv.resize(patch_grayscale, None, fx=scaleFactor, fy=scaleFactor, interpolation=cv.INTER_LINEAR)
        ty, tx = charCentroid - scaleFactor * centroid
        avg = avgImg(char, invertAndThreshold(resized_patch), int(ty), int(tx))
        hist, _ = np.histogram(avg, bins=3)
        _, numUnmatchedPixels, numMatchedPixels = hist
        matchPct = 2 * numMatchedPixels / (2 * numMatchedPixels + numUnmatchedPixels)
        if matchPct > 0.7:
            charMatches.append((matchPct, scaleFactor, charLabel, y - ty, x - tx))

    sortedCharMatches = list(reversed(sorted(charMatches)))
    if len(sortedCharMatches) == 0:
        return None
    matchPct, scaleFactor, label, y, x = sortedCharMatches[0]
    return scaleFactor, label, y, x

def findOptimalTransform(a, b, tr, sr):
    h, w = a.shape
    transforms = []
    for s in np.arange(1-sr, 1+sr, 0.002):
        for dy in range(2 * tr + 1):
            for dx in range(2 * tr + 1):
                patch = b
                patch = cv.resize(patch, None, fx=s, fy=s)
                patch = invertAndThreshold(patch)
                avg = avgImg(patch, a, dy - tr, dx - tr)
                hist, _ = np.histogram(avg, bins=3)
                _, numUnmatchedPixels, numMatchedPixels = hist
                matchPct = (2 * numMatchedPixels - numUnmatchedPixels) / avg.size
                tdx = (dx - tr) / s
                tdy = (dy - tr) / s
                transforms.append((matchPct, tdx, tdy, s))
    matchPct, dx, dy, s = list(reversed(sorted(transforms)))[0]
    return dx, dy, s

def compositeImg(base, img, x, y, s):
    img = cv.resize(img, None, fx=s, fy=s)
    h, w, _ = img.shape
    base[y:y+h, x:x+w] = img

def debug(img):
    cv.imshow("debug", img)
    cv.waitKey()

def main():
    parser = argparse.ArgumentParser(
        prog="deskew-text",
        description="Straighten and clean up scanned images of text")

    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)

    args = parser.parse_args()

    charMap = loadCharMap()

    img = cv.imread(args.input, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read"

    img = coarseDeskew(img)
    img_color = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

    boxes = boxConnectedComponents(invertAndThreshold(img))
    
    padding = 2
    boxes = list(map(lambda b: (b[0]-padding, b[1]-padding, b[2]+2*padding, b[3]+2*padding), boxes))

    for box in boxes:
        result = recognizeChar(img, box, charMap)
        if result is None:
            continue
        scaleFactor, label, ly, lx = result
        y, x, h, w = box
        cv.rectangle(img_color, (x, y), (x+w, y+h), (0, 0, 255), 1)
        # get char
        char, _, _ = charMap[label]
        charHeight, charWidth = char.shape

        # get patch around (lx, ly)
        pya = int(ly)
        pxa = int(lx)
        pyb = int(ly) + charHeight
        pxb = int(lx) + charWidth
        patch = img[pya:pyb, pxa:pxb]
        patch = cv.resize(patch, None, fx=scaleFactor, fy=scaleFactor)

        dx, dy, s = findOptimalTransform(char, patch, 2, .04)
        compositeImg(img_color, cv.cvtColor(char, cv.COLOR_GRAY2RGB),
            int(lx + dx), int(ly + dy), 1/scaleFactor)

    cv.imshow("debug", img_color)
    cv.waitKey()
    return

if __name__ == '__main__':
    main()
