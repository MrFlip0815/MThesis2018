import numpy as np


# gets an Image, performs Canny Edge Detection and returns the result Image
def PerformEdgeDetection(image):
	return cv2.Canny(image,CANNY_A,CANNY_B)

def CalculateNoseAngle(shape):
	xlist = []
	ylist = []

	for i in range(0,68):
		xlist.append(shape.item(i,0))
		ylist.append(shape.item(i,1))

	xmean = np.mean(xlist)
	ymean = np.mean(ylist)

	xcentral = [(x-xmean) for x in xlist] #Calculate distance centre <-> other points in both axes
	ycentral = [(y-ymean) for y in ylist]

	noseangle = 0
	#noseangle = int(np.arctan((ylist[27]-ylist[30])/(xlist[30]-xlist[27]))*180/np.pi)

	if xlist[30] == xlist[27]:
		noseangle = 0
	else:
		noseangle = np.arcsin((xlist[30]-xlist[27])/(ylist[27]-ylist[30]))*180/np.pi

	return noseangle

def interpolate_pixels_along_line(x0, y0, x1, y1):
    """Uses Xiaolin Wu's line algorithm to interpolate all of the pixels along a
    straight line, given two points (x0, y0) and (x1, y1)

    Wikipedia article containing pseudo code that function was based off of:
        http://en.wikipedia.org/wiki/Xiaolin_Wu's_line_algorithm
    """
    pixels = []
    steep = abs(y1 - y0) > abs(x1 - x0)

    # Ensure that the path to be interpolated is shallow and from left to right
    if steep:
        t = x0
        x0 = y0
        y0 = t

        t = x1
        x1 = y1
        y1 = t

    if x0 > x1:
        t = x0
        x0 = x1
        x1 = t

        t = y0
        y0 = y1
        y1 = t

    dx = x1 - x0
    dy = y1 - y0
    gradient = dy / dx  # slope

    # Get the first given coordinate and add it to the return list
    x_end = round(x0)
    y_end = y0 + (gradient * (x_end - x0))
    xpxl0 = x_end
    ypxl0 = round(y_end)
    if steep:
        pixels.extend([(ypxl0, xpxl0), (ypxl0 + 1, xpxl0)])
    else:
        pixels.extend([(xpxl0, ypxl0), (xpxl0, ypxl0 + 1)])

    interpolated_y = y_end + gradient

    # Get the second given coordinate to give the main loop a range
    x_end = round(x1)
    y_end = y1 + (gradient * (x_end - x1))
    xpxl1 = x_end
    ypxl1 = round(y_end)

    # Loop between the first x coordinate and the second x coordinate, interpolating the y coordinates
    for x in range(xpxl0 + 1, xpxl1):
        if steep:
            pixels.extend([(math.floor(interpolated_y), x), (math.floor(interpolated_y) + 1, x)])

        else:
            pixels.extend([(x, math.floor(interpolated_y)), (x, math.floor(interpolated_y) + 1)])

        interpolated_y += gradient

    # Add the second given coordinate to the given list
    if steep:
        pixels.extend([(ypxl1, xpxl1), (ypxl1 + 1, xpxl1)])
    else:
        pixels.extend([(xpxl1, ypxl1), (xpxl1, ypxl1 + 1)])

    return pixels


# give landmarks and unprocessed image (!) - gets ToH ToupleList
def GetTopOfHeadValues(landmarks,image):

	_getDistance = lambda a,b: b - a if b > a else a - b 
	# Berechnung der Verschiebung von X durch Winkel noseangle
	_CalculateEndpoint = lambda x_start, y_start, noseangle: x_start + y_start*np.tan(noseangle*np.pi/180)
	# make sure it doesn't get out of image border
	_f = lambda x: int(x) if x > 0 else 0

	noseangle = CalculateNoseAngle(landmarks)

	StartX = round((landmarks.item(22,0)+landmarks.item(21,0))/2)
	StartY = round((landmarks.item(22,1)+landmarks.item(21,1))/2)
	endPoint = round(_CalculateEndpoint(StartX,StartY,noseangle))
	# 12 is border from top - don't count until there, image set CGNET is broken
	pixels = interpolate_pixels_along_line(StartX,StartY,int(endPoint),TOP_BORDER_CG)

	result = [p for p in pixels if edgesImageCopy[p[1],p[0]] == 255]

	topOfHeadCoords = tuple([ int(sum(y) / len(y)) for y in zip(*result)])
	
	# Distance 8<->27 * 0.8 as minimum value DO NOT USE UNLESS ABSOLUTELY NECCESSARY; CORRELATION BETWEEN FEATURES
	ToH_YMin = (int(_CalculateEndpoint(StartX,StartY,noseangle)), _f(((StartY-_getDistance(landmarks.item(8,1),landmarks.item(27,1))*0.85))))
	ToH_YMax = (int(_CalculateEndpoint(StartX,StartY,noseangle)), _f(((StartY-_getDistance(landmarks.item(8,1),landmarks.item(27,1))*1.05))))
	# Min Y value in Touple Result set ( closest to top window border) is chosen as Max - can be lower than min actually
	ToH_YMaxEdge = (int(_CalculateEndpoint(StartX,StartY,noseangle)), min(result, default=(0,0), key=lambda  item:item[1] )[1])
	print(ToH_YMax, ToH_YMaxEdge)
	# Return Values

	def average(xs):
		N = float(len(xs))
		return tuple(int(sum(col)/N) for col in zip(*xs))
	
	# panic
	if topOfHeadCoords == () :
		topOfHeadCoords = average([ToH_YMin,ToH_YMax])
	#has to be lower than max edge
	# bad edge detection, should be lower than min in all cases
	if (ToH_YMaxEdge[1] > ToH_YMin[1]):
		ToH_YMaxEdge = ToH_YMin
	ToH_final = average([ToH_YMaxEdge,ToH_YMin,topOfHeadCoords])

	if ToH_final[1] < ToH_YMaxEdge[1]:
		ToH_final = ToH_YMaxEdge

	return topOfHeadCoords,ToH_YMin,ToH_YMax,ToH_YMaxEdge,ToH_final