import numpy as np
import cv2
from utils import *

def get_delaunay_triangulation(img,final_shapes):
    size = img.shape
    rect = (0, 0, size[1], size[0])
    subdiv  = cv2.Subdiv2D(rect)
    pt_list = []
    j=0

    for i in range(len(final_shapes[0])):
        pt = (int(final_shapes[0][i][0]),int(final_shapes[0][i][1]))
        subdiv.insert(pt)

    return subdiv

def draw_delaunay(img_orig, subdiv, delaunay_color ) :

    img = img_orig.copy()
    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList :
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)

    return triangleList,img


def sort_triangles(triangleList_src, src_shapes, triangleList_dst, dst_shapes):
    dst_shapes = np.reshape(dst_shapes, (68,2))
    src_shapes = np.reshape(src_shapes, (68,2))
    new_triangles_src = []

    for d in triangleList_dst:
        pt1 = [d[0], d[1]]
        pt2 = [d[2], d[3]]
        pt3 = [d[4], d[5]]
        pt1_s, pt2_s, pt3_s = [], [], []
        
        for i in range(len(dst_shapes)):
            
            if dst_shapes[i][0] == d[0] and dst_shapes[i][1] == d[1]:
                pt1_s = src_shapes[i]
            elif  dst_shapes[i][0] == d[2] and dst_shapes[i][1] == d[3]:
                pt2_s = src_shapes[i]
            elif  dst_shapes[i][0] == d[4] and dst_shapes[i][1] == d[5]:
                pt3_s = src_shapes[i]
        if len(pt1_s) ==0 or len(pt2_s) == 0 or len(pt3_s) == 0:
            pass
        else:
            new = [pt1_s[0], pt1_s[1], pt2_s[0], pt2_s[1], pt3_s[0], pt3_s[1]]
            new_triangles_src.append(new)
    return new_triangles_src


def getBarycentricMatrix(trianglePts):
    B = np.array([[trianglePts[0],trianglePts[2],trianglePts[4]],
                  [trianglePts[1],trianglePts[3],trianglePts[5]],
                  [1,1,1]])
    return B



# Of a given triangle and its B matrix, it returns the points inside the triange and the barycentric coordinates
def calculateBarycentricCoords(img, B, trianglepts):

    pt1 = (int(trianglepts[0]), int(trianglepts[1]))
    pt2 = (int(trianglepts[2]), int(trianglepts[3]))
    pt3 = (int(trianglepts[4]), int(trianglepts[5]))
    x,y,w,h = cv2.boundingRect(np.array([pt1,pt2,pt3]))
    
    xx,yy = np.indices((w, h))
    xx = xx + x
    yy = yy + y

    B_inv = np.linalg.inv(B)
    bary_matrix = []

    x_valid = []
    y_valid = []
    bary_valid = []
    for x,y in zip(xx.ravel(),yy.ravel()):
        img_coord = np.reshape([x,y,1],(3,1))
        bary_coord = np.matmul(B_inv,img_coord)

        bary_matrix.append(bary_coord)

        # Filter to keep valid barycentric coordinates
        if(bary_coord[0]>0 and bary_coord[0]<=1 and bary_coord[1]>0 and bary_coord[1]<=1 and \
            bary_coord[2]>0 and bary_coord[2]<=1 and 0.999999<=bary_coord[0]+bary_coord[1]+bary_coord[2]<=1.000001):
            x_valid.append(x)
            y_valid.append(y)
            bary_valid.append(bary_coord)
    
    return x_valid, y_valid, bary_valid 


def getSourceLocations(A, bary_coords,img_src):
    x_source = []
    y_source = []
    for bary_coord in bary_coords:
        cart_coord = np.matmul(A,bary_coord)
        x = int(cart_coord[0]/cart_coord[2])
        y = int(cart_coord[1]/cart_coord[2])
        x_source.append(x)
        y_source.append(y)

    return x_source,y_source


def TRI(img_src,img_dst,final_shapes_src,final_shapes_dst):
    subdiv = get_delaunay_triangulation(img_src,final_shapes_src)
    triangleList_src, delaunay_img = draw_delaunay(img_src, subdiv, (255,255,255))
    cv2.imwrite('triangles_src.jpg',delaunay_img)
    subdiv = get_delaunay_triangulation(img_dst,final_shapes_dst)
    triangleList_dst, delaunay_img = draw_delaunay(img_dst, subdiv, (255,255,255))
    cv2.imwrite('triangles_dst.jpg',delaunay_img)

    triangleList_src = sort_triangles(triangleList_src, final_shapes_src, triangleList_dst, final_shapes_dst)
    img_dst_copy = img_dst.copy()
    # Execute this process for every triangle:
    for i in range(len(triangleList_dst)):
        B = getBarycentricMatrix(triangleList_dst[i])
        x_target, y_target, bary_valid= calculateBarycentricCoords(img_dst,B,triangleList_dst[i])
        A = getBarycentricMatrix(triangleList_src[i])
        x_source,y_source = getSourceLocations(A, bary_valid,img_src)
        img_dst_copy = copyPixels(x_source,y_source,x_target,y_target,img_src,img_dst_copy)

    return img_dst_copy