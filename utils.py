import cv2
import numpy as np


def copyPixels(x_source,y_source,x_target,y_target,img_src,img_dst):
    for i in range(len(x_source)):
        img_dst[y_target[i]][x_target[i]] = img_src[y_source[i]][x_source[i]]
    return img_dst

def copyPixelsTPS(x_source,y_source,x_target,y_target,img_src,img_dst):
    for i in range(len(x_source)):
        img_dst[x_target[i]][y_target[i]] = img_src[x_source[i]][y_source[i]]
    return img_dst

def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True


def draw_rects(rects,img_orig):
    img = img_orig.copy()
    for r in rects:
        two_pts = [(r.rect.left(),r.rect.top()), (r.rect.right(),r.rect.bottom())]
        cv2.rectangle(img,two_pts[0],two_pts[1],(0,255,0),3)
    return img

def draw_rects_frontal(rects,img_orig):
    img = img_orig.copy()
    for r in rects:
        two_pts = [(r.left(),r.top()),(r.right(),r.bottom())]
        cv2.rectangle(img,two_pts[0],two_pts[1],(0,255,0),3)
    return img

def check_triangulation_order(img_src,triangleList_src,img_dst,triangleList_dst):
    
    for t_s,t_d in zip(triangleList_src,triangleList_dst) :
        img_src_copy = img_src.copy()
        pt1 = (int(t_s[0]), int(t_s[1]))
        pt2 = (int(t_s[2]), int(t_s[3]))
        pt3 = (int(t_s[4]), int(t_s[5]))

        cv2.line(img_src_copy, pt1, pt2, (255,0,0), 1, cv2.LINE_AA, 0)
        cv2.line(img_src_copy, pt2, pt3, (255,0,0), 1, cv2.LINE_AA, 0)
        cv2.line(img_src_copy, pt3, pt1, (255,0,0), 1, cv2.LINE_AA, 0)

        img_dst_copy = img_dst.copy()
        pt1d = (int(t_d[0]), int(t_d[1]))
        pt2d= (int(t_d[2]), int(t_d[3]))
        pt3d = (int(t_d[4]), int(t_d[5]))

        cv2.line(img_dst_copy, pt1d, pt2d, (0,0,255), 1, cv2.LINE_AA, 0)
        cv2.line(img_dst_copy, pt2d, pt3d, (0,0,255), 1, cv2.LINE_AA, 0)
        cv2.line(img_dst_copy, pt3d, pt1d, (0,0,255), 1, cv2.LINE_AA, 0)
        # cv2.imshow('src triangles',img_src_copy)
        # cv2.imshow('dst triangles',img_dst_copy)

        # cv2.waitKey(0)


def hull_masks(img_src,img_dst,final_shapes_src,final_shapes_dst):

    final_shapes_src = np.reshape(final_shapes_src, (68,2)).astype('int32')
    final_shapes_dst = np.reshape(final_shapes_dst, (68,2)).astype('int32')
    src_hull = cv2.convexHull(final_shapes_src, False)
    dst_hull = cv2.convexHull(final_shapes_dst, False)

    src_mask = np.zeros_like(img_src)
    src_mask = cv2.fillPoly(src_mask, [src_hull], color =(255,255,255))
    dst_mask = np.zeros_like(img_dst)
    dst_mask = cv2.fillPoly(dst_mask, [dst_hull], color =(255,255,255))
    
    r_src = cv2.boundingRect(src_hull)
    center_src = (r_src[0]+(r_src[2]//2), r_src[1]+(r_src[3]//2))

    r_dst = cv2.boundingRect(dst_hull)
    center_dst = (r_dst[0]+(r_dst[2]//2), r_dst[1]+(r_dst[3]//2))

    return src_hull,src_mask,center_src,dst_hull,dst_mask,center_dst