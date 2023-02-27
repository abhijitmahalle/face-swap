#!/usr/bin/python3
import cv2
import numpy as np
from api import PRN
from utils.render import render_texture
import os

def prnet_one_face_main(prn, img_src, img_dst):
    src_faces = prn.dlib_detect(img_src)
    dst_faces = prn.dlib_detect(img_dst)
    print(len(src_faces), len(dst_faces))
    if len(src_faces)<1 or len(dst_faces)<1:
        print("Only found one or less face. Exiting...")
        return None

    print("processing source image")
    pos_src = prn.process(img_src)
    img_src = img_src/255.0
    texture_src = cv2.remap(img_src, pos_src[:,:,:2].astype(np.float32), None,\
                            interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
    new_texture = texture_src

    print("processing destination image")
    [dst_h, dst_w, _] = img_dst.shape
    pos_dst = prn.process(img_dst)
    vertices_dst = prn.get_vertices(pos_dst)
    img_dst = img_dst/255.0
    print("Swapping")

    vis_colors = np.ones((vertices_dst.shape[0], 1))
    face_mask = render_texture(vertices_dst.T, vis_colors.T, prn.triangles.T, dst_h, dst_w, c=1)
    face_mask = np.squeeze(face_mask > 0).astype(np.float32)

    new_colors = prn.get_colors_from_texture(new_texture)
    new_image = render_texture(vertices_dst.T, new_colors.T, prn.triangles.T, dst_h, dst_w, c=3)
    new_image = img_dst*(1- face_mask[:,:,np.newaxis]) + new_image*face_mask[:,:,np.newaxis]

    print("Blending")
    # Possion Editing for blending image
    vis_ind = np.argwhere(face_mask>0)
    vis_min = np.min(vis_ind, 0)
    vis_max = np.max(vis_ind, 0)
    center = (int((vis_min[1] + vis_max[1])/2+0.5), int((vis_min[0] + vis_max[0])/2+0.5))
    output = cv2.seamlessClone((new_image*255).astype(np.uint8), (img_dst*255).astype(np.uint8), (face_mask*255).astype(np.uint8), center, cv2.NORMAL_CLONE)
    return output 

def prnet_two_faces_main(prn, img_src, img_dst, idx1, idx2):

    if idx1!=idx2:
        faces = prn.dlib_detect(img_src)
        print(len(faces))
        if len(faces)<2:
            print("less than two faces found in an image. exiting...")
            return None
    else:
        src_faces = prn.dlib_detect(img_src)
        dst_faces = prn.dlib_detetct(img_dst)
        if len(src_faces)<1 or len(dst_faces)<1:
            print("Only found one or less face. Exiting...")
            return None
    print("processing source image")

    [src_h, src_w, _] = img_src.shape

    pos_src = prn.process(img_src, index=idx1)
    vertices_src = prn.get_vertices(pos_src)
    img_src = img_src/255.0
    texture_src = cv2.remap(img_src, pos_src[:,:,:2].astype(np.float32), None,\
                            interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
    new_texture = texture_src

    print("processing destination image")

    [dst_h, dst_w, _] = img_dst.shape

    pos_dst = prn.process(img_dst, index=idx2)
    vertices_dst = prn.get_vertices(pos_dst)
    img_dst = img_dst/255.0
    texture_dst = cv2.remap(img_dst, pos_dst[:,:,:2].astype(np.float32), None, \
                            interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))

    print("Swapping")
    # Swapping
    vis_colors = np.ones((vertices_dst.shape[0], 1))
    face_mask = render_texture(vertices_dst.T, vis_colors.T, prn.triangles.T, dst_h, dst_w, c=1)
    face_mask = np.squeeze(face_mask > 0).astype(np.float32)

    new_colors = prn.get_colors_from_texture(new_texture)
    new_image = render_texture(vertices_dst.T, new_colors.T, prn.triangles.T, dst_h, dst_w, c=3)

    img_out = img_dst*(1- face_mask[:,:,np.newaxis]) + new_image*face_mask[:,:,np.newaxis]
    print("Blending")
    # Possion Editing for blending image
    vis_ind = np.argwhere(face_mask>0)
    vis_min = np.min(vis_ind, 0)
    vis_max = np.max(vis_ind, 0)
    center = (int((vis_min[1] + vis_max[1])/2+0.5), int((vis_min[0] + vis_max[0])/2+0.5))
    img_out = cv2.seamlessClone((img_out*255).astype(np.uint8), (img_dst*255).astype(np.uint8), (face_mask*255).astype(np.uint8), center, cv2.NORMAL_CLONE)

    img_out = img_out/255.0
    img_out_copy = img_out.copy()

    # Swapping
    print("Swapping")
    new_texture = texture_dst
    vis_colors = np.ones((vertices_src.shape[0], 1))
    face_mask = render_texture(vertices_src.T, vis_colors.T, prn.triangles.T, src_h, src_w, c=1)
    face_mask = np.squeeze(face_mask > 0).astype(np.float32)

    new_colors = prn.get_colors_from_texture(new_texture)
    new_image = render_texture(vertices_src.T, new_colors.T, prn.triangles.T, src_h, src_w, c=3)
    img_out = img_out_copy*(1- face_mask[:,:,np.newaxis]) + new_image*face_mask[:,:,np.newaxis]

    print("Blending")
    # Possion Editing for blending image
    vis_ind = np.argwhere(face_mask>0)
    vis_min = np.min(vis_ind, 0)
    vis_max = np.max(vis_ind, 0)
    center = (int((vis_min[1] + vis_max[1])/2+0.5), int((vis_min[0] + vis_max[0])/2+0.5))
    output = cv2.seamlessClone((img_out*255).astype(np.uint8), (img_out_copy*255).astype(np.uint8), (face_mask*255).astype(np.uint8), center, cv2.NORMAL_CLONE)

    return output

def parse_input_types(input1,input2):
    video_formats = ['mp4','avi','mov']
    image_formats = ['jpg','jpeg','png']
    input1_format = None
    input2_format = None
    if input1 is not None:
        input1_format = input1.split('.')[-1]
    if input2 is not None:
        input2_format = input2.split('.')[-1]

    input_formats = [input1_format,input2_format]
    input_types = [None,None]
    for i in range(len(input_formats)):
        if(input_formats[i] in image_formats):
            input_types[i] = 'img'
        elif input_formats[i] in video_formats:
            input_types[i] = 'vid'

    if input_types[0]=='img' and input_types[1]=='img':
        if input1==input2:
            swap_logic ="swap_two_faces_img"
        else:
            swap_logic ="swap_two_imgs" #Swap one face
    elif (input_types[0]=='vid' and input_types[1]=='img') :
        swap_logic = "swap_img_in_vid" #Swap one face (inside video)
    elif (input_types[0]=='vid' and input_types[1] is None):
        swap_logic = "swap_within_frame" # Swap two faces 
    else:
        swap_logic = None

    return swap_logic

if __name__=="__main__": 

    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    prnet = PRN(is_dlib=True)
    print("---------------------")

    input1 = '../data/TestSet/Test3.mp4'
    input2 = '../data/TestSet/Scarlett.jpg'

    output_name = 'testset3/frame'

    swap_logic = parse_input_types(input1,input2)
    SAVE = True
    print(swap_logic)
    if(swap_logic=="swap_within_frame"):
        cap = cv2.VideoCapture(input1)
        i = 1
        while(True):
            _i = format(i,'03')
            ret,img_src = cap.read()
            if ret:
                print(i)
                img_dst = img_src.copy()
                out_img = prnet_two_faces_main(prnet, img_src, img_dst, 0, 1)
                if out_img is not None and SAVE:
                    cv2.imwrite("../data/outputs/"+output_name+str(_i)+".jpg", out_img)
            else:
                print("Video completed")
                break
            i+=1
            
    elif(swap_logic=="swap_img_in_vid"):
        cap = cv2.VideoCapture(input1)
        img_src = cv2.imread(input2)
        i = 1
        while(True):
            _i = format(i,'03')
            ret,img_dst = cap.read()
            if ret:
                print(i)
                out_img = prnet_one_face_main(prnet, img_src, img_dst)
                if out_img is not None and SAVE:
                    cv2.imwrite("../data/outputs/"+output_name+str(_i)+".jpg", out_img)
            else:
                print("Video completed")
                break
            i+=1
            
    elif(swap_logic=="swap_two_imgs"):
        img_src = cv2.imread(input1)
        img_dst = cv2.imread(input2)
        cv2.imshow("src", img_src)
        cv2.imshow("dst", img_dst)
        out_img = prnet_one_face_main(prnet, img_src, img_dst)
        cv2.imshow("output", out_img)
        cv2.waitKey(0)
    elif(swap_logic=="swap_two_faces_img"):
        img_src = cv2.imread(input1)
        img_dst = cv2.imread(input2)
        cv2.imshow("src", img_src)
        cv2.imshow("dst", img_dst)
        out_img = prnet_two_faces_main(prnet, img_src, img_dst, 0, 1)
        cv2.imshow("output", out_img)
        cv2.waitKey(0)
    else:
        print("Invalid input formats. Please ensure input2 is an image if input 1 is an img, and an image or None if input1 is a video")
        exit()