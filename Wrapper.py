#!/usr/bin/python3
import cv2
import numpy as np
import dlib
import os 
from tri import TRI
from TPS import TPS
from utils import hull_masks

def get_facial_landmarks(img,rects):
    landmark_detector = dlib.shape_predictor("./packages/shape_predictor_68_face_landmarks.dat")
    final_shapes = []
    for rect in rects:
        shape = landmark_detector(img,rect.rect)
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        final_shapes.append(shape_np)
    return final_shapes

def draw_facial_landmarks(img_orig,final_shapes):
    img = img_orig.copy()

    count = 1
    for shape in final_shapes:
        for coord in shape:
            cv2.circle(img,tuple(coord),2,(0,0,255),-1)
            count += 1
    return img

def get_faces(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dnnFaceDetector = dlib.cnn_face_detection_model_v1("./packages/mmod_human_face_detector.dat")
    rects = dnnFaceDetector(img_gray, 1)
    return rects

def SwapOneFace(img_src,img_dst,method,swap_logic,resize=False):
    if resize:
        img_src = cv2.resize(img_src,(int(img_src.shape[1]//2),int(img_src.shape[0]//2)))
        img_dst = cv2.resize(img_dst,(int(img_dst.shape[1]//2),int(img_dst.shape[0]//2)))

    img_src_original = img_src.copy()

    img_src_gray = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)

    img_dst_original = img_dst.copy()
    img_dst_gray = cv2.cvtColor(img_dst,cv2.COLOR_BGR2GRAY)

    print("Getting Faces")
    src_rects= get_faces(img_src)
    dst_rects= get_faces(img_dst)
    num_faces = len(src_rects)
    print(len(src_rects), len(dst_rects))
    if len(src_rects)<1 or len(dst_rects)<1:
        print("One or More Faces Not Found. Exiting...")
        return None
    
    #Scenario 1 : two ima
    if swap_logic == "swap_within_frame":
        src_idx = 0
        dst_idx = 1
    else:
        src_idx = 0
        dst_idx = 0
    print("Getting Landmarks")
    final_shapes_dst = get_facial_landmarks(img_dst_gray,[dst_rects[dst_idx]])
    landmark_img_dst = draw_facial_landmarks(img_dst,final_shapes_dst)
    cv2.imwrite('dst_landmarks.jpg',landmark_img_dst)
    final_shapes_src = get_facial_landmarks(img_src_gray,[src_rects[src_idx]])
    landmark_img_src = draw_facial_landmarks(img_src,final_shapes_src)
    cv2.imwrite('src_landmarks.jpg',landmark_img_src)

    src_hull,src_mask,center_src,dst_hull,dst_mask,center_dst = hull_masks(img_src,img_dst,final_shapes_src,final_shapes_dst)
    print("Swapping")
    if method=="TRI":
        img_swap1 = TRI(img_src,img_dst,final_shapes_src,final_shapes_dst)
        img_swap1 = cv2.seamlessClone(np.uint8(img_swap1), img_dst, dst_mask, center_dst, cv2.NORMAL_CLONE)
    elif method == "TPS":
        
        img_swap1 = TPS(img_src,img_dst,final_shapes_src,final_shapes_dst,img_src_original)
        img_swap1 = cv2.seamlessClone(np.uint8(img_swap1), img_src, src_mask, center_src, cv2.NORMAL_CLONE)
    if resize:
        img_swap1 = cv2.resize(img_swap1,(img_swap1.shape[1]*2,img_swap1.shape[0]*2),interpolation=cv2.INTER_LINEAR)
    return img_swap1

def SwapTwoFaces(img_src,img_dst,method,swap_logic, resize=False):
    if resize:
        img_src = cv2.resize(img_src,(int(img_src.shape[1]//2),int(img_src.shape[0]//2)))
        img_dst = cv2.resize(img_dst,(int(img_dst.shape[1]//2),int(img_dst.shape[0]//2)))

    img_src_original = img_src.copy()

    img_src_gray = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)

    img_dst_original = img_dst.copy()
    img_dst_gray = cv2.cvtColor(img_dst,cv2.COLOR_BGR2GRAY)

    print("Getting Faces")
    src_rects= get_faces(img_src)
    dst_rects= get_faces(img_dst)
    print(len(src_rects), len(dst_rects))
    if len(src_rects)<2:
        print("Two Faces Not Found. Exiting...")
        return None
    #Scenario 1 : two ima
    if swap_logic == "swap_within_frame" or swap_logic=="swap_two_faces_img":
        src_idx = 0
        dst_idx = 1
    else:
        src_idx = 0
        dst_idx = 0
    print("Getting Landmarks")

    final_shapes_dst = get_facial_landmarks(img_dst_gray,[dst_rects[dst_idx]])
    landmark_img_dst = draw_facial_landmarks(img_dst,final_shapes_dst)
    cv2.imwrite('dst_facial_landmarks.jpg',landmark_img_dst)
    final_shapes_src = get_facial_landmarks(img_src_gray,[src_rects[src_idx]])
    landmark_img_src = draw_facial_landmarks(img_src,final_shapes_src)
    cv2.imwrite('src_facial_landmarks.jpg',landmark_img_src)


    src_hull,src_mask,center_src,dst_hull,dst_mask,center_dst = hull_masks(img_src,img_dst,final_shapes_src,final_shapes_dst)
    print("Swapping")
    
    if method=="TRI":
        img_swap1 = TRI(img_src,img_dst,final_shapes_src,final_shapes_dst)
        img_swap1 = cv2.seamlessClone(np.uint8(img_swap1),img_dst, dst_mask, center_dst, cv2.NORMAL_CLONE)

        img_swap2 = TRI(img_dst,img_swap1,final_shapes_dst,final_shapes_src)
        img_swap2 = cv2.seamlessClone(np.uint8(img_swap2),img_swap1, src_mask, center_src, cv2.NORMAL_CLONE)

    elif method == "TPS":
        
        img_swap1 = TPS(img_src,img_dst,final_shapes_src,final_shapes_dst,img_src_original)
        img_swap1 = cv2.seamlessClone(np.uint8(img_swap1),img_src, src_mask,center_src, cv2.NORMAL_CLONE)

        img_swap2 = TPS(img_swap1,img_src,final_shapes_dst,final_shapes_src,img_swap1)
        img_swap2 = cv2.seamlessClone(np.uint8(img_swap2),img_swap1, dst_mask, center_dst, cv2.NORMAL_CLONE)

    if resize:
        img_swap2 = cv2.resize(img_swap2,(img_swap2.shape[1]*2,img_swap2.shape[0]*2),interpolation=cv2.INTER_LINEAR)

    return img_swap2

# 2 imgs, 1 img and 1 vid, 1 vid and None
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

    input1 = './data/mes.jpg'
    input2 = './data/ron.jpg'

    # input1 = './data/two_faces.mp4'
    # input2 = None

    # input1 = './data/nitesh_reduced.mp4'
    # input2 = './data/ron.jpg'

    # input2 = './data/TestSet/Rambo.jpg'
    # input1 = './data/TestSet/Test1.mp4'

    # input2 = './data/TestSet/Scarlett.jpg'
    # input1 = './data/TestSet/Test3.mp4'

    method = "TRI"
    output_name = 'testset20/frame'

    if os.path.isdir("./data/outputs/"+output_name.split("/")[0]):
        pass
    else:
        cmd = 'mkdir ./data/outputs/' + output_name.split("/")[0]
        os.system(cmd)

    swap_logic = parse_input_types(input1,input2)
    print(swap_logic)
    if(swap_logic=="swap_within_frame"):
        cap = cv2.VideoCapture(input1)
        i = 1
        while(True):
            _i = format(i,'03')
            ret,img_src = cap.read()
            if ret:
                img_dst = img_src.copy()
                print(i)
                out_img = SwapTwoFaces(img_src,img_dst,method,swap_logic)
                if out_img is not None:
                    cv2.imwrite(f"./data/outputs/{output_name}{_i}.jpg", out_img)
            else:
                print("Video completed")
                break
            i+=1
            
            
    elif(swap_logic=="swap_img_in_vid"):
        cap = cv2.VideoCapture(input1)
        img_dst = cv2.imread(input2)
        i = 1
        while(True):
            _i = format(i,'03')
            ret,img_src = cap.read()
            if ret:
                print(i) 
                out_img = SwapOneFace(img_src,img_dst,method,swap_logic)
                if out_img is not None:
                    cv2.imwrite(f"./data/outputs/{output_name}{_i}.jpg", out_img)
            else:
                print("Video completed")
                break
            i+=1
    elif(swap_logic=="swap_two_imgs"):
        img_src = cv2.imread(input1)
        img_dst = cv2.imread(input2)
        img_src = cv2.resize(img_src,(500,500))
        img_dst = cv2.resize(img_dst,(500,500))

        out_img = SwapOneFace(img_src,img_dst,method,swap_logic)
        if out_img is not None:
            cv2.imwrite('./data/outputs/img_swap.jpg',out_img)

    elif(swap_logic=="swap_two_faces_img"):
        img_src = cv2.imread(input1)
        img_dst = cv2.imread(input2)
        out_img = SwapTwoFaces(img_src,img_dst,method,swap_logic)
        if out_img is not None:
            cv2.imwrite('./data/outputs/frame_swap.jpg',out_img)

    else:
        print("Invalid input formats. Please ensure input2 is an image if input 1 is an img, and an image or None if input1 is a video")
        exit()    
