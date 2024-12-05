import argparse
import os
import platform
import sys
from pathlib import Path
from PIL import ImageDraw, ImageFont
import torch
import numpy as np
import time
import pyrealsense2 as rs
import math

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[1]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# Set ROOT as the YOLOv5 root directory
ROOT = Path("E:/Robot/Robot/.venv/yolov5").resolve()  # Absolute path to YOLOv5 directory

# Add ROOT to sys.path if it's not already there
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.segment.general import masks2segments, process_mask, process_mask_native
from yolov5.utils.torch_utils import select_device, smart_inference_mode
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from math import atan2, cos, sin, sqrt, pi, asin, acos, degrees
from collections import Counter
from sklearn.decomposition import PCA

cfg = rs.config() #ok
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) #ok
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) #ok
pipe = rs.pipeline()
profile = pipe.start(cfg)
#pipe.start(cfg)
# align_to =
align = rs.align(rs.stream.color)

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics() #ok

parameter = []

def filter_points_within_range(center_point, points, r):
    if center_point is not None and points.size > 0:
        filtered_points = []
        #print(points)
        # Iterate over all points
        for p in points:
            # Calculate the Euclidean distance between the center point and the current point
            distance = np.linalg.norm(np.array(center_point) - np.array(p))

            # Check if the distance is within the specified range
            if distance <= r:
                filtered_points.append(p)

        # Calculate the center point of the filtered points
        if filtered_points:
            filtered_points_array = np.array(filtered_points)
            center_point_filtered = np.mean(filtered_points_array, axis=0).astype(int)
        else:
            center_point_filtered = center_point

        return filtered_points, center_point_filtered

def approximate_contour_with_pca(contour_points, epsilon):
    """
    Approximates the contour using the Douglas-Peucker algorithm and performs PCA on the approximated contour.

    Parameters:
        contour_points (numpy.ndarray): Array of contour points.
        epsilon (float): Approximation accuracy parameter.

    Returns:
        tuple: Tuple containing mean, eigenvectors, and eigenvalues of the approximated contour.
    """
    # Approximate the contour
    approximated_contour = cv2.approxPolyDP(contour_points, epsilon, closed=True)
    approximated_contour = approximated_contour.squeeze()

    # Perform PCA on the approximated contour
    pca = PCA(n_components=2)
    pca.fit(approximated_contour)
    mean = pca.mean_
    eigenvectors = pca.components_
    eigenvalues = pca.explained_variance_

    return mean, eigenvectors, eigenvalues

def find_plane(points):
    c = np.mean(points, axis=0)
    r0 = points - c
    u, s, v = np.linalg.svd(r0)
    nv = v[-1, :]
    ds = np.dot(points, nv)
    param = np.r_[nv, -np.mean(ds)]
    return param

def map_range(value, in_min, in_max, out_min, out_max):
    new_value = (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return new_value

def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

def get_frame_stream():
    print()
    # Wait for a coherent pair of frames: depth and color
    frames = pipe.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    depth_frame.get_height()

    if not depth_frame or not color_frame:
        # If there is no frame, probably camera not connected, return False
        print(
            "Error, impossible to get the frame, make sure that the Intel Realsense camera is correctly connected")
        return None, None

        # Apply filter to fill the Holes in the depth image
    color_intrin = color_frame.profile.as_video_stream_profile().get_intrinsics()

    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.holes_fill, 3)
    filtered_depth = spatial.process(depth_frame)

    hole_filling = rs.hole_filling_filter()
    filled_depth = hole_filling.process(filtered_depth)

    # Create colormap to show the depth of the Objects
    colorizer = rs.colorizer()
    depth_colormap = np.asanyarray(colorizer.colorize(filled_depth).get_data())

    # Convert images to numpy arrays
    # distance = depth_frame.get_distance(int(50),int(50))
    # print("distance", distance)
    depth_image = np.asanyarray(filled_depth.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    # cv2.imshow("Colormap", depth_colormap)
    # cv2.imshow("depth img", depth_image)

    return True, frames, color_image, depth_image, color_frame, depth_frame, depth_intrinsics


def run(
    weights=ROOT / 'yolo5s-seg.pt',  # model.pt path(s)
    source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / 'data/coco128-seg.yaml',  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width) #OG@(640, 640)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1,  # maximum detections per image
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / 'runs/predict-seg',  # save results to project/name
    name='exp',  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=1,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    retina_masks=False,
    WHITE=(225, 255, 255),  # color for displaying #BGR
    RED=(0, 0, 255),  # color for displaying #BGR
    GREEN=(0, 255, 0),  # color for displaying #BGR
    BLUE=(255, 0, 0),  # color for displaying #BGR
    GREEN_2=(0, 127, 0),  # color for displaying #BGR
    BLUE_2=(127, 0, 0),  # color for displaying #BGR
    TEAL=(255, 255, 0),  # color for displaying #BGR
    TEAL_2=(127, 127, 0),  # color for displaying #BGR
    PINK=(255, 0, 255),  # color for displaying #BGR
    PINK_2=(127, 0, 127),  # color for displaying #BGR
    YELLOW=(255, 255, 0),  # color for displaying #BGR
    ORANGE=(0, 128, 255),  # color for displaying #BGR
    counter = 0,#color for displaying
    image_callback=None,
    raw_image_callback=None,
    parameter_callback=None,
    class_callback = None,
    flag = None
):
    global parameter
    global pipeclass
    global pitch_counter
    global pitch_history
    new_pitch = 0
    pitch_history = []
    pitch_counter = 0
    first_time = 0
    counter += 1
    count = 0
    eps = 0.25
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    #print("before2")
    device = select_device(device)
    #print('counter',counter)
    if counter == 1:
        # print("before")
        #model = torch.hub.load('./yolov5', 'custom', source='local', path='best.pt', force_reload=False, skip_validation= True)
        # print("after")
        local_model_path = 'data_segmentv4/best.pt'
        model = DetectMultiBackend(weights=local_model_path, device=device, dnn=dnn, data=data, fp16=half)
    #model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    #model = torch.hub.load('./yolov5', 'custom', source='local', path='best.pt', force_reload=True)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    view_img = check_imshow(warn=True)
    #cap = cv2.VideoCapture(1)
    #frame_img = cap.read()
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    bs = len(dataset)
    vid_path, vid_writer = [None] * bs, [None] * bs
    #get_frame_stream()

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    for path, im, im0s, vid_cap, s in dataset:
        #print('p3')
        with dt[0]:
            frame_img = vid_cap
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            imnew = im.half() if model.fp16 else im.int()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred, proto = model(im, augment=augment, visualize=visualize)[:2]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image

            # # Detect number of threads
            # if num_threads == 2:
            #     print("Thread: Multi")
            # elif num_threads == 1:
            #     print("Thread: Single")

            seen += 1
            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            im_raw = im0.copy()
            s += f'{i}: '

            p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # im.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4]*1.0, im.shape[2:], upsample=True)  # HWC
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                masks2 = process_mask(proto[i], det[:, 6:], det[:, :4], imnew.shape[2:], upsample=True)  # HWC
                #print(masks2)
                det[:, :4] = scale_boxes(imnew.shape[2:], det[:, :4],
                                         im0.shape).round()  # rescale boxes to im0 size

                segments2 = [
                    scale_segments(im0.shape if retina_masks else imnew.shape[2:], x, im0.shape, normalize=False)
                    for x in reversed(masks2segments(masks))]  ## UN-NORMALIZED

                # NumberOfElements = sizeof(arr) / sizeof(arr[0]);
                # segments2_conver = np.array(segments2, dtype=np.int32)
                # print(segments2_conver)

                segments = [
                    scale_segments(im0.shape if retina_masks else im.shape[2:], x, im0.shape, normalize=True)
                    for x in reversed(masks2segments(masks))]  ## NORMALIZED

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    pipe_class = str(names[int(c)])
                    # g = f"{names[int(c)]}"
                    # cs = str(names[int(c)])
                    # print("Class_3: ", cs)
                    # print(n)

                # Mask plotting
                annotator.masks(
                    masks,
                    colors=[colors(x, True) for x in det[:, 5]],
                    im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(
                        0).contiguous() /
                           255 if retina_masks else im[i])

                ret, new_frame, bgr_image, depth_image, bgr_frame, depth_frame, depth_intrinsics = get_frame_stream()
                ###
                """
                This is the line befor for we do the filter so if is not reaching requirement we will filter it out
                only extract the grasping point of the object, then after is have been grasp and move to a certain 
                position the Robot will rise a flag for yolov5 know to show the whole class detection.
                we can say:
                if (pipe_class == "I_Pipe" or "Grasp_Pipe") and signal == 1:
                    then do the for loop down there otherwis dont do it
                    ...................................................
                else:
                    type_of_pipe = pipe_class
                    return this to the Robot for the detection
                """
                ###
                # Write results
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    seg = segments[j].reshape(-1)  # (n,2) to (n*2)
                    # print(segments[j])
                    line = (cls, *seg, conf) if save_conf else (cls, *seg)  # label format
                    # with open(f'{txt_path}.txt', 'a') as f:
                    # f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                    # this is for draw the known bounding box
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    box = xyxy
                    centerx, centery = int((int(box[0]) + int(box[2])) / 2), int((int(box[1]) + int(box[3])) / 2)
                    center_point = centerx, centery
                    # distance_mm = depth_image[centery, centerx]
                    # print(center_point)
                    # ImageDraw.ImageDraw.polygon(, segments2[j], outline=colors(c, True), width=3)
                    # cv2.circle(vid_cap,(segments2[j]), radius=1, color=(0,0,255), thickness=1)
                    im0 = annotator.result()

                    seg_point = np.array(segments2[j], dtype=np.int32)
                    if first_time == 10:
                        with open("seg_point.txt", "a") as f:
                            # Convert seg_point to a string
                            seg_point_str = ''.join(map(str, seg_point))
                            # Concatenate string and array
                            my_list = [pipe_class, seg_point_str]
                            # Convert the list to a string representation
                            output_string = '\n'.join(map(str, my_list))
                            print("class",output_string.split('\n')[0])
                            print(output_string, file=f)
                            cv2.imwrite('dethi/mask.jpg', im_raw)
                    first_time = first_time + 1
                    # print(first_time)

                    # cv2.putText(im0, "{} mm".format(distance_mm), (centerx, centery - 10), 0, 0.5, (0, 0, 255), 2)
                    # center_point_XYZ = rs.rs2_deproject_pixel_to_point(color_intrin, center_point, distance_mm)
                    # print(np.array(center_point_XYZ).astype(int))
                    # print("Center point: ", center_point)
                    cv2.polylines(im0, [np.array(segments2[j], dtype=np.int32)], isClosed=True, color=(0, 0, 0),
                                  thickness=3)
                    yolo_raw_frame = im0.copy()
                    ##########################################################
                    #################### AREA ESTIMATION. ####################
                    ##########################################################

                    area = cv2.contourArea(seg_point)
                    ctrl_area = area
                    if ctrl_area > 0 and ctrl_area < 900000:  # OG@5250-9500(9000)
                        "So this is the list of segmentation point segments2[j] "
                        "So we will move on to do the PCA for detecting vector X,Y for and object"
                        ""
                        ###########################################################
                        ##################### YAW ESTIMATION. #####################
                        ###########################################################

                        ## Performing PCA analysis
                        mean = np.empty((0))
                        mean2 = np.empty((0))
                        mean, eigenvectors, eigenvalues = cv2.PCACompute2(np.array(segments2[j], dtype=np.float64),mean)
                        eps_inc = 0.02
                        max_count = 5
                        mean2, eigenvectors2, eigenvalues2 = approximate_contour_with_pca(np.array(segments2[j], dtype=np.int32), epsilon=eps)

                        # print('mean2',mean2)
                        # print('eigenvectors2', eigenvectors2)
                        # print('eigenvalues2', eigenvalues2)
                        # print('mean',mean)
                        # print('eigenvectors', eigenvectors)
                        # print('eigenvalues', eigenvalues)
                        # Ensure the first eigenvector points in the desired direction
                        # desired_direction = np.array([1, 0])  # Replace with your desired direction
                        # if np.dot(eigenvectors[0, :], desired_direction) < 0:
                        #     eigenvectors[0, :] = -eigenvectors[0, :]
                        # if np.dot(eigenvectors[1, :], desired_direction) < 0:
                        #     eigenvectors[1, :] = -eigenvectors[1, :]
                        ## Storing the center of the object
                        cntr = (int(mean[0, 0]), int(mean[0, 1]))

                        cntr2 = (int(mean2[0]), int(mean2[1]))

                        ## Drawing the principal components
                        cv2.circle(im0, cntr, 3, PINK, 2)
                        cv2.circle(im0, cntr2, 3, RED, 2)
                        p1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
                              cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
                        p2 = (cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
                              cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])

                        p1_new = (cntr2[0] + 0.02 * eigenvectors2[0, 0] * eigenvalues2[0],
                                    cntr2[1] + 0.02 * eigenvectors2[0, 1] * eigenvalues2[0])
                        p2_new = (cntr2[0] - 0.02 * eigenvectors2[1, 0] * eigenvalues2[1],
                                cntr2[1] - 0.02 * eigenvectors2[1, 1] * eigenvalues2[1])

                        p3_point = (0, 0)
                        p4_point = (0, 0)

                        checking_point_T_pipe = (int(cntr[0] + 0.0825 * eigenvectors[1, 0] * eigenvalues[1, 0]),
                                                 int(cntr[1] + 0.0825 * eigenvectors[1, 1] * eigenvalues[1, 0]))  #Hệ số 0.0825 điều chỉnh khoảng cách từ cntr đến checking_point_T_pipe để đảm bảo nó nằm trên nhánh ngang.

                        checking_point_L_pipe = (int(cntr[0] + 0.06 * eigenvectors[0, 0] * eigenvalues[0, 0]
                                                             - 0.04 * eigenvectors[1, 0] * eigenvalues[1, 0]),
                                                 int(cntr[1] + 0.06 * eigenvectors[0, 1] * eigenvalues[0, 0]
                                                             - 0.04 * eigenvectors[1, 1] * eigenvalues[1, 0]))

                        checking_point_X_pipe = (int(cntr[0] + 0.06 * eigenvectors2[0, 0] * eigenvalues2[0]),
                                                 int(cntr[1] + 0.06 * eigenvectors2[1, 0] * eigenvalues2[1]))

                        cv2.circle(im0, checking_point_X_pipe, 3, PINK, 2)

                        data_checking_T_pipe = cv2.pointPolygonTest(np.array(segments2[j], dtype=np.int64),
                                                                (checking_point_T_pipe[0], checking_point_T_pipe[1])
                                                                    ,measureDist=False)

                        data_checking_L_pipe = cv2.pointPolygonTest(np.array(segments2[j], dtype=np.int64),
                                                                (checking_point_L_pipe[0], checking_point_L_pipe[1])
                                                                    ,measureDist=False)

                        data_checking_X_pipe = cv2.pointPolygonTest(np.array(segments2[j], dtype=np.int64),
                                                                (checking_point_X_pipe[0], checking_point_X_pipe[1])
                                                                    ,measureDist=False)
                        seg = np.array(segments2[j], dtype=np.int64)
                        # print("data_L",data_checking_L_pipe)
                        # print("data_X",data_checking_X_pipe)
                        #print(data_checking_T_pipe)

                        #while True:
                        if data_checking_X_pipe < 0:
                            count = count + 1
                            # print("cnt",count)
                            if count >= max_count:
                                eps += eps_inc
                                # print("eps",eps)
                                count = 0  # Reset the counter
                                mean2, eigenvectors2, eigenvalues2 = approximate_contour_with_pca(
                                    np.array(segments2[j], dtype=np.int32), epsilon=eps)
                                checking_point_X_pipe = (int(cntr[0] + 0.055 * eigenvectors2[0, 0] * eigenvalues2[0]),
                                                         int(cntr[1] + 0.055 * eigenvectors2[0, 1] * eigenvalues2[0]))
                                data_checking_X_pipe = cv2.pointPolygonTest(np.array(segments2[j], dtype=np.int64),
                                                                            (checking_point_X_pipe[0],
                                                                             checking_point_X_pipe[1])
                                                                            , measureDist=False)
                        else:
                            count = 0  # Reset the counter if the point is inside the contour
                            eps = 0.25
                            # print("catch a point")
                            # print(checking_point_X_pipe)


                        if pipe_class == "I_Pipe":
                            p3_point = (int(cntr[0] + 0.03 * eigenvectors[0, 0] * eigenvalues[0, 0]),
                                        int(cntr[1] + 0.03 * eigenvectors[0, 1] * eigenvalues[0, 0]))
                            p4_point = (int(cntr[0] - 0.03 * eigenvectors[0, 0] * eigenvalues[0, 0]),
                                        int(cntr[1] - 0.03 * eigenvectors[0, 1] * eigenvalues[0, 0]))
                        elif pipe_class == "T_pipe":
                            p3_point = (int(cntr[0] + 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0]),
                                        int(cntr[1] + 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0]))
                            p4_point = (int(cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0]),
                                        int(cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0]))
                        elif pipe_class == "L_pipe":
                            p3_point = (int(cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0]),
                                        int(cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0]))
                            if data_checking_L_pipe > 0:
                                p4_point = (int(cntr[0] + 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0]),
                                            int(cntr[1] + 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0]))
                            else:
                                p4_point = (int(cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0]),
                                            int(cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0]))
                        elif pipe_class == "X_pipe":
                            p3_point = (int(cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0]),
                                        int(cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0]))
                            p4_point = (int(cntr[0] - 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0]),
                                        int(cntr[1] - 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0]))

                        cv2.circle(im0, p3_point, radius=5, color=RED)
                        cv2.circle(im0, p4_point, radius=5, color=BLUE)

                        if 0 <= p3_point[0] < depth_frame.width and 0 <= p3_point[1] < depth_frame.height:
                            distances_p3 = depth_frame.get_distance(p3_point[0],p3_point[1])*1000
                        if 0 <= p4_point[0] < depth_frame.width and 0 <= p4_point[1] < depth_frame.height:
                            distances_p4 = depth_frame.get_distance(p4_point[0],p4_point[1])*1000

                        if pipe_class == "X_pipe":
                            drawAxis(im0, cntr, p1_new, GREEN, 1)
                            drawAxis(im0, cntr, p2_new, TEAL, 5)
                            drawAxis(im0, cntr, p1, GREEN_2, 1)
                            drawAxis(im0, cntr, p2, TEAL_2, 5)
                        else:
                            drawAxis(im0, cntr, p1, GREEN, 1)
                            drawAxis(im0, cntr, p2, TEAL, 5)
                        #roll = atan2(eigenvectors[2, 1], eigenvectors[2, 0])

                        yaw = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
                        print("yaw",yaw)
                        rotation_angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
                        yaw2 = atan2(eigenvectors2[0, 1], eigenvectors2[0, 0])  # orientation in radians
                        print("yaw2", yaw2)
                        # Convert rotation angle from radians to degrees
                        rotation_angle_degrees = np.degrees(rotation_angle)

                        ##########################################################
                        ################ ROLL, PITCH ESTIMATION. #################
                        ##########################################################
                        off_coef = 2
                        offset_x = int((xyxy[2] - xyxy[0]) / 4) + off_coef  # OG@10
                        offset_y = int((xyxy[3] - xyxy[1]) / 4) - off_coef  # OG@10
                        interval_x = int((xyxy[2] - xyxy[0] - 2 * offset_x) / 2)  # OG@2
                        interval_y = int((xyxy[3] - xyxy[1] - 2 * offset_y) / 2)  # OG@2
                        points = np.zeros([9, 3])  # OG@[9,3]
                        for i in range(3):  # OG@3 # step 0, 1, 2
                            for j in range(3):  # OG@3 # step 0, 1, 2
                                x = int(xyxy[0]) + offset_x + interval_x * i
                                y = int(xyxy[1]) + offset_y + interval_y * j
                                dist = depth_frame.get_distance(x, y) * 1000  # OG@*1000
                                Xtemp = dist * (x - intr.ppx) / intr.fx
                                Ytemp = dist * (y - intr.ppy) / intr.fy
                                Ztemp = dist
                                points[j + i * 3][0] = Xtemp
                                points[j + i * 3][1] = Ytemp
                                points[j + i * 3][2] = Ztemp

                        param = find_plane(points)

                        #roll = math.atan(param[2] / param[1]) * 180 / math.pi
                        roll = 90
                        # b/a
                        if (roll < 0):
                            roll = roll + 90
                        else:
                            roll = roll - 90

                        ##########################################################
                        ################## X, Y, Z ESTIMATION. ###################
                        ##########################################################

                        object_coordinates = []
                        cx = int((xyxy[0] + xyxy[2]) / 2)
                        cy = int((xyxy[1] + xyxy[3]) / 2)

                        depth_point = (0,0,0)
                        new_center_point = (0, 0)
                        #cv2.circle(im0, (cx,cy), radius=5, color=RED, thickness=1, lineType=cv2.LINE_AA)
                        #print(pipeclass)

                        ############### CHANGE FOR DIFFERENCE CLASS ###############
                        nem = 0
                        if pipe_class == "I_Pipe":
                            dist = depth_frame.get_distance(cx + 0, cy + 0) * 1000
                            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics,
                                                                        [cx, cy], dist)

                        elif pipe_class == "T_pipe":
                            if data_checking_T_pipe > 0:
                                new_center_point = (int(cntr[0] + 0.068 * eigenvectors[1, 0] * eigenvalues[1, 0]),
                                                    int(cntr[1] + 0.068 * eigenvectors[1, 1] * eigenvalues[1, 0]))
                                if seg.size > 0:
                                    _, new_center_point = filter_points_within_range(new_center_point,seg,r=25)
                            else:
                                new_center_point = (int(cntr[0] - 0.068 * eigenvectors[1, 0] * eigenvalues[1, 0]),
                                                    int(cntr[1] - 0.068 * eigenvectors[1, 1] * eigenvalues[1, 0]))
                                if seg.size > 0:
                                    _, new_center_point = filter_points_within_range(new_center_point,seg,r=25)
                            cv2.circle(im0, new_center_point, radius=5, color=RED, thickness=1, lineType=cv2.LINE_AA)
                            if 0 <= new_center_point[0] < depth_frame.width and 0 <= new_center_point[
                                1] < depth_frame.height:
                                dist = depth_frame.get_distance(new_center_point[0], new_center_point[1])*1000
                            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics,
                                                                          [new_center_point[0], new_center_point[1]], dist)
                            yaw = yaw + np.pi/2
                        elif pipe_class == "L_pipe":
                            if depth_point[2] < 350:
                                if data_checking_L_pipe > 0:
                                    new_center_point = (int(cntr[0] + 0.0375 * eigenvectors[0, 0] * eigenvalues[0, 0]
                                                                 - 0.0225 * eigenvectors[1, 0] * eigenvalues[1, 0]),
                                                     int(cntr[1] + 0.0375 * eigenvectors[0, 1] * eigenvalues[0, 0]
                                                                 - 0.0225 * eigenvectors[1, 1] * eigenvalues[1, 0]))
                                    if seg.size > 0:
                                        _, new_center_point = filter_points_within_range(new_center_point, seg, r=25)
                                else:
                                    new_center_point = (int(cntr[0] + 0.0375 * eigenvectors[0, 0] * eigenvalues[0, 0]
                                                                 + 0.0225 * eigenvectors[1, 0] * eigenvalues[1, 0]),
                                                     int(cntr[1] + 0.0375 * eigenvectors[0, 1] * eigenvalues[0, 0]
                                                                 + 0.0225 * eigenvectors[1, 1] * eigenvalues[1, 0]))
                                    if seg.size > 0:
                                        _, new_center_point = filter_points_within_range(new_center_point, seg, r=25)

                                cv2.circle(im0, new_center_point, radius=5, color=TEAL, thickness=1, lineType=cv2.LINE_AA)
                                if 0 <= new_center_point[0] < depth_frame.width and 0 <= new_center_point[1] < depth_frame.height:
                                    dist = depth_frame.get_distance(new_center_point[0], new_center_point[1]) * 1000
                                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics,
                                                                              [new_center_point[0], new_center_point[1]], dist)
                                #print("yaw_deg",np.degrees(yaw))
                                quarterstring = str()
                                if np.pi/4 > yaw > 0:
                                    if data_checking_L_pipe > 0:
                                        yaw = yaw - np.pi / 4
                                    else:
                                        yaw = yaw + np.pi / 4  # done
                                    quarterstring = "1st quarter" #done
                                elif np.pi/2 > yaw > np.pi/4:
                                    if data_checking_L_pipe > 0:
                                        yaw = yaw + np.pi / 4
                                    else:
                                        yaw = yaw - np.pi / 4 #done
                                    quarterstring = "2nd quarter"
                                elif 3*np.pi/4 > yaw > np.pi/2:
                                    if data_checking_L_pipe > 0:
                                        yaw = yaw + np.pi / 4
                                    else:
                                        yaw = yaw - np.pi / 4 #done
                                    quarterstring = "3rd quarter"
                                elif np.pi > yaw > 3*np.pi/4:
                                    yaw = yaw + np.pi / 4 #done
                                    quarterstring = "4th quarter"
                                elif np.pi/2 > yaw > np.pi/4:
                                    yaw = yaw + np.pi / 4 #done
                                    quarterstring = "5th quarter"
                                elif 0 > yaw > -np.pi/4:
                                    if data_checking_L_pipe > 0:
                                        yaw = yaw - np.pi/4
                                    else:
                                        yaw = yaw + np.pi / 4
                                    quarterstring = "6th quarter"
                                elif -np.pi/4 > yaw > -np.pi/2:
                                    yaw = yaw - np.pi/4
                                    quarterstring = "7th quarter"
                                elif -np.pi/2 > yaw > -3*np.pi/4:
                                    yaw = yaw - np.pi/4
                                    quarterstring = "8th quarter"
                                elif -3*np.pi/4 > yaw > -np.pi:
                                    yaw = yaw - np.pi/4
                                    quarterstring = "9th quarter"
                        elif pipe_class == "X_pipe":
                            if depth_point[2] < 350:
                                if data_checking_X_pipe > 0:
                                    #print("catch a point")
                                    #new_center_point = checking_point_X_pipe
                                    new_center_point = (int(cntr[0] + 0.0475 * eigenvectors2[0, 0] * eigenvalues2[0]),
                                                        int(cntr[1] + 0.0475 * eigenvectors2[0, 1] * eigenvalues2[0]))
                                    if seg.size > 0:
                                        _, new_center_point = filter_points_within_range(new_center_point, seg, r=22)
                                    ### old ###
                                    # new_center_point = (int(cntr[0] + 0.05 * eigenvectors2[0, 0] * eigenvalues2[0]),
                                    #                     int(cntr[1] + 0.05 * eigenvectors2[1, 0] * eigenvalues2[1]))
                                cv2.circle(im0, new_center_point, radius=5, color=TEAL, thickness=1,lineType=cv2.LINE_AA)
                                if 0 <= new_center_point[0] < depth_frame.width and 0 <= new_center_point[1] < depth_frame.height:
                                    dist = depth_frame.get_distance(new_center_point[0], new_center_point[1]) * 1000
                                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics,[new_center_point[0],
                                                                                new_center_point[1]], dist)
                            yaw = atan2(eigenvectors2[0, 1], eigenvectors2[0, 0])

                            if yaw >= np.pi :
                                yaw = yaw -np.pi
                            elif yaw <= np.pi :
                                yaw = yaw + np.pi

                        Xtarget = depth_point[0]
                        Ytarget = depth_point[1]
                        Ztarget = depth_point[2]

                        ##########################################################
                        ############## Refining Roll and Pitch angles ############
                        ##########################################################
                        #new_pitch = 0
                        X_3 = distances_p3 * (p3_point[0] + 0 - intr.ppx) / intr.fx   # the distance from RGB camera to realsense center
                        Y_3 = distances_p3 * (p3_point[1] + 0 - intr.ppy) / intr.fy
                        X_4 = distances_p4 * (p4_point[0] + 0 - intr.ppx) / intr.fx   # the distance from RGB camera to realsense center
                        Y_4 = distances_p4 * (p4_point[1] + 0 - intr.ppy) / intr.fy
                        # b/a
                        length = sqrt(pow((X_3 - X_4), 2) + pow((Y_3 - Y_4), 2))
                        # print("length",length)
                        # print("distance",abs(distances_p3 - distances_p4))

                        #Calcualte pitch######
                        if int(Ztarget) > 340 :
                            pitch = 0
                        else:
                            if ((distances_p3 - distances_p4) < 0):
                                #pitch = +atan2(abs(distances_p3 - distances_p4), length)
                                pitch = -atan2(abs(distances_p3 - distances_p4), length)
                                pitch = pitch * 180.0 / pi
                            else:
                                # pitch = -atan2(abs(distances_p3 - distances_p4), length)
                                pitch = +atan2(abs(distances_p3 - distances_p4), length)
                                pitch = pitch * 180.0 / pi

                        # print("pitch",pitch)
                        #########################################################
                        ############## DISPLAYING PARAMETER SECTION. ############
                        #########################################################

                        ## Displaying Class parameter ##
                        # print("Class: ", pipeclass)

                        ## Displaying X,Y,Z parameters ##
                        label_coordinate = "(" + str(round(Xtarget)) + ", " + str(round(Ytarget)) + ", " + str(
                            round(Ztarget)) + ")"
                        cv2.putText(im0, label_coordinate,
                                    (int((xyxy[0] + xyxy[2]) / 2) - 40, int((xyxy[1] + xyxy[3]) / 2) + 60),
                                    cv2.FONT_HERSHEY_PLAIN, 1, WHITE, thickness=1, lineType=cv2.LINE_AA)
                        # print("(" + str(round(Xtarget)) + ", " + str(round(Ytarget)) + ", " + str(
                        #     round(Ztarget)) + ")")

                        ## Displaying Roll, Pitch, Yaw angles and Area ##
                        cvtR2D = np.rad2deg(yaw)
                        cvtR2D_2 = np.rad2deg(yaw2)
                        #print("cvtR2D", cvtR2D)
                        cvtR2D_new = map_range(cvtR2D, 0, 360, -180, 180) # quy đổi giá trị
                        # print("cvtR2D_new", cvtR2D_new)
                        label_roll = "Roll: " + str(round(roll, 2))
                        label_pitch = "Pitch: " + str(round(pitch, 2))

                        #label_yaw = "Yaw: " + str(round((-int(cvtR2D) - 90), 2))
                        label_yaw = "Yaw: " + str(round(cvtR2D, 2))
                        label_yaw2 = "Yaw2: " + str(round(cvtR2D_2, 2))
                        label_area = "Area: " + str(int(area))
                        # cv2.putText(im0, label_roll,
                        #             (int((xyxy[0] + xyxy[2]) / 2) - 40, int((xyxy[1] + xyxy[3]) / 2)),
                        #             cv2.FONT_HERSHEY_PLAIN, 1, WHITE, thickness=1, lineType=cv2.LINE_AA)
                        # cv2.putText(im0, label_pitch,
                        #             (int((xyxy[0] + xyxy[2]) / 2) - 40, int((xyxy[1] + xyxy[3]) / 2) + 20),
                        #             cv2.FONT_HERSHEY_PLAIN, 1, WHITE, thickness=1, lineType=cv2.LINE_AA)
                        # cv2.putText(im0, label_yaw,
                        #             (int((xyxy[0] + xyxy[2]) / 2) - 40, int((xyxy[1] + xyxy[3]) / 2) + 40),
                        #             cv2.FONT_HERSHEY_PLAIN, 1, WHITE, thickness=1, lineType=cv2.LINE_AA)
                        # cv2.putText(im0, label_yaw2,
                        #             (int((xyxy[0] + xyxy[2]) / 2) - 40, int((xyxy[1] + xyxy[3]) / 2) + 80),
                        #             cv2.FONT_HERSHEY_PLAIN, 1, WHITE, thickness=1, lineType=cv2.LINE_AA)
                        # cv2.putText(im0, label_area,
                        #             (int((xyxy[0] + xyxy[2]) / 2) - 40, int((xyxy[1] + xyxy[3]) / 2) + 80),
                        #             cv2.FONT_HERSHEY_PLAIN, 1, WHITE, thickness=1, lineType=cv2.LINE_AA)
                        # print("Roll: " + str(round(roll, 2)) + "Pitch: " + str(round(pitch, 2)) + ", Yaw: " + str(
                        #     round((-int(np.rad2deg(yaw)) - 90), 2)))
                        # print("End One detection phase")
                        # print()
                        # if data_checking_X_pipe > 0:
                        #     # print("catch a point")
                        #     new_center_point = (int(cntr[0] + 0.05 * eigenvectors2[0, 0] * eigenvalues2[0]),
                        #                         int(cntr[1] + 0.05 * eigenvectors2[1, 0] * eigenvalues2[1]))
                        #     #cv2.imwrite(f'dethi/saved_image{counter}.jpg', im0)
                        # counter += 1
                        # print(counter)
                        if pipe_class == "L_pipe":
                            pipeclass = pipe_class + "_" + quarterstring
                        else:
                            pipeclass = pipe_class
                        parameter = (float(Xtarget), float(Ytarget), float(Ztarget), float(roll), float(pitch), float(cvtR2D_new), 1.0)
                        #pitch_history.append(pitch)
        # t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        # LOGGER.info(f'seg:%.1fms seg, Speed: %.1fms pre-process, %.1fms inference' % t)
        # %.1fms NMS per image at shape {(1, 3, *imgsz)}

        if image_callback is not None:
            image_callback(im0)
        if raw_image_callback is not None:
            raw_image_callback(yolo_raw_frame)
        if parameter_callback is not None:
            parameter_callback(parameter)
        if class_callback is not None:
            class_callback(pipeclass)

        if flag == 1:  # 1 millisecond
            break

            #Stream results
            # if view_img:
            #     if platform.system() == 'Linux' and p not in windows:
            #         windows.append(p)
            #         #cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            #         cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                #return im0
                #cv2.imshow('Frame_yolo', im0)
                # if cv2.waitKey(0) == ord('q'):  # 1 millisecond
                #     cv2.destroyAllWindows()
                #     if np.all(im0) != None and im0.shape == (480,640,3):
                #         processed_frame = im0.copy()
                #         return True, processed_frame

        # Print time (inference-only)
        #LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    # t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'seg:%.1fms seg, Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-seg.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
