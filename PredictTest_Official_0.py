import argparse
import os
import platform
import sys
from pathlib import Path
from PIL import ImageDraw, ImageFont
import time
import pyrealsense2 as rs
import math
import numpy as np
import torch
#from timeit import default_timer as timer

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[1]  # YOLOv5 root directory'E:/Robot/Robot/.venv/yolov5' #
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

from filterpy.kalman import KalmanFilter
from scipy.signal import medfilt

# Khởi tạo bộ lọc Kalman với 2D position (x, y)
kf = KalmanFilter(dim_x=2, dim_z=2)
kf.x = np.array([0., 0.])  # Giá trị khởi tạo của center_point (tọa độ đầu)
kf.F = np.eye(2)           # Ma trận chuyển đổi trạng thái
kf.H = np.eye(2)           # Ma trận quan sát
kf.P *= 1000.              # Độ không chắc chắn ban đầu
kf.R = np.eye(2) * 0.1     # Ma trận nhiễu quan sát
kf.Q = np.eye(2) * 0.1     # Ma trận nhiễu hệ thống

# Hàm cập nhật vị trí từ Kalman Filter
def kalman_smooth_position(measured_position):
    kf.predict()
    kf.update(measured_position)
    return kf.x  # Trả về vị trí làm mượt


def filter_point_within_range(center_point, points, r):
    ##filter_points_within_range(): Filters points around a given center point within a radius r,
    # which is likely useful for focusing on the detected object (e.g., filtering out noisy points).
    if center_point is not None and points.size > 0:
        filtered_points = []
        for p in points:
            # Calculate the Euclidean distance between the center point and the current point # distance function
            distance = np.linalg.norm(np.array(center_point) - np.array(p))

            # Checking if the distance is within the specified range
            if distance <= r:
                filtered_points.append(p)

        # Calculate the center point of the filtered points
        if filtered_points:
            filtered_points_array = np.array(filtered_points)
            center_point_filtered = np.mean(filtered_points_array, axis=0).astype(int)
            center_point_filtered = np.median(filtered_points_array, axis=0).astype(int)
        else:
            center_point_filtered = center_point

        return filtered_points, center_point_filtered


def approximate_contour_with_pca(contour_points, epsilon):
    ##approximate_contour_with_pca(): Uses the Douglas-Peucker algorithm to approximate contour shapes,
    # then performs PCA on those points.
    # This function suggests file employs PCA on contours rather than raw depth points,
    # which could simplify pose estimation.
    # Approximate the contour
    approximated_contour = cv2.approxPolyDP(contour_points, epsilon, closed=True)
    approximated_contour = approximated_contour.squeeze()

    # Perform PCA on the approximate contour
    pca = PCA(n_components=2)
    pca.fit(approximated_contour)
    mean = pca.mean_
    eigenvectors = pca.components_
    eigenvalues = pca.explained_variance_

    return mean, eigenvectors, eigenvalues

def map_range(value, in_min, in_max, out_min, out_max):
    new_value = (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return new_value


def drawAxis(im0, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)

    ## [visualization1]
    angle = np.arctan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = np.sqrt((p[1] - q[1]) ** 2 + (p[0] - q[0]) ** 2)

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * np.cos(angle)
    q[1] = p[1] - scale * hypotenuse * np.sin(angle)
    cv2.line(im0, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 1, cv2.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * np.cos(angle + np.pi / 4)
    p[1] = q[1] + 9 * np.sin(angle + np.pi / 4)
    cv2.line(im0, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 1, cv2.LINE_AA)

    p[0] = q[0] + 9 * np.cos(angle - np.pi / 4)
    p[1] = q[1] + 9 * np.sin(angle - np.pi / 4)
    cv2.line(im0, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 1, cv2.LINE_AA)

def find_plane(points):
    ##This function uses Singular Value Decomposition (SVD) to fit a plane to a set of points.
    # This plane fitting might be used to establish the object's orientation relative to the plane it’s on,
    # supporting roll, pitch, and yaw angle calculations.
    c = np.mean(points, axis=0)
    r0 = points - c
    u, s, v = np.linalg.svd(r0)
    nv = v[-1, :]
    ds = np.dot(points, nv)
    param = np.r_[nv, -np.mean(ds)]
    return param


cfg = rs.config() #ok
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipe = rs.pipeline()
profile = pipe.start(cfg)

align = rs.align(rs.stream.color)

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
parameter = []

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

    return True, frames, color_image, depth_image, color_frame,  depth_frame, depth_intrinsics

#@smart_inference_mode()
def run(
    weights=ROOT / 'yolov5s-seg.pt',  # model.pt path(s)
    #source=ROOT / 'data/images' ,  # file/dir/URL/glob/screen/0(webcam)
    source=0,
    #data=ROOT / 'dataset/coco128-seg.yaml',  # dataset.yaml path
    data = 'E:\Robot\Robot\.venv\dataset-segment\Segment.v2i.yolov5pytorch\data.yaml',
    imgsz=(640, 640),  # inference size (height, width)
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
    counter=0,# color for displaying
    image_callback=None,
    raw_image_callback=None,
    parameter_callback=None,
    class_callback=None,
    flag=None
):
    global parameter
    global animalclass
    global pitch_history
    global pitch_counter
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
    device = select_device(device)
    if counter == 1:
        load_model_path = 'data_segmentv4/best.pt'
        model = DetectMultiBackend(weights=load_model_path, device=device, dnn=dnn, data=data, fp16=half)
    # model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    view_img = check_imshow(warn=True)
    # cap = cv2.VideoCapture(1)
    # frame_img = cap.read()
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    bs = len(dataset)
    vid_path, vid_writer = [None] * bs, [None] * bs
    # get_frame_stream()

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    for path, im, im0s, vid_cap, s in dataset:
        #start_time = timer()
        with dt[0]:
            frame_img = vid_cap
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            imnew = im.half() if model.fp16 else im.int() # uint8 to fp16/32
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
            seen += 1
            #if webcam:  # batch_size >= 1
            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            im_raw = im0.copy()
            s += f'{i}: '
            #else:
            #    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # im.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4]*1.0, im.shape[2:], upsample=True) #HWC
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                masks2 = process_mask(proto[i], det[:, 6:], det[:, :4],imnew.shape[2:], upsample=True)
                det[:, :4] = scale_boxes(imnew.shape[2:], det[:, :4], im0.shape).round() # rescale boxes to im0 size
                # Segments
                segments2 = [
                    scale_segments(im0.shape if retina_masks else imnew.shape[2:], x, im0.shape, normalize=False)
                    for x in reversed(masks2segments(masks))]  ## UN-NORMALIZED

                segments = [
                        scale_segments(im0.shape if retina_masks else im.shape[2:], x, im0.shape, normalize=True)
                        for x in reversed(masks2segments(masks))]  ## NORMALIZED
                # hsv = cv2.cvtColor(im0, cv2.COLOR_BGR2HSV)
                # lower_bound = np.array([10, 100, 100])  # Lower HSV bounds for orange
                # upper_bound = np.array([20, 255, 255]) # Adjust these values for your object
                #
                # # Create a mask for the defined color range
                # mask = cv2.inRange(hsv, lower_bound, upper_bound)
                # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #
                # # Create an empty black image (same size as the frame)
                # output = np.zeros_like(mask)
                #
                # # Fill the detected object(s) with white
                # for contour in contours:
                #     cv2.drawContours(output, [contour], -1, (255), thickness=cv2.FILLED)
                # object_on_black = cv2.bitwise_and(im0, im0, mask=mask)
                # cv2.imshow("im0", output)

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    animal_class = str(names[int(c)])

                # Mask plotting
                annotator.masks(
                    masks,
                    colors=[colors(x, True) for x in det[:, 5]],
                    im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(
                        0).contiguous() /
                            255 if retina_masks else im[i])

                ret, new_frame, bgr_image, depth_image, bgr_frame, depth_frame, depth_intrinsics = get_frame_stream()

                # Write results
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    seg = segments[j].reshape(-1)  # (n,2) to (n*2)
                    line = (cls, *seg, conf) if save_conf else (cls, *seg)  # label format
                    #   with open(f'{txt_path}.txt', 'a') as f:
                    #      f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    c = int(cls)  # integer class
                    label = None if hide_labels else(names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                    # this is for draw the known bounding box
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    box = xyxy
                    centerx, centery = int((int(box[0]) + int(box[2])) / 2), int((int(box[1]) + int(box[3])) / 2)
                    center_point = centerx, centery
                    im0 = annotator.result()

                    seg_point = np.array(segments2[j], dtype=np.int32)
                    if first_time == 10:
                        with open("seg_point.txt", "a") as f:
                            # Convert seg_point to string
                            seg_point_str = ''.join(map(str, seg_point))
                            # Concatenate string and array
                            my_list = [animal_class, seg_point_str]
                            # Convert the list to a string representation
                            output_string = '\n'.join(map(str, my_list))
                            print("class", output_string.split('\n')[0])
                            print(output_string, file=f)
                            cv2.imwrite('dethi/mask.jpg', im_raw)
                    first_time = first_time + 1

                    cv2.polylines(im0, [np.array(segments2[j], dtype=np.int32)], isClosed=True, color =(0,0,0), thickness=3)
                    yolo_raw_frame = im0.copy()
                    ##########################################################
                    #################### AREA ESTIMATION. ####################
                    ##########################################################

                    area = cv2.contourArea(seg_point)
                    ctrl_area = area
                    if ctrl_area > 0 and ctrl_area < 9000000: # OG@5250-9500(9000)

                        ##########################################################
                        #################### YAW ESTIMATION. ####################
                        ##########################################################

                        ## Performing PCA analysis
                        mean = np.empty((0))
                        mean2 = np.empty((0))
                        mean, eigenvectors, eigenvalues = cv2.PCACompute2(np.array(segments2[j], dtype=np.float64), mean)
                        eps_inc = 0.02
                        max_count = 5
                        mean2, eigenvectors2, eigenvalues2 = approximate_contour_with_pca(np.array(segments2[j], dtype= np.int32), epsilon=eps)

                        # Storing the center of the object
                        cntr = (int(mean[0, 0]), int(mean[0, 1]))

                        cntr2 = (int(mean2[0]), int(mean2[1]))

                        # Drawing the principal components
                        # cv2.circle(im0, cntr, 3, color=(255, 0, 255), thickness=2)
                        # cv2.circle(im0, cntr2, 3, color=(0, 0, 255), thickness=2)
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

                        checking_point_seal = (int(cntr[0] + 0.05 * eigenvectors2[0, 0] * eigenvalues2[0]),
                                               int(cntr[1] + 0.05 * eigenvectors2[1, 0] * eigenvalues2[1])) #X

                        checking_point_duck = (int(cntr[0] + 0.03 * eigenvectors[1, 0] * eigenvalues[1, 0]), #T
                                               int(cntr[1] + 0.03 * eigenvectors[1, 0] * eigenvalues[1, 0])) #Xác định hướng chính (thân vịt) và hướng phụ (đầu vịt).

                        checking_point_duck_head = (int(cntr[0] + 0.025 * eigenvectors[0, 0] * eigenvalues[0, 0]),
                                                    int(cntr[1] + 0.025 * eigenvectors[0, 1] * eigenvalues[0, 0]))

                        # cv2.circle(im0, checking_point_duck, 3, color=(255, 0, 255), thickness=2)
                        cv2.circle(im0, checking_point_duck_head, 3, color=(100, 0, 255), thickness=2)

                        data_checking_seal = cv2.pointPolygonTest(np.array(segments2[j], dtype=np.int64),
                                                                   (checking_point_seal[0], checking_point_seal[1]),
                                                                  measureDist=False)

                        data_checking_duck = cv2.pointPolygonTest(np.array(segments2[j], dtype=np.int64),
                                                                  (checking_point_duck[0], checking_point_duck[1]),
                                                                  measureDist=False)

                        data_checking_duck_head = cv2.pointPolygonTest(np.array(segments2[j], dtype=np.int64),
                                                                       (checking_point_duck_head[0], checking_point_duck_head[1]),
                                                                       measureDist=False)
                        seg = np.array(segments2[j], dtype=np.int64)


                        #while True:
                        if data_checking_seal < 0:
                            count = count + 1
                            if count >= max_count:
                                eps += eps_inc
                                count = 0  # reset the counter
                                mean2, eigenvectors2, eigenvalues2 = approximate_contour_with_pca(
                                    np.array(segments2[j], dtype=np.int32), epsilon=eps)
                                checking_point_seal = (int(cntr[0] + 0.055 * eigenvectors2[0, 0] * eigenvalues2[0]),
                                                        int(cntr[1] + 0.055 * eigenvectors2[1, 0] * eigenvalues2[1]))
                                data_checking_seal = cv2.pointPolygonTest(np.array(segments2[j], dtype=np.int64),
                                                                          (checking_point_seal[0],
                                                                           checking_point_seal[1]),
                                                                          measureDist=False)
                        else:
                            count = 0  # reset the counter if the point is inside the contour
                            eps = 0.25

                        if animal_class == "seal":
                            p3_point = (int(cntr[0] + 0.05 * eigenvectors[0, 0] * eigenvalues[0, 0]),
                                        int(cntr[1] + 0.05 * eigenvectors[0, 1] * eigenvalues[0, 0]))
                            p4_point = (int(cntr[0] - 0.04 * eigenvectors[0, 0] * eigenvalues[0, 0]),
                                        int(cntr[1] - 0.04 * eigenvectors[0, 1] * eigenvalues[0, 0]))
                        elif animal_class == "duck":
                            p3_point = (int(cntr[0] + 0.03 * eigenvectors[0, 0] * eigenvalues[0, 0]),
                                        int(cntr[1] + 0.03 * eigenvectors[0, 1] * eigenvalues[0, 0]))
                            p4_point = (int(cntr[0] - 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0]),
                                        int(cntr[1] - 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0]))
                        elif animal_class == "sealX":
                            p3_point = (int(cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0]),
                                        int(cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0]))
                            p4_point = (int(cntr[0] - 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0]),
                                        int(cntr[1] - 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0]))

                        # cv2.circle(im0, p3_point, radius=5, color=(0, 0, 255))
                        # cv2.circle(im0, p4_point, radius=5, color=(255, 0, 0))

                        if 0 <= p3_point[0] < depth_frame.width and 0 <= p3_point[1] < depth_frame.height:
                            distances_p3 = depth_frame.get_distance(p3_point[0], p3_point[1]) * 1000
                        if 0 <= p4_point[0] < depth_frame.width and 0 <= p4_point[1] < depth_frame.height:
                            distances_p4 = depth_frame.get_distance(p4_point[0], p4_point[1]) * 1000

                        if animal_class == "sealX":
                            drawAxis(im0, cntr, p1_new, color=(0, 255, 0), scale=1)
                            drawAxis(im0, cntr, p2_new, color=(255, 255, 0), scale=5)
                            drawAxis(im0, cntr, p1, color=(0, 127, 0), scale=1)
                            drawAxis(im0, cntr, p2, color=(127, 127, 0), scale=5)
                        else:
                            drawAxis(im0, cntr, p1, color=(0, 255, 0), scale=1)
                            drawAxis(im0, cntr, p2, color=(255, 255, 0), scale=5)

                        yaw = atan2(eigenvectors[0, 1], eigenvectors[0, 0]) # orientation in radius
                        # print("yaw ", yaw)
                        rotation_angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
                        yaw2 = atan2(eigenvectors2[0, 1], eigenvectors2[0, 0]) # orientation in radius
                        # print("yaw2 ", yaw2)
                        # Convert rotation angle from radians to degrees
                        rotation_angle_degrees = np.degrees(rotation_angle)
                        # print('rotation_angle', rotation_angle_degrees)

                        ##########################################################
                        ################ ROLL, PITCH ESTIMATION. #################
                        ##########################################################
                        off_coef = 2
                        offset_x = int((xyxy[2] - xyxy[0]) / 4) + off_coef
                        offset_y = int((xyxy[3] - xyxy[1]) / 4) - off_coef
                        interval_x = int((xyxy[2] - xyxy[0] - 2 * offset_x) / 2)
                        interval_y = int((xyxy[3] - xyxy[1] - 2 * offset_y) / 2)
                        points = np.zeros([9, 3])
                        for i in range(3): # step 0, 1, 2
                            for j in range(3):  # step 0, 1, 2
                                x = int(xyxy[0]) + offset_x + interval_x * i
                                y = int(xyxy[1]) + offset_y + interval_y * j
                                dist = depth_frame.get_distance(x, y) * 1000
                                Xtemp = dist * (x - intr.ppx) / intr.fx
                                Ytemp = dist * (y - intr.ppy) / intr.fy
                                Ztemp = dist
                                points[j + i * 3][0] = Xtemp
                                points[j + i * 3][1] = Ytemp
                                points[j + i * 3][2] = Ztemp

                        param = find_plane(points)

                            # Pitch = math.atan2(-param[2], np.sqrt(param[0]**2 + param[1]**2)) * 180 / math.pi #Pitch
                            #
                            # if(Pitch<0):
                            #     Pitch = Pitch + 90
                            # else:
                            #     Pitch = Pitch - 90

                            #roll = math.atan2(param[2], param[1]) * 180 / math.pi #roll
                        roll = 90

                        if(roll < 0):
                            roll = roll + 90
                        else:
                            roll = roll - 90

                        ##########################################################
                        ################## X, Y, Z ESTIMATION. ###################
                        ##########################################################

                        object_coordinates = []
                        cx = int((xyxy[0] + xyxy[2]) / 2)
                        cy = int((xyxy[1] + xyxy[3]) / 2)

                        depth_point = (0, 0, 0)
                        new_center_point = (0, 0)
                        new_center_point_head = (0, 0)

                        ################## CHANGE DIFFERENCE CLASS ###################
                        nem = 0
                        if animal_class == "seal":
                            dist = depth_frame.get_distance(cx + 0, cy + 0) * 1000
                            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics,
                                                                        [cx, cy], dist)
                            # yaw = yaw + np.pi / 2
                        elif animal_class == "duck":
                            dist = depth_frame.get_distance(cx + 0, cy + 0) * 1000
                            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics,
                                                                        [cx, cy], dist)
                            yaw = yaw + np.pi / 2

                        elif animal_class == "duckT":
                            if data_checking_duck > 0:
                                if data_checking_duck_head > 0:
                                    # print("duong")
                                    # # Calculate the head position offset based on the primary eigenvector (assuming forward direction)
                                    # head_offset_distance = 0.01 * eigenvalues[0, 0]  # Adjust this factor as needed for correct head placement
                                    # head_offset = (int(eigenvectors[0, 0] * head_offset_distance),
                                    #                 int(eigenvectors[0, 1] * head_offset_distance))
                                    # new_center_point = (int(cntr[0] + 0.03 * eigenvectors[0, 0] * eigenvalues[0, 0]),
                                    #                     int(cntr[1] + 0.03 * eigenvectors[0, 1] * eigenvalues[0, 0]))
                                    # new_center_point_head = (int(cntr[0] + 0.01 * eigenvectors[0, 0] * eigenvalues[0, 0]) + head_offset[0],
                                    #                         int(cntr[1] + 0.01 * eigenvectors[0, 1] * eigenvalues[0, 0]) + head_offset[1])
                                    # if seg.size > 0:
                                    #     _, new_center_point_head = filter_point_within_range(new_center_point_head, seg, r=30)
                                    #     _, new_center_point = filter_point_within_range(new_center_point, seg, r=30)
                                    # new_center_point_head = kalman_smooth_position(new_center_point_head)
                                    # new_center_point = kalman_smooth_position(new_center_point)
                            #     else:
                            #         print("am")
                                    # Calculate the head position offset based on the primary eigenvector (assuming forward direction)
                                    head_offset_distance = 0.01 * eigenvalues[0, 0]  # Adjust this factor as needed for correct head placement
                                    head_offset = (int(eigenvectors[0, 0] * head_offset_distance),
                                                   int(eigenvectors[0, 1] * head_offset_distance))
                                    new_center_point = (int(cntr[0] - 0.03 * eigenvectors[0, 0] * eigenvalues[0, 0]),
                                                        int(cntr[1] - 0.03 * eigenvectors[0, 1] * eigenvalues[0, 0]))
                                    new_center_point_head = (int(cntr[0] - 0.01 * eigenvectors[0, 0] * eigenvalues[0, 0]) - head_offset[0],
                                                            int(cntr[1] - 0.01 * eigenvectors[0, 1] * eigenvalues[0, 0]) - head_offset[0])
                                    if seg.size > 0:
                                        _, new_center_point_head = filter_point_within_range(new_center_point_head, seg, r=30)
                                        _, new_center_point = filter_point_within_range(new_center_point, seg, r=30)
                                    new_center_point_head = kalman_smooth_position(new_center_point_head) # kalman filter
                                    new_center_point = kalman_smooth_position(new_center_point)
                            new_center_point = tuple(map(int, new_center_point))
                            new_center_point_head = tuple(map(int, new_center_point_head))
                            # else:
                            #     print("am")
                            #     # Calculate the head position offset based on the primary eigenvector (assuming forward direction)
                            #     head_offset_distance = 0.01 * eigenvalues[0, 0]  # Adjust this factor as needed for correct head placement
                            #     head_offset = (int(eigenvectors[0, 0] * head_offset_distance),
                            #                    int(eigenvectors[0, 1] * head_offset_distance))
                            #     new_center_point = (int(cntr[0] - 0.03 * eigenvectors[0, 0] * eigenvalues[0, 0]),
                            #                         int(cntr[1] - 0.03 * eigenvectors[0, 1] * eigenvalues[0, 0]))
                            #     new_center_point_head = (int(cntr[0] - 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0]) + head_offset[0],
                            #                             int(cntr[1] - 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0]) + head_offset[0])
                            #     if seg.size > 0:
                            #         _, new_center_point_head = filter_point_within_range(new_center_point_head, seg, r=25)
                            #         _, new_center_point = filter_point_within_range(new_center_point, seg, r=25)
                            cv2.circle(im0, new_center_point, radius=5, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                            cv2.circle(im0, new_center_point_head, radius=5, color=(100, 50, 255), thickness=2, lineType=cv2.LINE_AA)
                            # if 0 <= new_center_point[0] < depth_frame.width and 0 <= new_center_point[1] < depth_frame.height:
                            #     dist = depth_frame.get_distance(new_center_point[0], new_center_point[1]) * 1000
                            # depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics,[new_center_point[0], new_center_point[1]], dist)
                            if 0 <= new_center_point_head[0] < depth_frame.width and 0 <= new_center_point_head[1] < depth_frame.height:
                                dist = (depth_frame.get_distance(new_center_point_head[0], new_center_point_head[1])) * 1000
                                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics,[new_center_point_head[0],
                                                                                            new_center_point_head[1]], dist)
                            yaw = yaw + np.pi / 2
                        elif animal_class == "sealX":
                            if depth_point[2] < 350:
                                if data_checking_sealX > 0:
                                    new_center_point = (int(cntr[0] + 0.0475 * eigenvectors2[0, 0] * eigenvalues2[0]),
                                                        int(cntr[1] + 0.0475 * eigenvectors2[0, 1] * eigenvalues2[0]))
                                    if seg.size > 0:
                                        _, new_center_point = filter_point_within_range(new_center_point, seg, r=22)
                                #cv2.circle(im0, new_center_point, radius=5, color=(255,255,0), thickness=1, lineType=cv2.LINE_AA)
                                if 0 <= new_center_point[0] < depth_frame.width and 0 <= new_center_point[1] < depth_frame.height:
                                    dist = depth_frame.get_distance(new_center_point[0], new_center_point[1]) * 1000
                                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [new_center_point[0], new_center_point[1]], dist)
                            yaw = atan2(eigenvectors2[0, 1], eigenvectors2[0, 0])

                            if yaw >= np.pi:
                                yaw = yaw - np.pi
                            elif yaw < np.pi:
                                yaw = yaw + np.pi

                        Xtarget = depth_point[0]
                        Ytarget = depth_point[1]
                        Ztarget = depth_point[2]

                        ##########################################################
                        ############## Refining roll and Pitch angles ############
                        ##########################################################

                        X_3 = distances_p3 * (p3_point[0] + 0 - intr.ppx) / intr.fx # the distance from RGB camera to realsense center
                        Y_3 = distances_p3 * (p3_point[1] + 0 - intr.ppx) / intr.fx
                        X_4 = distances_p4 * (p4_point[0] + 0 - intr.ppx) / intr.fx # the distance from RGB camera to realsense center
                        Y_4 = distances_p4 * (p4_point[1] + 0 - intr.ppx) / intr.fx
                        # b/a
                        length = sqrt(pow((X_3 - X_4), 2) + pow((Y_3 - Y_4), 2))
                        # print("length", length)

                        # Calculate pitch
                        if int(Ztarget) > 350:
                            pitch = 0
                        else:
                            if ((distances_p3 - distances_p4) < 0): # Z_3 - Z_4
                                pitch = +atan2(abs(distances_p3 - distances_p4), length)
                                pitch = pitch * 180/pi
                            else:
                                pitch = -atan2(abs(distances_p3 - distances_p4), length)
                                pitch = pitch * 180/pi

                        if animal_class == "duck":
                            roll = atan2(Y_4 - Y_3, X_4 - X_3) * 180 / pi # -160 - 135
                        #
                        if animal_class == "seal":
                            # roll = -atan2(Y_4 - Y_3, distances_p4 - distances_p3) * 180/pi
                            roll = atan2(X_4 - X_3, Y_4 - Y_3) * 180 / pi

                        #########################################################
                        ############## DISPLAYING PARAMETER SECTION. ############
                        #########################################################

                        ## Displaying X,Y,Z parameter ##
                        label_coordinate = "(" + str(round(Xtarget)) + ", " + str(round(Ytarget)) + ", " + str(
                            round(Ztarget)) + ")"
                        cv2.putText(im0, label_coordinate,
                                    (int((xyxy[0] + xyxy[2]) / 2) - 40, int((xyxy[1] + xyxy[3]) / 2) + 60),
                                    cv2.FONT_HERSHEY_PLAIN, 1, color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)
                        ## Displaying roll, Pitch, Yaw and Area ##
                        cvtR2D = np.rad2deg(yaw)
                        cvtR2D_2 = np.rad2deg(yaw2)
                        cvtR2D_new = map_range(cvtR2D, 0, 360, -180, 180)  # quy đổi giá trị
                        label_roll = "roll: " + str(round(roll, 2))
                        label_pitch = "Pitch: " + str(round(pitch, 2))
                        label_yaw = "Yaw: " + str(round(cvtR2D_new, 2))
                        label_yaw2 = "Yaw2: " + str(round(cvtR2D_2, 2))
                        label_area = "Area: " + str(int(area))

                        # cv2.putText(im0, label_roll,
                        #             (int((xyxy[0] + xyxy[2]) / 2) - 40, int((xyxy[1] + xyxy[3]) / 2)),
                        #             cv2.FONT_HERSHEY_PLAIN, 1, color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)
                        # cv2.putText(im0, label_pitch,
                        #             (int((xyxy[0] + xyxy[2]) / 2) - 40, int((xyxy[1] + xyxy[3]) / 2) + 20),
                        #             cv2.FONT_HERSHEY_PLAIN, 1, color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)
                        # cv2.putText(im0, label_yaw,
                        #             (int((xyxy[0] + xyxy[2]) / 2) - 40, int((xyxy[1] + xyxy[3]) / 2) + 40),
                        #             cv2.FONT_HERSHEY_PLAIN, 1, color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)
                        # cv2.putText(im0, label_yaw2,
                        #             (int((xyxy[0] + xyxy[2]) / 2) - 40, int((xyxy[1] + xyxy[3]) / 2) + 80),
                        #             cv2.FONT_HERSHEY_PLAIN, 1, color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)
                        # cv2.putText(im0, label_area,
                        #             (int((xyxy[0] + xyxy[2]) / 2) - 40, int((xyxy[1] + xyxy[3]) / 2) + 80),
                        #             cv2.FONT_HERSHEY_PLAIN, 1, color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)
                        # print("roll: " + str(round(roll, 2)) + "Pitch: " + str(round(pitch, 2)) + ", Yaw: " + str(
                        #     round((-int(np.rad2deg(yaw)) - 90), 2)))

                        parameter = (float(Xtarget), float(Ytarget), float(Ztarget), float(roll), float(pitch), float(cvtR2D_new), length, 1.0)


            # if platform.system() == 'Linux' and p not in windows:
            #     windows.append(p)
            #     cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            #     cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            # #return im0
            # # cv2.imshow("im0", im0)
            # #cv2.imshow("Frame_yolo", yolo)
            # if cv2.waitKey(1) == ord('q'):  # 1 millisecond
            #     cv2.destroyAllWindows()
            #     if np.all(im0) != None and im0.shape(480, 640, 3):
            #         processed_frame = im0.copy()
            #         return True, processed_frame


        if image_callback is not None:
            image_callback(im0)
        if raw_image_callback is not None:
            raw_image_callback(yolo_raw_frame)
        if parameter_callback is not None:
            parameter_callback(parameter)
        if class_callback is not None:
            class_callback(animal_class)

        if flag == 1: # 1 millisecond
            break

                        ## [visualization1]
                        # def getOrientation(pts, im0):
                        #     ## [pca]
                        #     # Construct a buffer used by the pca analysis
                        #     sz = len(pts)
                        #     data_pts = np.empty((sz, 2), dtype=np.float64)
                        #     for i in range(data_pts.shape[0]):
                        #         data_pts[i, 0] = pts[i, 0, 0]
                        #         data_pts[i, 1] = pts[i, 0, 1]
                        #
                        #     ###########################################################
                        #     ##################### YAW ESTIMATION. #####################
                        #     ###########################################################
                        #
                        #     # Perform PCA analysis
                        #     mean = np.empty((0))
                        #     mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
                        #
                        #     # Store the center of the object
                        #     cntr = (int(mean[0, 0]), int(mean[0, 1]))
                        #     ## [pca]
                        #
                        #     ## [visualization]
                        #     # Draw the principal components
                        #     cv2.circle(im0, cntr, 3, (255, 0, 255), 2)
                        #     p1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
                        #           cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
                        #     p2 = (cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
                        #           cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
                        #     drawAxis(im0, cntr, p1, (0, 255, 0), 20)
                        #     drawAxis(im0, cntr, p2, (255, 255, 0), 20)
                        #
                        #     angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / math.pi  # orientation in radians
                        #
                        #     text3 = "Yaw: " + str(round(angle))
                        #     ## [visualization]
                        #     cv2.putText(im0, text3,(int((xyxy[0] + xyxy[2]) / 2) - 40, int((xyxy[1] + xyxy[3]) / 2) + 40),
                        #                 cv2.FONT_HERSHEY_PLAIN, 1, (200, 255, 0) , thickness=2, lineType=cv2.LINE_AA)
                        #     print("Yaw: ", round(angle,2))
                        #
                        #     return angle

                        # for k,mask in enumerate(masks):
                        #     mask = mask.cpu().numpy().astype(np.uint8)  # Chuyển về numpy array và định dạng uint8
                        #     mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)[1]  # Áp dụng threshold để chuyển về binary mask
                        #
                        #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        #
                        # for cnt in contours:
                        #
                        #     # Tính PCA và vẽ trục
                        #     angle_radians = getOrientation(cnt, im0)
                        #     angle_degrees = angle_radians * 180.0 / np.pi
                        #
                        #     # Vẽ contours và trung tâm
                        #     cv2.drawContours(im0, [cnt], -1, (0, 255, 0), 2)
                        #
                        #     # Calculate area and check if it's zero
                        #     area = cv2.contourArea(cnt)
                        #     if area == 0:
                        #         # Skip this contour because it's too small or might be an error
                        #         continue
                        #     M = cv2.moments(cnt) #calculate center of mass of the object, area of the object
                        #     cx = int(M['m10'] / M['m00'])
                        #     cy = int(M['m01'] / M['m00'])
                        #     cv2.circle(im0, (cx, cy), 2, (0, 0, 255), -1)
                        #     # Vẽ và viết góc xoay lên hình ảnh
                        #     text = f"Angle: {angle_degrees:.2f} degrees"
                        #     cv2.putText(im0, text, (cx , cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                        #                 2)
                        #     cv2.putText(im0, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # # print(real_world_coordinates)

                        ################## Display roll and Pitch Angles #################


                        ########### Display XYZ parameters ############
                        # coordinates_text = "(" + str(
                            # Decimal(str(Xtarget)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) + \
                            #                ", " + str(
                            # Decimal(str(Ytarget)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) + \
                            #                ", " + str(
                            # Decimal(str(Ztarget)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) + ")"
                        # fixed_color = (0, 0, 255)
                        #label = None if hide_labels else (
                        #   names[c] if hide_conf else f'{names[c]} {conf:.2f}{coordinates_text}'
                        #                              f'{text1}'
                        #                              f'{text2}')
                        #annotator.box_label(xyxy, label, color=fixed_color)
                        # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)
                    #if save_crop:
                    #   save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            # Stream results
            # if view_img:
            #     if platform.system() == 'Linux' and p not in windows:
            #         windows.append(p)
            #         cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            #         cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            # return im0
            # cv2.imshow("Frame_yolo", yolo)
            # if cv2.waitKey(1) == ord('q'):  # 1 millisecond
            #     cv2.destroyAllWindows()
            #     if np.all(im0) != None and im0.shape(480, 640, 3):
            #         processed_frame = im0.copy()
            #         return True, processed_frame

    #
    #     # Print time (inference-only)
    #     LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
    #
    # # Print results
    # t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")


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
