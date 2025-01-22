from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils import get_car, read_license_plate

coco_model = YOLO('../weights/yolov8n.pt')
license_plate_model = YOLO('../weights/best.pt')
tracker = DeepSort(0)

object_dict = {2: 'car', 3: 'motorbike', 5: 'bus', 7: 'truck'}
tracks = []
conf_thresh_object = 0.4
conf_thresh_license = 0.5

image_path = './input_test/image/6.jpg'
video_path = './input_test/video/1.mp4'

mode = 'video'  # camera, video, image
if mode == 'camera':
    cap = cv2.VideoCapture(0)
elif mode == 'video':
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Can't read file!")
elif mode == 'image':
    frame = cv2.imread(image_path)
    frame_height, frame_width = frame.shape[:2]    

# Get the display size on the screen
screen_width = 1920
screen_height = 1080

if mode in ['camera', 'video']:
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if frame_width > screen_width or frame_height > screen_height:
    scale = min(screen_width / frame_width, screen_height / frame_height)
    display_width = int(frame_width * scale)
    display_height = int(frame_height * scale)
else:
    display_width = frame_width
    display_height = frame_height


while True:
    if mode in ['camera', 'video']:
        ret, frame = cap.read()
        if not ret:
            break
    elif mode == 'image':
        ret = False
    
    original_frame = frame.copy()
    
    object_detections = coco_model(frame, conf=conf_thresh_object)[0]
    license_detections = license_plate_model(frame, conf=conf_thresh_license, iou=0.4)[0]
    
    detections_ = []
    track_id = 0
    for object in object_detections.boxes.data.tolist():
        track_id += 1
        score, class_id = object[4:6]
        x1, y1, x2, y2 = map(int, object[:4])
        if int(class_id) in object_dict.keys():
            detections_.append([[x1, y1, x2-x1, y2-y1], score, class_id])
    
    tracks = tracker.update_tracks(detections_, frame=frame)
    
    for track in tracks:
        if mode == 'image' or track.is_confirmed():
            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            
            label = f'{object_dict[class_id]}-{track_id}'
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(frame, (x1 - 1, y1 - 28), (x1 + len(label) * 12, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    for license_plate in license_detections.boxes.data.tolist():
        score, class_id = license_plate[4:6]
        x1, y1, x2, y2 = map(int, license_plate[:4])
        xc1, yc1, xc2, yc2, vehicle_id = get_car(license_plate, tracks)
        license_plate_crop = original_frame[int(y1):int(y2), int(x1):int(x2), :]
        resized_license_plate_crop = cv2.resize(license_plate_crop, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
        license_plate_text, license_plate_score = read_license_plate(resized_license_plate_crop)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        if license_plate_score != 0:
            label = f'{license_plate_text} {license_plate_score:.2f}'
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    resized_frame = cv2.resize(frame, (display_width, display_height))
    
    cv2.imshow('YOLO Detection', resized_frame)
    
    if mode in ['camera', 'video'] and cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break
    elif mode == 'image':
        cv2.waitKey(0)
        break
        
cv2.destroyAllWindows()
