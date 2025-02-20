from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
import random
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils import get_car, read_license_plate, write_csv
import os

# Environment settings for threading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

upload_folder = 'static/uploads'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

# Model setup
coco_model = YOLO('../weights/yolov8n.pt')
license_plate_model = YOLO('../weights/best.pt')
tracker = DeepSort(max_age=20)
object_dict = {2: 'car', 3: 'motorbike', 5: 'bus', 7: 'truck'}

@app.route('/', methods=['GET', 'POST'])
@cross_origin(origins='*')
def home_page():
    if request.method == 'POST':
        try:
            # Configuration values
            conf_thresh_object = float(request.form.get('conf_thresh_object', 0.4))
            conf_thresh_license = float(request.form.get('conf_thresh_license', 0.4))
            resize_factor = float(request.form.get('resize_factor', 2.0))
            file = request.files.get('image_file') or request.files.get('video_file')
            if file:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                
                tracks = []
                results = {}
                
                # Determine if file is an image or video
                file_ext = file.filename.split('.')[-1].lower()
                if file_ext in ['jpg', 'jpeg', 'png']:
                    type = 0 # Image
                    processed_file_path = process_img(file_path, conf_thresh_object, conf_thresh_license, 
                                                      resize_factor, tracks, results)
                elif file_ext in ['mp4', 'avi', 'mkv']:
                    type = 1 # Video
                    processed_file_path = process_video(file_path, conf_thresh_object, conf_thresh_license, 
                                                        resize_factor, tracks, results)
                else:
                    return render_template('index.html', msg='Unsupported file format')
                
                write_csv(results, './result.csv')
                return render_template('index.html', user_file=processed_file_path, type=type,
                                        rand=str(random.randint(1000, 9999)), msg='File processed successfully.')
            else:
                return render_template('index.html', msg='Please select a file to upload.')    
        except Exception as ex:
            print(ex)
            return render_template('index.html', msg='Error processing file.')
    else:
        return render_template('index.html')

def process_img(file_path, conf_thresh_object, conf_thresh_license, resize_factor, tracks, results):
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{os.path.basename(file_path)}')
    frame = cv2.imread(file_path)
    original_frame = frame.copy()
    frame_nmr = 0
    results[frame_nmr] = {}
    
    # Object detection
    object_detections = coco_model(frame, conf=conf_thresh_object)[0]
    license_detections = license_plate_model(frame, conf=conf_thresh_license, iou=0.4)[0]
    
    detections_ = []
    for object in object_detections.boxes.data.tolist():
        score, class_id = object[4:6]
        x1, y1, x2, y2 = map(int, object[:4])
        if int(class_id) in object_dict.keys():
            detections_.append([[x1, y1, x2 - x1, y2 - y1], score, class_id])
    
    tracker.delete_all_tracks()
    tracks = tracker.update_tracks(detections_, frame=frame)
    
    for track in tracks:
        draw_track(frame, track, results, frame_nmr)            
        
    for license_plate in license_detections.boxes.data.tolist():
        process_license_plate(license_plate, original_frame, frame, frame_nmr, resize_factor, tracks, results)
        
    cv2.imwrite(output_path, frame)
    return output_path

def process_video(file_path, conf_thresh_object, conf_thresh_license, resize_factor, tracks, results):
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{os.path.basename(file_path)}')
    cap = cv2.VideoCapture(file_path)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, 30, 
                          (int(cap.get(3)), int(cap.get(4))))
    
    frame_nmr = -1
    while cap.isOpened():
        frame_nmr += 1
        ret, frame = cap.read()
        if not ret:
            break
        
        results[frame_nmr] = {}
        original_frame = frame.copy()
        
        # Object detection
        object_detections = coco_model(frame, conf=conf_thresh_object)[0]
        license_detections = license_plate_model(frame, conf=conf_thresh_license, iou=0.4)[0]
        
        detections_ = []
        for object in object_detections.boxes.data.tolist():
            score, class_id = object[4:6]
            x1, y1, x2, y2 = map(int, object[:4])
            if int(class_id) in object_dict.keys():
                detections_.append([[x1, y1, x2 - x1, y2 - y1], score, class_id])
        
        tracks = tracker.update_tracks(detections_, frame=frame)
        for track in tracks:
            if track.is_confirmed():
                draw_track(frame, track, results, frame_nmr)
        for license_plate in license_detections.boxes.data.tolist():
            process_license_plate(license_plate, original_frame, frame, frame_nmr, resize_factor, tracks, results)
        out.write(frame)
    
    cap.release()
    out.release()
    tracker.delete_all_tracks()
    
    return output_path

def draw_track(frame, track, results, frame_nmr):
    track_id = track.track_id
    ltrb = track.to_ltrb()
    class_id = track.get_det_class()
    score = track.get_det_conf() or 0
    x1, y1, x2, y2 = map(int, ltrb)
    label = f'{object_dict[class_id]}-{track_id}-{score:.2f}'
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.rectangle(frame, (x1 - 1, y1 - 28), (x1 + len(label) * 12, y1), (0, 255, 0), -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
    results[frame_nmr][track_id] = {
        "vihicle_name": object_dict[class_id],
        "vihicle_score": score,
        "license_plate": None,
        "license_score": None
    }

def process_license_plate(license_plate, original_frame, frame, frame_nmr, resize_factor, tracks, results):
    score, class_id = license_plate[4:6]
    x1, y1, x2, y2 = map(int, license_plate[:4])
    xc1, yc1, xc2, yc2, vehicle_id, vehicle_score = get_car(license_plate, tracks, frame_nmr)
    license_plate_crop = original_frame[int(y1):int(y2), int(x1):int(x2), :]
    resized_license_plate_crop = cv2.resize(license_plate_crop, None, 
                                            fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)
    license_plate_text, license_plate_score = read_license_plate(resized_license_plate_crop)

    if license_plate_score > 0 and license_plate_text:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f'{license_plate_text} {license_plate_score:.2f}'
        cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if vehicle_id in results[frame_nmr]:
            results[frame_nmr][vehicle_id]['license_plate'] = license_plate_text
            results[frame_nmr][vehicle_id]['license_score'] = license_plate_score
        else:
            results[frame_nmr]['X' + str(len(results[frame_nmr]))] = {
                "vihicle_name": None,
                "vihicle_score": None,
                "license_plate": license_plate_text,
                "license_score": license_plate_score
            }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)