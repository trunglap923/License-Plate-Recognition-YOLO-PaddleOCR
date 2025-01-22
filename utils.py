from paddleocr import PaddleOCR
import numpy as np
import re

ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, lang='en', enable_mkldnn=False)

def get_car(license_plate, tracks, frame_nmr):
    x1, y1, x2, y2 = map(int, license_plate[:4])
    isFound = False
    for track in tracks:
        if track.is_confirmed() or not frame_nmr:
            track_id = track.track_id
            score = track.get_det_conf()
            ltrb = track.to_ltrb()
            xc1, yc1, xc2, yc2 = map(int, ltrb)
            if xc1 < x1 + 10 and yc1 < y1 + 10 and xc2 > x2 - 10 and yc2 > y2 - 10:
                isFound = True
                vehicle_id = track_id
                vehicle_score = score
                break
            
    if isFound:
        return xc1, yc1, xc2, yc2, vehicle_id, vehicle_score
    return -1, -1, -1, -1, -1, -1

def read_license_plate(license_plate):
    result = ocr.ocr(license_plate)
    license_plate_text = ''
    license_plate_score = 1
    if result[0] != None:
        texts_use = []
        for r in result[0]:
            score = r[1][1]
            
            if np.isnan(score):
                score = 0
            if score > 0.6:
                license_plate_score *= score
                pattern = re.compile('[\W]')
                text = pattern.sub('', r[1][0])
                text = text.replace("???", "")
                text = text.replace("O", "0")
                text = text.replace("粤", "")
                texts_use.append(text)
                
        texts_sorted = sorted(texts_use, key=len)
        license_plate_text = ''.join(texts_sorted).strip()
        
        return license_plate_text, license_plate_score
    return '', 0

def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{}\n'.format('frame_nmr', 'vehicle_id', 'vehicle_name',
                                       'vehicle_score', 'license_plate', 'license_score'))
    
        for frame_nmr in results:
            for vehicle_id in results[frame_nmr]:
                f.write('{},{},{},{},{},{}\n'.format(frame_nmr, vehicle_id,
                                                    results[frame_nmr][vehicle_id]['vihicle_name'],
                                                    results[frame_nmr][vehicle_id]['vihicle_score'],
                                                    results[frame_nmr][vehicle_id]['license_plate'],
                                                    results[frame_nmr][vehicle_id]['license_score']))
        f.close()
            