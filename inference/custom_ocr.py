from paddleocr import PaddleOCR, draw_ocr
ocr = PaddleOCR(lang="en")  # English only

det_db_box_thresh=0.4  # Lower the detection threshold to capture more text
det_db_unclip_ratio=1.6  # Reduce box expansion slightly to avoid overlapping text
rec_algorithm="SVTR_LCNet"  # Already good, but try CRNN for printed IDs
rec_image_shape="3, 48, 320"  # Increase height for small text detection
rec_char_dict_path="custom_dict.txt"  # Use a dictionary of common ID words
use_angle_cls=True
cls_thresh=0.9  # Make rotation correction more strict


def get_text_data(result, thres=0.8):
    filtered_text = "\n".join(
        word_info[1][0] for line in result for word_info in line if word_info[1][1] > thres
    )
    return filtered_text

ocr = PaddleOCR(
    lang="en",
    use_angle_cls=True,  # Fix rotated text
    drop_score=0.5,  # Lower to include more text
    det_db_box_thresh=det_db_box_thresh,  # Lower to detect more text boxes
    det_db_unclip_ratio=det_db_unclip_ratio,  # Reduce expansion to avoid overlapping detections
    rec_algorithm=rec_algorithm,  # Best for printed ID text
    rec_image_shape=rec_image_shape,  # Improve small text recognition
#     rec_char_dict_path="custom_dict.txt",  # Use a custom dictionary
    cls_thresh=0.9,  # Improve rotation correction
    use_gpu=True,  # Enable GPU for faster processing
)
