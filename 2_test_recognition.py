import net
import torch
import os
from face_alignment import align
import numpy as np
import cv2
from PIL import Image


adaface_models = {
    'ir_50':"pretrained/adaface_ir50_ms1mv2.ckpt",
}

def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    # tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    tensor = torch.tensor(np.array([brg_img.transpose(2,0,1)])).float()
    return tensor

def face_distance(face_encodings, face_to_compare, tolerance=0.6):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.

    :param face_encodings: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    distances = []
    for encoding in face_encodings:
        distance = torch.norm(encoding - face_to_compare)
        distances.append(distance.item())
    return distances
    # matches = [distance <= tolerance for distance in distances]
    # return matches

def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    matches = [distance <= tolerance for distance in face_distance(known_face_encodings, face_encoding_to_check)]
    return matches

if __name__ == '__main__':

    model = load_pretrained_model('ir_50')
    # 데이터 로드
    known_face_encodings = torch.load('embed/features.pt')
    known_face_names = torch.load('embed/ids.pt')

    face_names = []
    process_this_frame = True

    video_capture = cv2.VideoCapture('video/iAM.mp4')
    
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        height, width = frame.shape[:2]

        frame = cv2.flip(frame, 1)
        # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        if process_this_frame:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(frame).convert('RGB')

            aligned_rgb_img, bboxes = align.get_aligned_face_for_webcam('', pil_im)
            bboxes = [[int(xy) for (xy) in bbox] for bbox in bboxes]
            face_names = []
            for img in aligned_rgb_img:
                bgr_tensor_input = to_input(img)
                face_encoding, _ = model(bgr_tensor_input)
                # See if the face is a match for the known face(s)
                matches = compare_faces(known_face_encodings, face_encoding, 10)
                name = "Unknown"

                face_distances = face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)

        process_this_frame = not process_this_frame


        for (x1, y1, x2, y2, _), f_name in zip(bboxes, face_names):
            cv2.rectangle(frame,(x1, y1), (x2, y2),(0, 0, 255), 1)
            cv2.rectangle(frame, (x1, y2 - 30), (x2, y2), (0, 0, 255), cv2.FILLED)
            
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f_name, (x1 + 6, y2 - 6), font, .5, (0, 0, 0), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()