import net
import torch
import os
from face_alignment import align
import numpy as np
import cv2
from PIL import Image
import argparse
import sys
import time

sys_path = "src/faceRec"

adaface_models = {
    'ir_50': os.path.join(sys_path, "pretrained/adaface_ir50_ms1mv2.ckpt"),
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

def to_input(pil_rgb_image, RGB=False):
    np_img = np.array(pil_rgb_image)
    if RGB:
        brg_img = ((np_img / 255.) - 0.5) / 0.5 # rgb 기준 ..?
    else:
        brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5 # bgr 기준 ..?
    tensor = torch.tensor(np.expand_dims(brg_img.transpose(2,0,1),axis=0)).float()
    return tensor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--thresh', nargs='+', type=str, default=.2, help='unknown confidence < .2')
    parser.add_argument('--max_obj', type=int, default=6, help='detect 가능한 최대 얼굴의 개수')
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pretrained_model('ir_50') # .to(device)
    print('face recongnition model loaded')
    torch.set_grad_enabled(False) # for 메모리

    # 데이터 로드
    database_dir = os.path.join(sys_path, 'embed/features.pt')
    if (os.path.exists(database_dir)):
        known_face_encodings = torch.load(database_dir).to(device)
        known_face_names = torch.load(os.path.join(sys_path, 'embed/ids.pt'))
        print("knwon face list: ", len(known_face_names))
    else:
        print("Error: Face database not found")
        sys.exit(1)

    video_capture = cv2.VideoCapture(0) # 4, 6
    while True:
        face_encodings = []
        # Grab a single frame of video
        ret, frame = video_capture.read()
        # height, width = frame.shape[:2]
        if not ret:
            print("Warning: no frame")
            break

        frame = cv2.flip(frame, 1)
        pil_im = Image.fromarray(frame).convert('RGB')

        ## 1. 얼굴 feature 추출
        aligned_rgb_img, bboxes = align.get_aligned_face_for_webcam('', pil_im, opt.max_obj)
        bboxes = [[int(xy) for (xy) in bbox] for bbox in bboxes]
        for img in aligned_rgb_img:
            bgr_tensor_input = to_input(img)
            face_encoding, _ = model(bgr_tensor_input)
            face_encodings.append(face_encoding)
        
        if not face_encodings:
            continue
        
        ## 2. 얼굴 유사도 측정 with tensor
        # start_time = time.time() # 연산에 대한 실행 시간(start) check
        face_encodings = torch.squeeze(torch.stack(face_encodings), dim=1).to(device) # torch.squeeze(torch.stack(face_encodings), dim=1) # torch.squeeze()
        with torch.no_grad():
            face_distances = torch.matmul(face_encodings, known_face_encodings.T)
        best_match_index = torch.argmax(face_distances, dim=1)
        thresh = opt.thresh
        face_names = ["unknown" if torch.any(face_distances[i][idx] < thresh) else known_face_names[idx] for i, idx in enumerate(best_match_index)]
        # face_names = [known_face_names[idx] for idx in best_match_index] # threshold 없는 경우 ('unkown' 처리 안한 경우)
        # end_time = time.time() # 연산에 대한 실행 시간(end) check
        # print("Execution Time:", (end_time - start_time), "sec") # 실행 시간 0.0003 ~
        

        ## 3. bbox 시각화
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