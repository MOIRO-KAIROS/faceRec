import net
import torch
import os
from face_alignment import align
import numpy as np


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

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    # tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    tensor = torch.tensor(np.array([brg_img.transpose(2,0,1)])).float()
    return tensor

def store_embeddings_to_db(folder):
    # 모든 얼굴의 임베딩 계산
    # features = dict()
    features = []
    ids = []
    for fname in sorted(os.listdir(test_image_path)):
        path = os.path.join(test_image_path, fname)
        aligned_rgb_img = align.get_aligned_face(path)
        bgr_tensor_input = to_input(aligned_rgb_img)
        feature, _ = model(bgr_tensor_input)

        # features[fname.split('.')[0]] = feature
        features.append(feature)
        ids.append(fname.split('.')[0])
    # print(features)
    
    if not os.path.exists('embed'):
        os.makedirs('embed')

    # Embeddings와 passage_ids를 저장
    features = torch.squeeze(torch.stack(features), dim=1)
    # print(features.shape)
    torch.save(features, os.path.join(sys_path, 'embed/features.pt'))
    torch.save(ids, os.path.join(sys_path, 'embed/ids.pt'))

    return features, ids

if __name__ == '__main__':

    model = load_pretrained_model('ir_50')
    feature, norm = model(torch.randn(2,3,112,112))

    test_image_path = os.path.join(sys_path, 'face_dataset/test')
    f, i = store_embeddings_to_db(test_image_path)
    # print(f)
    # print(i)
