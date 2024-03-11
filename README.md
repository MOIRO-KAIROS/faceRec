# faceRec

## conda 설치
#참고: https://webnautes.tistory.com/1844 # 레퍼런스

-  Architecture 확인
    ```
    uname -a # x86_64

    sudo apt-get update && sudo apt-get upgrade
    ```
- nvidia driver 설치

    ```
    apt --installed list | grep nvidia-driver # 설치할 수 있는 드라이버 버전을 확인
    sudo apt-get install nvidia-driver-525 # sudo apt install는 옛날버전이므로 X
    sudo reboot
    ```
- 잘 설치했는지 확인
    ```
    nvidia-smi
    sudo apt-get update && sudo apt-get upgrade
    ```
- cuda 설치 (11.8 ver)
    ```
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    sudo sh cuda_11.8.0_520.61.05_linux.run
    ```
    > 참고 페이지(https://webnautes.tistory.com/1844)에서 터미널 설정 확인: Continue / accept / Driver 해제 / Install

- 환경변수 추가
    ```
    vim ~/.bashrc
    ```
    > export PATH="/usr/local/cuda-11.8/bin:$PATH"
    
    > export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"

    ```
    source ~/.bashrc
    ```
- CUDA 잘 설치했는지 확인
    ```
    nvcc --version
    ```

## 가상환경 설치 (pytorch 설치)
```
conda create --name adaface python=3.9 && conda activate adaface
conda activate adaface
pip install pyyaml
pip install typeguard
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install scikit-image matplotlib pandas scikit-learn
pip install pytorch pytorch-lightning==1.8.6
pip install tqdm bcolz-zipline prettytable menpo mxnet opencv-python
```


```
conda list | grep torch
```
> pytorch-lightning         1.8.6                    pypi_0    pypi

> torch                     2.2.1+cu118              pypi_0    pypi

> torchaudio                2.2.1+cu118              pypi_0    pypi

> torchmetrics              1.3.1                    pypi_0    pypi

> torchvision               0.17.1+cu118             pypi_0    pypi


## RUN *** 전에 할 것

```pretrained``` 폴더 생성 후, weight(.ckpt) 다운로드
- 다운로드 (링크 클릭)
    | Arch | Dataset    | Link                                                                                         |
    |------|------------|----------------------------------------------------------------------------------------------|
    | R50  | MS1MV2     | [gdrive](https://drive.google.com/file/d/1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI/view?usp=sharing) |

- 파일 구조
    ```
    pretrained
        |
        |_____ adaface_ir50_ms1mv2.ckpt
    ```

## Run Inference
```
python inference.py
```

> warning이 뜬다면? 그냥 진행해도 OK, 아니면 오류문구 보고 고치기 (모르겠으면 바로 질문!!)

## Run Demo File (WebCam)


1. ```python3 0_store_embedding.py``` # face_dataset/test에 있는 얼굴들에 대한 특징값 추출 후 저장
2. ```python3 1_run_recognition.py``` # webcam 활성화 후, demo file 실행

## Run Demo File (.mp4)

1. video/iAm.zip 압축풀기 > iAm.mp4
    - 파일 구조
    ```
    video
      |
      |_____ iAm.mp4
    ```
2. ```python3 0_store_embedding.py```
3. ```python3 2_test_recognition.py``` # mp4에 대한 face Recognition 수행
    ```
    video_capture = cv2.VideoCapture('video/iAM.mp4') # 경로 설정 후 실행하기
    ```