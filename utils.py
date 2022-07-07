from fileinput import filename
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import os
import math
import shutil
import glob

# 결과 저장을 위한 폴더 변수 
FOLDER_BASE = "/home/bim/Programming_onC/Crack_Classification_20220527/stream_env"

FOLDER_DNNMODELS = os.path.join(FOLDER_BASE,"models")
FOLDER_IMAGES = os.path.join(FOLDER_BASE,"images")
FOLDER_RESULTS = os.path.join(FOLDER_BASE,"results")
FOLDER_TEMP = os.path.join(FOLDER_BASE,"temp")

FOLDER_MARKED_ORIGIN = FOLDER_TEMP # 마크업된 전체 사진을 저장하는 폴더
FOLDER_FOUND_CRACK = FOLDER_TEMP # 크랙으로 처음에 탐지된 파일을 저장하는 폴더``

FOLDER_CONFIRMED_NOCRACK = os.path.join(FOLDER_RESULTS, "1_confirmed_nocrack") # 크랙 사용자 확인 결과 크랙이 아닌 것으로 확인된 파일의 모음. 이 폴더는 재학습을 위해 사용된다.
FOLDER_CONFIRMED_CRACK = os.path.join(FOLDER_RESULTS, "2_confirmed_crack") # 크랙 사용자 확인 결과 크랙이 맞는 것으로 확인된 파일의 모음. 이 폴더는 재학습을 위해 사용된다.
FOLDER_CONFIRMED_FULLIMAGE = os.path.join(FOLDER_RESULTS, "3_confirmed_full_images") # 크랙 사용자 확인 결과 크랙이 맞는 것으로 확인된 것을 최종 표시한 파일을 저장하는 폴더


def get_model_lists():
    ''' 상용가능한 ML 모델리스트를 검색하여, {모델명: 모델 절대 경로} 형태의 딕셔너리를 반환합니다.'''
    model_list = glob.glob(os.path.join(FOLDER_DNNMODELS,'model_*'))
    model_list.sort(reverse=True)
    model_dict = {}
    for model in model_list:
        model_dict[os.path.basename(model)]=model

    return model_dict

def get_images(filename):
    ''' 이미지 폴더내에 있는 파일 전체 경로를 반환합니다.'''
    return os.path.join(FOLDER_IMAGES,filename)

def get_fullpath_crackfile(filename):
    ''' 윈도우 사이즈에 맞는 크기의 크랙파일 경로를 반환합니다. '''
    return os.path.join(FOLDER_FOUND_CRACK, filename)

def get_fullpath_markedorigin(filename):
    ''' marked된 전체 이미지 파일의 경로를 반환합니다.  '''
    return os.path.join(FOLDER_MARKED_ORIGIN, filename)

def copy_to_confirmed_crack(filename):
    '''크랙으로 최종 확인된 파일을 작업'''
    shutil.copy(
        os.path.join(FOLDER_FOUND_CRACK,filename),
        os.path.join(FOLDER_CONFIRMED_CRACK,filename)
    )

def copy_to_confirmed_nocrack(filename):
    '''크랙으로 최종 확인된 파일을 작업'''
    shutil.copy(
        os.path.join(FOLDER_FOUND_CRACK,filename),
        os.path.join(FOLDER_CONFIRMED_NOCRACK,filename)
    )

def write_confirmed_crack(reshaped_img_filename, crack_coord_list):
    '''
    원본 img에 수정된 crack coordination을 마크업 하고 이를 원본이름_crack_confirmed.png로 저장함.

    reshaped_img_filename: reshaped된 원본 이미지 파일 명
    crack_coord_list: crack 좌표 정보를 가지고 있는 리스트

    return 저장된 파일 이름
    '''

    img = cv2.imread(os.path.join(FOLDER_MARKED_ORIGIN,reshaped_img_filename))
    for coord in crack_coord_list:
        cv2.rectangle(img,coord[0], coord[1],(0,0,255),5)
    
    title, ext = os.path.splitext(reshaped_img_filename)
    file = f"{title}_final.png"
    file = os.path.join(FOLDER_CONFIRMED_FULLIMAGE, file)
    cv2.imwrite(file,img)

    return file


def crack_searcher(ml_model, img, source_file_name, threshold,progress_bar, file_postfix="", img_width=7700):
    '''
    crack_searcher 함수는 주어진 이미지를 window 사이즈로 위에서 아래로,
    왼쪽에서 오른쪽으로 Clipping 하여, Clipping된 이미지의 crack을 전달된 
    ml_model로 탐지합니다. 탐지된 결과를 이미지로 저장합니다.

    ml_model: Clipped된 이미지를 탐지할 DNN 모델 
    img: 이미지를 담은 numpy array
    source_file_name: 이미지 파일의 이름(확장자 포함)
    threshold: 탐색 민감도(0~1.0)
    progress_bar:탐색 단계 표시를 위한 st.progressbar 객체
    file_postfix: 저장된 파일을 위한 구분 기호값
    img_width: 이미지의 실제 폭(mm), 학습환경이었던, 1px = 0.95mm 에 맞추기 위한 조정값

    return 탐지된 crack 파일의 리스트
    '''

    ## 기존 작업 파일 삭제
    files_to_delete = glob.glob(os.path.join(FOLDER_TEMP,f"*{file_postfix}*"))
    for file in files_to_delete:
        os.remove(file)
        print(f" - existing file deleted: {file}")


    ## 학습환경 조건 
    classes = ['Nocrack','Crack'] # 검색 대상은 2개
    WINDOW_SIZEpx = 160 # DNN input 크기 160 x 160
    MMperPx = 0.95 # 이미지 픽셀과 실제 mm의 비율 (1px = 0.95mm)
    source_file_name = os.path.splitext(source_file_name)[0] # 확장자명을 제외환 파일 이름만 저장


    ## 학습환경에 맞추기 위한 이미지 사이즈 변경
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    PIC_Hpx, PIC_Wpx,_ = img.shape

    image_resize_ratio = round(img_width / (MMperPx * PIC_Wpx),2) 
    img = cv2.resize(img,(0,0),fx=image_resize_ratio,fy=image_resize_ratio)
    PIC_Hpx, PIC_Wpx,_ = img.shape

    
    ## 탐색할 이미지 폭에 맞는 window의 개수(올림)
    NUM_SEARCH_BOX_W = math.ceil(PIC_Wpx / WINDOW_SIZEpx)
    NUM_SEARCH_BOX_H = math.ceil(PIC_Hpx / WINDOW_SIZEpx)

    ## window 개수에 맞춰서 실제 탐색할 이미지를 새로 만들어 줌 
    ## (가로끝, 세로끝을 조금 늘려 탐색 window가 끝가지 완전히 탐색할 수 있도록 함)
    RESHAPED_PIC_Wpx = WINDOW_SIZEpx * NUM_SEARCH_BOX_W
    RESHAPED_PIC_Hpx = WINDOW_SIZEpx * NUM_SEARCH_BOX_H

    #### - 원본 이미지에서 증가가 필요한 마진 길이 계산
    additional_void_Wpxs = RESHAPED_PIC_Wpx - PIC_Wpx
    additional_void_Hpxs = RESHAPED_PIC_Hpx - PIC_Hpx

    #### - 실제 탐색할 이미지 생성
    reshaped_img = np.append(img, np.full((PIC_Hpx, additional_void_Wpxs,3),255), axis=1) #그림의 가로 바닥 연장
    reshaped_img = np.append(reshaped_img, np.full((additional_void_Hpxs, RESHAPED_PIC_Wpx,3),255), axis=0) # 그림의 오른쪽 끝 연장
    reshaped_img = reshaped_img.astype(np.uint8)
    
    ## 마크업을 위한 이미지 복사본 준비 
    ## (원본에 마크업을 하면, 검색에 영향을 줄 수 있으므로, 검색 결과를 마크업할 별도의 이미지를 만들어줌)
    img_for_markup = reshaped_img.copy()

    ## 탐색 전 필요한 전역 변수 설정
    crack_cnt =0 # 탐색된 크랙의 총 개수

    font = cv2.FONT_HERSHEY_SIMPLEX # 표시할 폰트 정보
    font_scale =4
    font_thickness = 10
    font_color=(255,0,0)

    crack_files_list =[] # 탐색된 crack 파일명 리스트 (파일명만)
    crack_coord_list =[] # 크랙이 검출된 좌표를 저장하는 리스트 ((x1, y1), (x2,y2))

    ## 탐색 시장 (바깥 순환문, X 값, 안쪽 순환값 Y 값. 즉 위에서 아래로 훑으면서 왼쪽에서 오른쪽으로 탐색을 시작함.)
    ## 바깥쪽 루프 시작(왼족에서 오른쪽)
    for n, X in enumerate(range(0,RESHAPED_PIC_Wpx,WINDOW_SIZEpx)):
        # 진행상태 표시를 위한 Progress bar설정
        progress = (n+1)/NUM_SEARCH_BOX_W
        progress_bar.progress(progress)

        # 탐색 X 값 from~to설정
        from_X = X
        to_X = X + WINDOW_SIZEpx

        ## 안쪽 루프 시작(위에서 아래로)
        for Y in range(0,RESHAPED_PIC_Hpx, WINDOW_SIZEpx):

            # 탐색 Y값 from~to설정
            from_Y = Y
            to_Y = Y + WINDOW_SIZEpx 
            
            # 전체 이미지에서 윈도우 사이즈에 맞는 Clip 이미지 추출
            cliped_img = reshaped_img[from_Y:to_Y,from_X:to_X,:]

            # 추출된 이미지 판별
            prediction_result = ml_model.predict(cliped_img[np.newaxis, :])
            prediction_result = tf.nn.sigmoid(prediction_result[0])
            prediction_result_F = tf.where(prediction_result[0] < threshold, 0, 1) # 1에 가까운 값일 수록 crack일 가능성 업

            # Crack으로 판정된 경우
            if prediction_result_F == 1:
                crack_cnt +=1 # Crack개수 더하기
                crack_coord_list.append(((from_X, from_Y),(to_X, to_Y))) # 크랙 위치 추가

                # 마크업용 이미지에 박스 & 넘버링 문자 그리기
                cv2.rectangle(img_for_markup,(from_X, from_Y), (to_X, to_Y), (255,0,0), 5) # 박스 치기
                caption = f"{crack_cnt}" ## 넘버링 문자 중앙 정렬을 위한 좌표 확인
                caption_size = cv2.getTextSize(caption,font,font_scale, font_thickness)[0]
                caption_loc_X = int(from_X + WINDOW_SIZEpx / 2 - caption_size[0] / 2)
                caption_loc_Y = int(from_Y + WINDOW_SIZEpx / 2 + caption_size[1] / 2)
                cv2.putText(img_for_markup, caption ,(caption_loc_X+6, caption_loc_Y+6),font,font_scale,(255,255,255),font_thickness) ## 문자가 잘 보이기 위한 흰색 음영
                cv2.putText(img_for_markup, caption ,(caption_loc_X, caption_loc_Y),font,font_scale,font_color,font_thickness)

                # 크랙이 검출된 윈도우 이미지 저장
                file = f"{source_file_name}_{file_postfix}_crack{crack_cnt:03d}.png"
                crack_files_list.append(file)
                file = os.path.join(FOLDER_FOUND_CRACK,file)
                cv2.imwrite(file, cv2.cvtColor(cliped_img,cv2.COLOR_RGB2BGR))

    # Reshaped 원본 이미지 파일 저장                
    file=f"{source_file_name}_{file_postfix}.png"
    crack_files_list.append(file) # crackfile리스트의 제일 마지막 전값은 Reshaped 원본 이미지 파일명
    file=os.path.join(FOLDER_MARKED_ORIGIN,file)
    cv2.imwrite(file, cv2.cvtColor(reshaped_img,cv2.COLOR_RGB2BGR))

    # 탐색이 완료된 마크업 이미지 파일 저장                
    file=f"{source_file_name}_{file_postfix}_crack.png"
    crack_files_list.append(file) # crackfile리스트의 제일 마지막 값은 크랙이 표시된 이미지 파일명
    file=os.path.join(FOLDER_MARKED_ORIGIN,file)
    cv2.imwrite(file, cv2.cvtColor(img_for_markup,cv2.COLOR_RGB2BGR))

    return crack_files_list, crack_coord_list
###############################################################################