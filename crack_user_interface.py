import streamlit as st
import tensorflow as tf
from PIL import Image
import glob
import cv2
import numpy as np
import os
from datetime import datetime
import math
import shutil
from utils import *


crack_files_list =[] # 탐색된 crack 파일명 리스트 (파일명만)
crack_coord_list =[] # 탐색된 crack의 위치를 저장하는 리스트

# 파일 저장시 중복된 파일명으로 인한 replace 방지를 위한 구분자
# 같은 세션에서 동일 파일로 작업하는 경우, 중복을 방지하기 위해서 해당 정보는 Session에 저장합니다.
if 'file_postfix' not in st.session_state:
    st.session_state['file_postfix'] = datetime.now().strftime("%Y%m%d_%H%M%s") 

# 선택된 모델의 이름과 위치를 저장하는 딕셔너리 객체
st.session_state['modelname'] = "" 

# 검색과 업데이트 버튼이 여러번 눌러지는 것을 방지하기 위한 disable option 입니다. 
if "disable_opt1" not in st.session_state:
    st.session_state.disable_opt1 = False

if "disable_opt2" not in st.session_state:
    st.session_state.disable_opt2 = False

# 모델을 로딩하기 위한 함수입니다. 굳이 함수로 구현한 이유는 화면 Reloading시 매번 ML 모델이 로딩되는 것을 
# 방지하기 위한 streamlit의 cache 기능을 사용하기 위합니다. 
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(st.session_state['modelname'])
    # model._make_predict_function()
    model.summary()
    return model


st.set_page_config("아파트 외관 균열 탐지기",page_icon='🔍️')

st.title('🔍️ 아파트 외관 균열 탐지')
with st.expander("상세 정보"):
    st.write('  - 마지막 업데이트 날짜: 2022-06-29')
    st.write('  - 학습에 사용한 이미지 크기: 160x160px')
    st.write('  - 학습에 사용한 실측과 이미지 픽셀간 스케일: 1px = 0.95mm')
    st.write('  - 사용문의: 정동균 책임( 내선:1456, 이메일:[dgjeong@gsenc.com](mailto:dgjeong@gsenc.com) )')
    st.write('---')


st.sidebar.header("사용법")
st.sidebar.write('''
### 1단계
- 균열을 탐지하고자하는 파일을 업로드 합니다.
- 업로드한 사진의 실제 폭을 입력합니다. 
- 학습에 사용된 이미지의 크랙사이즈와 매칭을 하기 위한 중요한 정보입니다. 
- 아주 정확할 필요는 없으나, 몇 십 센티미터 정도 수준으로 입력해야 합니다. 
---
### 2단계
- 탐지에 사용할 ML모델을 선택합니다.
- 특별한 이유가 없다면, 기본을 선택된 최신 모델을 선택합니다.
---
### 3단계
- 검색 정확도를 입력합니다. 
- 일반적으로는 **0.5**를 입력하는 것이 좋습니다. 
- 하지만, 학습을 위한 균열 사진 정보 취득이 목적인 경우, 정확도를 좀 더 낮춰, 학습시 균열 이미지로 혼동하는 자료를 추출하는 것이 추후 재학습을 통한 학습 정확도를 높이는데 도움이 됩니다. 
---
### 4단계
- 2, 3단계에서 선택된 조건을 확인 후, **[탐지 실행]** 버튼을 클릭합니다. 
- 사전에 정의된 기준 사이즈에 맞춰 입력된 사진을 스캐닝하면서 균열을 탐지 합니다. 
---
### 5단계
- AI가 균열을 정확히 탐지하였는지 확인을 합니다.
- 전체 이미지 사진을 다운로드 후, 상세 크랙 이미지와 비교하며 확인하는 것을 추천합니다. 
- 크랙이 아닌 경우, Checkbox를 클릭하여 Check 합니다. ~~(시스템 특성상, Check box를 클릭할 때 마다 reload하는 시간이 필요합니다.)~~
- 확인 과정이 모두 끝난 후, **[균열정보 업데이트]** 버튼을 클릭하여, 균열 정보를 업데이트 합니다.
- 이 과정에서 수정된 균열 정보는 재학습을 위해 학습 폴더로 복사됩니다. 
---
### 6단계
- 최종적으로 업데이트 된 이미지를 확인합니다.
- 필요시 최종 파일을 다운로드 받습니다.
- 다른 파일을 작


''')



#####################################################################
#### 1단계. 크랙탐지를 위한 파일을 업로드 후 화면에 표시합니다. 
#####################################################################
st.write("")
st.write("---")
st.write("#### |단계 1. 크랙을 탐지할 사진 파일을 선택하세요")
source_image = st.file_uploader("", type=['png','jpg','jpeg'])

if source_image is not None:
    img = Image.open(source_image)
    st.image(img, caption=f"원본파일({source_image.name})")
    width = st.number_input("   - 📣️ 사진의 대략적인 실제 폭을 M 단위로 입력하세요.(기본값: 7.7m) ",value=7.7)
    width = int(width * 1000) # m > mm
    st.session_state['isStep1Finished']=True
    
#####################################################################
#### 2단계. AI 탐지 모델을 받아와서 사용자가 선택할 수 있는 selectionbox
####       를 구현합니다. 
####       (사용자 입력 대기가 필요없어서, 바로 3단계가 표시됩니다.) 
#####################################################################
if 'isStep1Finished' in st.session_state:

    st.write("")
    st.write("---")
    st.write("#### |단계 2. 탐지에 사용한 AI 모델을 선택하세요.")
    st.write("  - 📣️ 기본 선택된 모델을 사용하실 것을 추천합니다.")
 
    model_dict = get_model_lists()
    dnn_model = st.selectbox("", model_dict.keys())
    st.session_state['isStep2Finished'] = True


#####################################################################
#### 3-1단계. 학습 실시를 위한 화면을 구성합니다. 
#####################################################################
btn_exec = False ## 학습 실행버튼을 block외에서 사용하기 위해 사전에 변수를 정의합니다. 
if 'isStep2Finished' in st.session_state:

    st.write("")
    st.write("---")
    st.write("#### |단계 3. 검색 정확도를 선택하세요")
    st.write("  - 💡️ 0.1은 학습된 균열과 유사한 것은 모두 찾음(최대한 많이 찾음)")
    st.write("  - 💡️ 0.9은 학습된 균열과 최대한 유사한 것만 찾음(최대한 정확히 찾음)")
    st.write("  - 📣️ 균열 사진 취합단계 에서는 0.3 사용을 추천합니다.")
    level = float(st.slider("",min_value=0.1, max_value=0.9,value=0.3))

    st.write("")
    st.write("---")
    st.write("#### |단계 4.균열 탐지")
    btn_exec = st.button("🔍️ 탐지 실행",disabled = st.session_state.disable_opt1)
    # st.write(btn_exec)
    st.session_state['isStep3Finished']=True

#####################################################################
#### 3-2단계. 학습버튼이 클릭된 후, 화면에 프로그래스 표시 에니메이션을 구현합니다.
#####################################################################
    col1, col2 = st.columns([2,1])
    col1.write("**아래 조건으로 탐지을 실시합니다.**")
    col1.write(f"   - 선택된 모델: {dnn_model}")
    col1.write(f"   - 탐지 정확도: {level}")
    col1.write(f"   - 사진의 실제폭(대략): {width}mm")

    if btn_exec:
        # st.write(btn_exec)
        st.session_state.disable_opt1 = True
        img_prog = col2.image(get_images("progress.gif"))

        my_bar =st.progress(0)
        img_nparray = np.array(img)
        st.session_state['modelname'] = model_dict[dnn_model]
        running_model = load_model()
        crack_files_list,crack_coord_list = crack_searcher(
            ml_model = running_model, 
            img = img_nparray, 
            source_file_name = os.path.splitext(source_image.name)[0], 
            threshold = level, 
            progress_bar = my_bar,
            file_postfix = st.session_state.file_postfix,
            img_width = width)
        st.session_state['crack_files_list'] = crack_files_list
        st.session_state['crack_coord_list'] = crack_coord_list
        
        img_prog.empty()
        st.success("👷️ 균열 탐지가 완료되었습니다.")
        st.session_state['isStep4Finished'] = True

        

#####################################################################
#### 5-1단계. 탐지된 균열 이미지 디테일을 표시해 줍니다.  
#####################################################################
if 'isStep4Finished' in st.session_state:

    crack_files_list = st.session_state['crack_files_list'] 
    crack_coord_list = st.session_state['crack_coord_list']
    st.write("")
    st.write("---")
    st.write("#### |단계 5.결과 확인")
    st.write("  - 💡️ 균열이 아닌 경우 Check 하고 아래에 있는 [업데이트] 버튼을 클릭하세요.")
    st.write("  - 📣️ 아래 균열 예상 부위가 넘버링된 사진을 다운로드 받아서 상세 이미지와 같이 비교하며 균열 여부를 판단하는 것을 추천합니다.")

    ## 사진을 표시하고 다운로드 할 수 있는 화면을 구성합니다. 
    with open(get_fullpath_markedorigin(crack_files_list[-1]),"rb") as f:
        st.download_button("💾️ 사진 다운로드", data=f,file_name=crack_files_list[-1],mime="image/png")
    st.image(get_fullpath_markedorigin(crack_files_list[-1])) ## 전체 마크업된 파일 보여주기

    ## 균열 상세 이미지 표시를 위한 변수를 설정합니다. 
    crack_counts = len(crack_files_list)-2
    nocrack_chk = []

    ## 균열 상세 이미지와 각 이미지당 1개씩 Checkbox를 화면에 표시합니다.
    for row in range(crack_counts):
        cols = st.columns([1,4])
        cols[0].write(f"## [{row+1}]")
        nocrack_chk.append(cols[0].checkbox(f"균열 아님.",key=row))
        cols[1].image(get_fullpath_crackfile(crack_files_list[row]),use_column_width='always')
        
    ## 균열 정보 업데이트 버튼과 진행바를 준비합니다. 
    st.write("")
    st.write("---")
    cols = st.columns([3,1,3])
    result = cols[0].button("🚧️ 균열 정보 업데이트",disabled = st.session_state.disable_opt2)
    my_bar2=st.progress(0.0)

#####################################################################
#### 5-2단계. 실제 균열인지 사용자의 확인을 받아 균열이미지 정보를 업데이트 합니다. 
#####################################################################
    if result:
        img = cols[1].image(get_images('loading.gif'))
        st.session_state.disable_opt2 = True
        confirmed_crack_coord = []
        st.session_state['isStep5Finished']=True
        
        ## 균열 중 사용자가 크랙으로 판별한 것과 아닌 것을 구분하는 작업을 합니다. 
        ## 이때 균열의 위치 정보도 동일하게 업데이트 합니다. 
        ## 사용자가 최종 확인한 정보는 results폴더에 저장되어, 재학습에 사용합니다.
        for i in range(crack_counts):
            if nocrack_chk[i]: # 크랙이 아닌 것으로 확인된 경우
                copy_to_confirmed_nocrack(crack_files_list[i])
            else: # 크랙이 맞는 것으로 확인된 경우
                copy_to_confirmed_crack(crack_files_list[i])
                confirmed_crack_coord.append(crack_coord_list[i])
                st.session_state["final_file"] = write_confirmed_crack(crack_files_list[-2], confirmed_crack_coord)
            
            my_bar2.progress(i/(crack_counts-1))


        result=False
        img.empty()
        st.success("👷️ 균열정보 업데이트가 완료되었습니다.")

#####################################################################
#### 6단계. 최종적으로 업데이트된 Full-shot 이미지를 표시하고 다운로드 할 수 
####       있는 화면을 구성합니다.  
#####################################################################
if ('isStep5Finished' in st.session_state) and ("final_file" in st.session_state):    
    final_file=st.session_state['final_file']
    filename_only = os.path.split(final_file)[1]
    st.write("")
    st.write("---")
    st.write("#### |단계6. 최종결과 확인 및 결과물 다운로드 받기")
    with open(final_file,"rb") as f:
        st.download_button("💾️ 사진 다운로드", data=f, file_name=filename_only,mime="image/png")
    st.image(final_file, caption=f"최종이미지({filename_only})")
    st.session_state.clear()
    st.success("👷️ 균열 탐지 작업이 모두 마무리 되었습니다. 다른 사진으로 1단계 부터 작업을 시작하세요.")
