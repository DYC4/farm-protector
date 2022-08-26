# 해충 판독에 대한 영상 처리 모델과 농약 추천 서비스 구현
<img width="706" alt="main" src="https://user-images.githubusercontent.com/101931446/186851768-403404ce-58e5-4457-a0e7-6d31b49fd5d0.png">
2022 데이터청년캠퍼스 고려대학교 4조 프로젝트

## OUR PROJECT 
- 주제 : 해충 판별 및 농약 안내 시스템
- 팀원 : 김누리, 서수원, 송윤아, 안정민, 임채명, 장세음 

## 사용 STACK
- Model : [Few-Shot Learning](https://github.com/DYC4/farm-protector/blob/main/models/few-shot.ipynb), [Transfer Learning](https://github.com/DYC4/farm-protector/blob/main/models/transfer-learning.ipynb), [Domain Adaptation](https://github.com/DYC4/farm-protector/blob/main/models/DA.ipynb)
- Back-end : Flask, Python, MySQL
- Front-end : HTML, CSS, Java Script

## OUR ROLE
- 김누리 : Back-end
- 서수원 : Few Shot Learning
- 송윤아 : Domain Adaptation
- 안정민 : Front-end
- 임채명 : Transfer-Learning
- 장세음 : Back-end

## OUR SERVICE
### 주제 선정 배경
<img width="863" alt="주제 선정 배경" src="https://user-images.githubusercontent.com/101931446/186854226-251f8d17-c26d-48de-8e5a-5f603cc94c10.png">
첫번째로 병해충으로 인한 농부의 피해가 증가하고 있습니다. 5년 사이 병해충 피해로 인한 농부의 피해면적이 7.6배 증가
2020년 농가 피해액은 약 342억원에 달했습니다.


엎친 데 덮친 격으로, 기상이상, 농업환경 변화, 작물 재배양식 다양화로 돌발해충들이 등장하면서 심각성은 더욱 커지고 있습니다. 여기서 돌발해충이란 시기나 장소에 한정되지 않고 돌발적으로 발생하여 농작물이나 일부 산림에 피해를 주는 토착 또는 외래해충을 말합니다. 



이러한 심각성에도 불구하고 우리나라에는 병해충 전문가가 부족한 상황입니다. [경상대학교 식물의학과 교수님](https://www.nongmin.com/opinion/OPP/SWE/PRO/334146/view)께서는 농기원이나 농기센터에 관련 전문가가 거의 없는 상황을 지적하기도 했습니다.



전문가를 고용하지 않고 검색을 통해 해충에 대해 알아보려고 하니, 화면에 보이는 것과 같은 상황을 마주하게 됩니다. 화면에서 보이는 것만 해도 사과 관련 해충이 수십개입니다. 이런 많은 종류의 해충을 일일이 확인하고 대조하여 찾기에는 시간과 노력이 많이 필요해보입니다.

### 활용 데이터
<img width="862" alt="활용데이터" src="https://user-images.githubusercontent.com/101931446/186854010-32117259-bd01-4184-bf78-9538ae5fce21.png">

### 시스템 구성도
<img width="938" alt="시스템구성도" src="https://user-images.githubusercontent.com/101931446/186854765-a2291a1a-fbe9-4d63-bdd3-e85956291f7a.png">

### 서비스 소개
<img width="863" alt="서비스 소개" src="https://user-images.githubusercontent.com/101931446/186854282-2b338c83-3cb5-44d6-873d-e52a152988db.png">


