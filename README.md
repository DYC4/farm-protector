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
<img width="863" alt="주제 선정 배경" src="https://user-images.githubusercontent.com/101931446/186854226-251f8d17-c26d-48de-8e5a-5f603cc94c10.png">
첫번째로 병해충으로 인한 농부의 피해가 증가하고 있습니다. 5년 사이 병해충 피해로
인한 농부의 피해면적이 7.6배 증가하였고, 2020년 농가 피해액은 약 342억원에 달했습니
다. 엎친 데 덮친 격으로, 기상이상, 농업환경 변화, 작물 재배양식 다양화로 돌발해충들이 등장하면서 심각성은 더욱 커지고 있습니다. 여기서 [돌발해충이란 시기나 장소에 한정되지 않고 돌발적으로 발생하여 농작물이나 일부 산림에 피해를 주는 토착 또는 외래해충](https://www.nongsaro.go.kr/portal/ps/psv/psvr/psvre/curationDtl.ps?menuId=PS03352&srchCurationNo=1158)을 말합니다. 

이러한 심각성에도 불구하고 우리나라에는 병해충 전문가가 부족한 상황입니다. [경상대학교 식물의학과 교수님](https://www.nongmin.com/opinion/OPP/SWE/PRO/334146/view)께서는 농기원이나 농기센터에 관련 전문가가 거의 없는 상황을 지적하기도 했습니다.
전문가를 고용하지 않고 검색을 통해 해충에 대해 알아보려고 하니, 화면에 보이는 것과 같은 상황을 마주하게 됩니다. 화면에서 보이는 것만 해도 사과 관련 해충이 수십개입니다. 이런 많은 종류의 해충을 일일이 확인하고 대조하여 찾기에는 시간과 노력이 많이 필요해보입니다.

<img width="862" alt="활용데이터" src="https://user-images.githubusercontent.com/101931446/186854010-32117259-bd01-4184-bf78-9538ae5fce21.png">
저희 서비스는 국가농작물병해충관리시스템에 등록된 해충 중, 돌발해충을 포함하여 작물 약 200종에서 가장 빈번하게 발생하는 상위 41종의 해충에 대한 판별을 진행합니다. 즉 저희 classification model의 class 개수는 41개입니다. 
학습에 사용된 해충 데이터는 저희가 직접 수집하였습니다. 해충 이미지가 많이 존재하지 않기 때문에 각 class 별로 train에 사용할 이미지 10개와 test에 사용할 이미지 3개, class 다 합쳐서 총 533개의 이미지를 수집하였습니다.

<img width="938" alt="시스템구성도" src="https://user-images.githubusercontent.com/101931446/186854765-a2291a1a-fbe9-4d63-bdd3-e85956291f7a.png">
- 백
파이썬 웹 프레임워크인 Flask 를 이용하여 백엔드를 구성했습니다. 내장 패키지의 수가 적어도, rest API를 통해 확장할 수 있으며 , Django에 비해 더 단순한 구조와 빠른 속도를 가지고 있기 때문입니다.  서비스 구동 과정을 간략히 말씀드리자면, 먼저 사용자가 웹 페이지에 해충 이미지 파일을 업로드하면 모델을 통해 웹서버에 저장된 이미지에 대한 예상 라벨을 얻습니다. 이후 라벨에 해당하는 해충과 사용자가 선택한 작물 종류에 맞는 농약을 데이터베이스에서 읽어와 웹 페이지를 통해 리턴하도록 하였습니다.   
- 서버
서버 연결은, 아마존의 무료 서버를 이용 할 수 있는 AWS EC2를 이용해 로컬에서만 사용 할 수 있는 플라스크 웹 서버를  배포 하였습니다.
인스턴트는 우분투 리눅스 서버를 사용 했으며, 각종 패키지들을 다 가상환경에 import 한 후 git에 올려놓은 저희의 풀스택 자료를 EC2에 가져온 후 서버를 구동 하는데 성공 하였습니다.
EC2의 인스턴스도 하나의 서버이기 때문에 IP가 존재 하는데, 같은 인스턴스를 사용함에도 재시작 할 때 IP가 새로 할당 되는 문제를 해결 하기 위해 탄력적 IP주소를 할당 해 줘서 문제를 해결 했습니다.
- 프론트
프론트에 사용한 언어는 HTML, Javascript, CSS 를 이용하여 구성했습니다.

<img width="863" alt="서비스 소개" src="https://user-images.githubusercontent.com/101931446/186854282-2b338c83-3cb5-44d6-873d-e52a152988db.png">
첫번째로 사용자가 ‘우리가 하는 일’을 클릭하면 전체적인 서비스의 설명과 개발하게 된 이유 등을 확인할 수 있습니다.
두번째는 서비스를 구성한 사람들의 역할을 확인 가능한 페이지입니다. 이미지 클릭시, 개인 깃허브로 연결되며 구현한 기술들을 확인할 수 있습니다.
다음으로는 사용자가 서비스를 이용하는 페이지입니다. 아래의 폼에 사용자가 사용자의 정보와 해충으로 인해 현재 피해받고 있는 대상 작물의 이름을 입력합니다. 이때 작물 이름을 정확하게 확인하고 싶다면 <작물 이름을 확인해보세요> 버튼을 누르면 됩니다. 해당 리스트는 가나다 순서대로 작물 이름이 정렬되어 있기 때문에 사용자가 편리하게 확인할 수 있습니다. 사용자는 갤러리에서 확인하고 싶은 해충사진을 선택해 첨부해주시면 됩니다. 
마지막으로 확인을 누른다면 해당 사진, 해충 이름, 작물 이름, 그리고 5개의 농약 추천이 포함된 결과지가 출력됩니다.


