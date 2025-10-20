# Google Colab에서 기존 리포지토리 구조 복제하기

## 방법 1: GitHub에서 직접 클론 (권장)

### 1. GitHub에 리포지토리 업로드

먼저 현재 프로젝트를 GitHub에 업로드하세요:

```bash
# 로컬에서 실행
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/isml-assignment.git
git push -u origin main
```

### 2. Colab에서 클론

Colab 노트북 첫 번째 셀:

```python
# GitHub에서 리포지토리 클론
!git clone https://github.com/yourusername/isml-assignment.git

# 작업 디렉토리 변경
import os
os.chdir('/content/isml-assignment')

# 현재 디렉토리 확인
!pwd
!ls -la
```

## 방법 2: Google Drive 연동

### 1. Google Drive에 프로젝트 업로드

1. Google Drive에 접속
2. `isml-assignment` 폴더 생성
3. 모든 파일을 드래그 앤 드롭으로 업로드

### 2. Colab에서 Drive 마운트

```python
# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 프로젝트 디렉토리로 이동
import os
os.chdir('/content/drive/MyDrive/isml-assignment')

# 파일 구조 확인
!ls -la
```

## 방법 3: 파일 직접 업로드

### 1. 필요한 파일들 업로드

```python
from google.colab import files
import os

# 디렉토리 생성
!mkdir -p isml-assignment
os.chdir('/content/isml-assignment')

# 파일 업로드 (여러 번 실행)
uploaded = files.upload()

# 업로드된 파일 확인
!ls -la
```

## 방법 4: ZIP 파일로 업로드

### 1. 로컬에서 ZIP 파일 생성

```bash
# 프로젝트 폴더를 ZIP으로 압축
zip -r isml-assignment.zip isml-assignment/
```

### 2. Colab에서 압축 해제

```python
from google.colab import files
import zipfile

# ZIP 파일 업로드
uploaded = files.upload()

# 압축 해제
with zipfile.ZipFile('isml-assignment.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/')

# 작업 디렉토리 변경
import os
os.chdir('/content/isml-assignment')

# 구조 확인
!find . -type f -name "*.py" | head -10
```

## 완전한 Colab 설정 코드

```python
# 1. 필요한 라이브러리 설치
!pip install torch torchvision torchaudio
!pip install matplotlib tqdm
!pip install tiktoken

# 2. GitHub에서 클론 (방법 1 사용 시)
!git clone https://github.com/yourusername/isml-assignment.git

# 3. 작업 디렉토리 설정
import os
os.chdir('/content/isml-assignment')

# 4. 프로젝트 구조 확인
print("=== 프로젝트 구조 ===")
!find . -type f -name "*.py" | sort
print("\n=== 주요 디렉토리 ===")
!ls -la

# 5. 환경 확인
import torch
print(f"\n=== 환경 정보 ===")
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 6. 기존 코드 실행
print("\n=== 기존 코드 실행 ===")
exec(open('language_model_comparison.py').read())
```

## 프로젝트 구조 유지 확인

실행 후 다음과 같은 구조가 유지되어야 합니다:

```
/content/isml-assignment/
├── bpeasy/
│   ├── bpeasy/
│   │   ├── __init__.py
│   │   ├── tokenizer.py
│   │   └── convert.py
│   └── requirements.txt
├── Char_Transformer_Language_Model/
│   ├── char_transformer_language_model.ipynb
│   ├── char_transformer_language_model.py
│   ├── input.txt
│   └── README.md
├── tiktoken/
│   ├── tiktoken/
│   └── tests/
├── language_model_comparison.py
├── colab_language_model_comparison.py
└── COLAB_SETUP_GUIDE.md
```

## 권장사항

1. **GitHub 사용**: 가장 안정적이고 버전 관리 가능
2. **Drive 연동**: 파일 편집이 필요한 경우
3. **ZIP 업로드**: 빠른 테스트용

이렇게 하면 Colab에서도 기존 리포지토리 구조를 그대로 유지하면서 실험할 수 있습니다!
