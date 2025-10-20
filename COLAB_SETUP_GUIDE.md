# Google Colab에서 언어 모델 비교 실험 실행 가이드

## 1. Google Colab 설정

### GPU 런타임 활성화

1. [Google Colab](https://colab.research.google.com/) 접속
2. 새 노트북 생성
3. `런타임` → `런타임 유형 변경`
4. 하드웨어 가속기: **GPU** 선택 (T4 또는 더 좋은 GPU)
5. 저장

## 2. 필요한 라이브러리 설치

첫 번째 셀에 다음 코드 실행:

```python
# 필요한 라이브러리 설치
!pip install torch torchvision torchaudio
!pip install matplotlib tqdm
!pip install tiktoken

# 설치 확인
import torch
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## 3. 코드 실행

두 번째 셀에 전체 코드를 복사하여 실행하거나, 파일을 업로드하여 실행:

### 방법 1: 직접 코드 실행

- 제공된 `colab_language_model_comparison.py` 파일의 내용을 복사하여 Colab 셀에 붙여넣기

### 방법 2: 파일 업로드

```python
# 파일 업로드
from google.colab import files
uploaded = files.upload()

# 업로드된 파일 실행
exec(open('colab_language_model_comparison.py').read())
```

## 4. 실험 결과

실행하면 다음과 같은 결과를 얻을 수 있습니다:

1. **토크나이저 훈련**: 단어 기반 토크나이저 생성
2. **모델 훈련**: RNN과 Transformer 모델 동시 훈련
3. **훈련 곡선**: 손실 변화 시각화
4. **텍스트 생성**: 두 모델의 생성 결과 비교
5. **성능 비교**: 최종 검증 손실 비교

## 5. 하이퍼파라미터 조정

Colab에서 더 빠른 실험을 위해 조정된 설정:

- **배치 크기**: 64 (GPU 메모리에 맞게 증가)
- **컨텍스트 길이**: 256 (더 긴 시퀀스)
- **임베딩 차원**: 512 (더 큰 모델)
- **최대 반복**: 5000 (빠른 수렴을 위해)
- **평가 간격**: 250 (더 자주 모니터링)

## 6. 예상 실행 시간

- **GPU T4**: 약 30-45분
- **GPU V100**: 약 20-30분
- **GPU A100**: 약 15-20분

## 7. 결과 해석

### 성능 지표

- **검증 손실**: 낮을수록 좋음
- **텍스트 품질**: 생성된 텍스트의 일관성과 의미
- **수렴 속도**: 손실이 안정화되는 속도

### 예상 결과

- Transformer가 일반적으로 더 나은 성능을 보임
- RNN은 더 빠르게 수렴할 수 있음
- 긴 시퀀스에서 Transformer의 장점이 더 두드러짐

## 8. 추가 실험 아이디어

1. **다른 데이터셋**: 더 큰 텍스트 데이터 사용
2. **하이퍼파라미터 튜닝**: 학습률, 배치 크기 등 조정
3. **모델 크기 비교**: 더 큰/작은 모델 실험
4. **다른 RNN 구조**: GRU, BiLSTM 등 시도

## 9. 문제 해결

### 메모리 부족 오류

```python
# 배치 크기 줄이기
BATCH_SIZE = 32  # 또는 16

# 모델 크기 줄이기
N_EMBD = 256
N_LAYER = 4
```

### 실행 시간이 너무 긴 경우

```python
# 반복 횟수 줄이기
MAX_ITERS = 2000

# 평가 간격 늘리기
EVAL_INTERVAL = 500
```

## 10. 결과 저장

```python
# 모델 저장
torch.save(rnn_model.state_dict(), 'rnn_model.pt')
torch.save(transformer_model.state_dict(), 'transformer_model.pt')

# 결과 다운로드
from google.colab import files
files.download('training_comparison.png')
files.download('rnn_model.pt')
files.download('transformer_model.pt')
```

이 가이드를 따라하면 Colab에서 효율적으로 언어 모델 비교 실험을 수행할 수 있습니다!
