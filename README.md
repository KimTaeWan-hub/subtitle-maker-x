# Subtitle Maker X

영상에서 자동으로 자막을 생성하는 웹 애플리케이션입니다.

## 프로젝트 개요

Subtitle Maker X는 AI 기반 음성 인식 기술을 활용하여 영상 파일에서 자동으로 자막을 생성하는 풀스택 웹 애플리케이션입니다. 
사용자는 웹 브라우저를 통해 영상을 업로드하고, 실시간으로 전처리 과정을 확인하며, 생성된 자막을 편집하고 다운로드할 수 있습니다.

## 주요 기능

### 🎬 영상 처리
- 다양한 형식의 영상 파일 지원 (MP4, AVI, MOV, MKV, WEBM)
- 드래그 앤 드롭 업로드
- 웹 기반 영상 재생

### 🎤 음성 인식
- WhisperX 기반 고정밀 음성 인식
- 다국어 지원 (한국어, 영어, 일본어, 중국어 등 90개 이상)
- 자동 언어 감지
- 단어 수준 타임스탬프 정렬

### 🔊 전처리
- Silero VAD를 사용한 음성 구간 탐지
- 음성이 아닌 구간 자동 제거
- 실시간 전처리 상태 업데이트 (WebSocket)

### 👥 화자 분리 (선택사항)
- 여러 화자 자동 구분
- 각 자막에 화자 레이블 표시

### ✏️ 자막 편집
- 웹 기반 실시간 자막 편집
- 자막 클릭 시 해당 시간으로 영상 이동
- 영상과 자막 동기화

### 💾 자막 다운로드
- 다양한 형식 지원: SRT, VTT, TXT, JSON
- 영상 편집 프로그램 호환

## 기술 스택

### 백엔드
- **FastAPI**: 고성능 웹 프레임워크
- **WhisperX**: 음성 인식 및 타임스탬프 정렬
- **Silero VAD**: 음성 구간 탐지
- **FFmpeg**: 오디오/비디오 처리
- **PyTorch**: 딥러닝 프레임워크
- **WebSocket**: 실시간 상태 업데이트

### 프론트엔드
- **React 19**: UI 프레임워크
- **HTML5 Video**: 영상 재생
- **WebSocket API**: 실시간 통신
- **Fetch API**: REST API 통신

## 프로젝트 구조

```
subtitle-maker-x/
├── subtitle_maker_x_backend/     # 백엔드 (FastAPI)
│   ├── main.py                   # FastAPI 서버
│   ├── audio_extractor.py        # 오디오 추출
│   ├── audio_detector.py         # 음성 구간 탐지
│   ├── subtitle_processor.py     # 자막 생성 통합
│   ├── requirements.txt          # Python 패키지
│   ├── uploads/                  # 업로드된 영상
│   ├── outputs/                  # 생성된 자막 및 오디오
│   └── README.md                 # 백엔드 문서
│
├── subtitle_maker_x_frontend/    # 프론트엔드 (React)
│   ├── src/
│   │   ├── App.js               # 메인 앱
│   │   └── components/          # React 컴포넌트
│   │       ├── VideoUploader.js
│   │       ├── VideoPlayer.js
│   │       ├── PreprocessingStatus.js
│   │       ├── SubtitleEditor.js
│   │       └── SubtitleDownloader.js
│   ├── package.json             # Node.js 패키지
│   └── README.md                # 프론트엔드 문서
│
└── README.md                     # 이 파일
```

## 빠른 시작

### 시스템 요구사항

- **Python**: 3.8 이상
- **Node.js**: 16 이상
- **FFmpeg**: 시스템에 설치 필요
- **GPU**: CUDA 지원 GPU 권장 (CPU도 가능하지만 느림)

### 1. FFmpeg 설치

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# Windows
# https://ffmpeg.org/download.html 에서 다운로드
```

### 2. 백엔드 설정 및 실행

```bash
# 백엔드 디렉토리로 이동
cd subtitle_maker_x_backend

# Python 패키지 설치
pip install -r requirements.txt

# (선택사항) 화자 분리를 위한 Hugging Face 토큰 설정
# .env 파일 생성 후 다음 내용 추가:
# HF_TOKEN=your_huggingface_token_here

# 서버 실행
python main.py
```

백엔드 서버가 http://localhost:8000 에서 실행됩니다.

### 3. 프론트엔드 설정 및 실행

```bash
# 프론트엔드 디렉토리로 이동
cd subtitle_maker_x_frontend

# Node.js 패키지 설치
npm install

# 개발 서버 실행
npm start
```

프론트엔드가 http://localhost:3000 에서 실행되며 자동으로 브라우저가 열립니다.

## 사용 방법

1. **영상 업로드**
   - 웹 브라우저에서 http://localhost:3000 접속
   - 영상 파일을 드래그 앤 드롭 또는 파일 선택

2. **전처리 대기**
   - 업로드 후 자동으로 전처리 시작
   - 진행 상황이 실시간으로 표시됨
   - 오디오 추출 → 음성 구간 탐지 → 전처리

3. **자막 생성**
   - 전처리 완료 후 "자막 생성하기" 버튼 클릭
   - 음성 인식 진행 (시간이 소요될 수 있음)

4. **자막 편집**
   - 우측 패널에서 자막 확인 및 수정
   - 자막 클릭 시 해당 시간으로 영상 이동

5. **자막 다운로드**
   - 원하는 형식(SRT, VTT, TXT, JSON) 선택
   - 다운로드 버튼 클릭

## 처리 파이프라인

```
영상 업로드 (MP4, AVI 등)
    ↓
[1단계] 오디오 추출
    → FFmpeg로 WAV 16kHz 모노 변환
    ↓
[2단계] 음성 구간 탐지
    → Silero VAD로 음성 구간 탐지
    → 음성이 아닌 구간 음소거
    ↓
[3단계] 음성 인식
    → WhisperX로 음성을 텍스트로 변환
    ↓
[4단계] 타임스탬프 정렬
    → 단어 수준 정밀 타임스탬프
    ↓
[5단계] 화자 분리 (선택적)
    → 각 세그먼트에 화자 레이블 할당
    ↓
자막 파일 생성 (SRT, VTT, TXT, JSON)
```

## API 엔드포인트

### REST API

- `POST /api/upload` - 영상 업로드
- `GET /api/video/{file_id}` - 영상 스트리밍
- `POST /api/transcribe/{file_id}` - 자막 생성
- `GET /api/download/{file_id}/{format}` - 자막 다운로드
- `GET /api/status/{file_id}` - 처리 상태 조회
- `DELETE /api/file/{file_id}` - 파일 삭제

### WebSocket

- `ws://localhost:8000/ws/{file_id}` - 전처리 상태 실시간 업데이트

API 문서: http://localhost:8000/docs

## 지원 언어

WhisperX는 90개 이상의 언어를 지원합니다:
- 한국어 (ko)
- 영어 (en)
- 일본어 (ja)
- 중국어 (zh)
- 스페인어 (es)
- 프랑스어 (fr)
- 독일어 (de)
- 기타 등등

언어 코드를 지정하지 않으면 자동으로 감지됩니다.

## 모델 정보

현재 백엔드는 WhisperX `base` 모델을 기본으로 사용합니다.
더 높은 정확도가 필요한 경우 `main.py`에서 모델을 변경할 수 있습니다:

| 모델 | 크기 | 속도 | 정확도 |
|------|------|------|--------|
| tiny | 39M | 매우 빠름 | 낮음 |
| base | 74M | 빠름 | 보통 |
| small | 244M | 보통 | 좋음 |
| medium | 769M | 느림 | 매우 좋음 |
| large-v2 | 1550M | 매우 느림 | 최고 |

## 문제 해결

### FFmpeg 오류
```bash
# FFmpeg 설치 확인
ffmpeg -version
```

### CUDA 오류
```bash
# GPU 사용 가능 여부 확인
python -c "import torch; print(torch.cuda.is_available())"
```
GPU가 없으면 자동으로 CPU 모드로 동작합니다.

### 포트 충돌
- 백엔드: 8000 포트 사용
- 프론트엔드: 3000 포트 사용

다른 포트를 사용하려면:
```bash
# 백엔드
uvicorn main:app --host 0.0.0.0 --port 8001

# 프론트엔드
PORT=3001 npm start
```

## 성능 최적화

- **GPU 사용**: CUDA 지원 GPU 사용 시 처리 속도 대폭 향상
- **모델 선택**: 작은 모델(tiny, base) 사용 시 속도 향상
- **배치 크기**: GPU 메모리에 따라 조정 가능

## 라이선스

이 프로젝트는 다음 오픈소스 라이브러리를 사용합니다:
- FFmpeg (LGPL/GPL)
- Silero VAD (MIT)
- WhisperX (BSD-4-Clause)
- OpenAI Whisper (MIT)
- FastAPI (MIT)
- React (MIT)

## 참고 자료

- [WhisperX GitHub](https://github.com/m-bain/whisperX)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [FFmpeg](https://ffmpeg.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://react.dev/)
