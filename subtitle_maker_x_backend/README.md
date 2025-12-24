# Subtitle Maker X - Backend

영상에서 자동으로 자막을 생성하는 백엔드 시스템입니다.

## 주요 기능

1. **FastAPI 웹 서버** (`main.py`)
   - RESTful API 제공
   - WebSocket을 통한 실시간 전처리 상태 업데이트
   - 프론트엔드와의 통신
   - 파일 업로드/다운로드 관리

2. **오디오 추출** (`audio_extractor.py`)
   - 영상 파일에서 오디오 추출
   - FFmpeg 기반 고품질 오디오 변환
   - 볼륨 정규화 및 모노 채널 변환

3. **음성 구간 탐지** (`audio_detector.py`)
   - Silero VAD를 사용한 음성 구간 탐지
   - 음성이 아닌 구간 음소거 처리
   - 음성 통계 분석

4. **자막 생성** (`subtitle_processor.py`)
   - WhisperX를 사용한 고정밀 음성 인식
   - 단어 수준 타임스탬프 정렬
   - 화자 분리 (Speaker Diarization)
   - 다양한 형식 지원 (SRT, VTT, TXT, JSON)

## 설치 방법

### 1. 시스템 요구사항

- Python 3.8 이상
- FFmpeg (시스템에 설치 필요)
- CUDA 지원 GPU (선택사항, CPU도 가능하지만 느림)

### 2. FFmpeg 설치

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# Windows
# https://ffmpeg.org/download.html 에서 다운로드
```

### 3. Python 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. Hugging Face 토큰 설정 (화자 분리용)

화자 분리 기능을 사용하려면 Hugging Face 토큰이 필요합니다:

1. https://huggingface.co/settings/tokens 에서 토큰 생성
2. 환경 변수로 설정:

```bash
export HF_TOKEN='your_huggingface_token_here'
```

## 빠른 시작 (FastAPI 서버)

### 1. 서버 실행

```bash
# macOS/Linux
./start_server.sh

# Windows
start_server.bat

# 또는 직접 실행
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

서버가 시작되면:
- **API 서버**: http://localhost:8000
- **API 문서 (Swagger UI)**: http://localhost:8000/docs
- **API 문서 (ReDoc)**: http://localhost:8000/redoc

### 2. 프론트엔드 연결

프론트엔드에서 다음 엔드포인트를 사용하여 통신합니다:

```javascript
// 영상 업로드
const formData = new FormData();
formData.append('file', videoFile);

const response = await fetch('http://localhost:8000/api/upload', {
  method: 'POST',
  body: formData
});

const { fileId } = await response.json();

// WebSocket으로 전처리 상태 수신
const ws = new WebSocket(`ws://localhost:8000/ws/${fileId}`);

ws.onmessage = (event) => {
  const { status, progress, message } = JSON.parse(event.data);
  console.log(`${status}: ${message} (${progress}%)`);
};

// 자막 생성
const transcribeResponse = await fetch(
  `http://localhost:8000/api/transcribe/${fileId}`,
  { method: 'POST' }
);

const { segments } = await transcribeResponse.json();

// 자막 다운로드
window.location.href = `http://localhost:8000/api/download/${fileId}/srt`;
```

## API 엔드포인트

### POST /api/upload
영상 파일 업로드

**요청**:
- `file`: 영상 파일 (multipart/form-data)

**응답**:
```json
{
  "fileId": "uuid-string",
  "filename": "video.mp4",
  "message": "업로드 완료. 전처리가 시작되었습니다."
}
```

### GET /api/video/{file_id}
업로드된 영상 스트리밍

### WebSocket /ws/{file_id}
전처리 상태 실시간 업데이트

**메시지 형식**:
```json
{
  "status": "processing",
  "progress": 50,
  "message": "음성 구간 탐지 중..."
}
```

### POST /api/transcribe/{file_id}
자막 생성

**응답**:
```json
{
  "segments": [
    {
      "start": 0.512,
      "end": 3.264,
      "text": "안녕하세요",
      "speaker": "SPEAKER_00"
    }
  ],
  "language": "ko",
  "message": "자막 생성이 완료되었습니다."
}
```

### GET /api/download/{file_id}/{format}
자막 파일 다운로드

**형식**: `srt`, `vtt`, `txt`, `json`

### GET /api/status/{file_id}
파일 처리 상태 조회

### DELETE /api/file/{file_id}
파일 및 관련 데이터 삭제

## 사용 방법 (Python API)

### 기본 사용 예시

```python
from subtitle_processor import SubtitleProcessor
import os

# Hugging Face 토큰 설정
hf_token = os.environ.get("HF_TOKEN")

# 프로세서 초기화
processor = SubtitleProcessor(
    whisper_model="large-v2",  # 모델: tiny, base, small, medium, large-v2
    device="cuda",              # "cuda" 또는 "cpu"
    compute_type="float16",     # "float16", "int8" 등
    hf_token=hf_token          # 화자 분리용 토큰
)

# 영상 처리
result = processor.process_video(
    video_path="input_video.mp4",
    output_dir="output",
    language="ko",              # 언어 코드 (None이면 자동 감지)
    enable_diarization=True,    # 화자 분리 활성화
    min_speakers=2,             # 최소 화자 수
    max_speakers=5              # 최대 화자 수
)

# 자막 파일 생성
processor.export_subtitles(result['segments'], "output/subtitle.srt", format="srt")
processor.export_subtitles(result['segments'], "output/subtitle.vtt", format="vtt")

# 메모리 정리
processor.cleanup()
```

### 명령줄에서 테스트

```bash
# 기본 테스트
python subtitle_processor.py test_video.mp4

# 또는 직접 경로 지정
python subtitle_processor.py /path/to/your/video.mp4
```

### 개별 모듈 사용

#### 1. 오디오 추출만

```python
from audio_extractor import AudioProcessor

processor = AudioProcessor(sample_rate=16000)
processor.extract_audio("video.mp4", "output_audio.wav")
```

#### 2. 음성 구간 탐지만

```python
from audio_detector import VoiceActivityDetector

detector = VoiceActivityDetector(sample_rate=16000, threshold=0.5)

# 음성 구간 탐지
segments = detector.detect_voice_segments("audio.wav")

# 통계 확인
stats = detector.get_speech_statistics("audio.wav", segments)

# 음성만 추출 (음소거 처리)
detector.extract_speech_only_audio("audio.wav", "speech_only.wav", segments)
```

## 전체 파이프라인

```
영상 파일 (MP4, AVI 등)
    ↓
[1단계] audio_extractor.py
    → 오디오 추출 (WAV 16kHz 모노)
    ↓
[2단계] audio_detector.py
    → 음성 구간 탐지 (Silero VAD)
    → 음성이 아닌 구간 음소거
    ↓
[3단계] WhisperX 음성 인식
    → 음성을 텍스트로 변환
    ↓
[4단계] 타임스탬프 정렬
    → 단어 수준 정밀 타임스탬프
    ↓
[5단계] 화자 분리 (선택적)
    → 각 세그먼트에 화자 레이블 할당
    ↓
자막 파일 (SRT, VTT, TXT, JSON)
```

## 결과 데이터 구조

```python
{
    "segments": [
        {
            "start": 0.512,
            "end": 3.264,
            "text": "안녕하세요",
            "speaker": "SPEAKER_00",  # 화자 분리 활성화 시
            "words": [
                {
                    "word": "안녕하세요",
                    "start": 0.512,
                    "end": 3.264,
                    "score": 0.95
                }
            ]
        }
    ],
    "language": "ko",
    "audio_path": "output/video_audio.wav",
    "preprocessed_audio_path": "output/video_preprocessed.wav",
    "vad_segments": [...],
    "statistics": {
        "total_segments": 42,
        "detected_language": "ko",
        "num_speakers": 2,
        "speakers": ["SPEAKER_00", "SPEAKER_01"],
        "vad_statistics": {
            "total_duration": 120.5,
            "speech_duration": 98.3,
            "silence_duration": 22.2,
            "speech_ratio": 0.8157,
            "num_segments": 15
        }
    }
}
```

## 지원 언어

WhisperX는 다음 언어를 지원합니다:
- 한국어 (ko)
- 영어 (en)
- 일본어 (ja)
- 중국어 (zh)
- 스페인어 (es)
- 프랑스어 (fr)
- 독일어 (de)
- 기타 90개 이상의 언어

언어 코드를 지정하지 않으면 자동으로 감지됩니다.

## 모델 선택 가이드

| 모델 | 크기 | 속도 | 정확도 | 권장 용도 |
|------|------|------|--------|-----------|
| tiny | 39M | 매우 빠름 | 낮음 | 빠른 테스트 |
| base | 74M | 빠름 | 보통 | 일반 테스트 |
| small | 244M | 보통 | 좋음 | 균형잡힌 사용 |
| medium | 769M | 느림 | 매우 좋음 | 높은 정확도 필요 |
| large-v2 | 1550M | 매우 느림 | 최고 | 최고 품질 |

## 성능 최적화

### GPU 메모리가 부족한 경우

```python
processor = SubtitleProcessor(
    whisper_model="base",      # 작은 모델 사용
    compute_type="int8",       # 양자화 사용
    batch_size=8               # 배치 크기 줄이기
)
```

### CPU만 사용하는 경우

```python
processor = SubtitleProcessor(
    whisper_model="base",
    device="cpu",
    compute_type="int8"
)
```

## 문제 해결

### FFmpeg 오류

```bash
# FFmpeg가 설치되어 있는지 확인
ffmpeg -version

# 설치되지 않았다면 설치
brew install ffmpeg  # macOS
```

### CUDA 오류

```bash
# PyTorch CUDA 버전 확인
python -c "import torch; print(torch.cuda.is_available())"

# False가 나오면 CPU 모드로 사용
```

### 화자 분리 오류

```bash
# Hugging Face 토큰이 설정되어 있는지 확인
echo $HF_TOKEN

# 설정되지 않았다면
export HF_TOKEN='your_token_here'
```

## 라이선스

이 프로젝트는 다음 오픈소스 라이브러리를 사용합니다:
- FFmpeg (LGPL/GPL)
- Silero VAD (MIT)
- WhisperX (BSD-4-Clause)
- OpenAI Whisper (MIT)

## 참고 자료

- [WhisperX GitHub](https://github.com/m-bain/whisperX)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [FFmpeg](https://ffmpeg.org/)

