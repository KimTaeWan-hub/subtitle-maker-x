"""
FastAPI 기반 자막 생성 백엔드 서버
프론트엔드와 통신하여 영상 업로드, 전처리, 자막 생성 기능 제공
"""

import os
import uuid
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

from subtitle_processor import SubtitleProcessor

# .env 파일 로드
load_dotenv()
logger_init = logging.getLogger(__name__)
hf_token_status = os.environ.get("HF_TOKEN")
if hf_token_status:
    logger_init.info(f"✅ HF_TOKEN 로드 완료 (길이: {len(hf_token_status)} 문자)")
else:
    logger_init.warning("⚠️  HF_TOKEN이 설정되지 않았습니다. 화자 분리 기능이 비활성화됩니다.")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="Subtitle Maker X API",
    description="영상 자막 자동 생성 API",
    version="1.0.0"
)

# CORS 설정 (프론트엔드와 통신)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 개발 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 설정
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# 전역 변수
file_storage: Dict[str, Dict] = {}  # 파일 정보 저장
websocket_connections: Dict[str, WebSocket] = {}  # WebSocket 연결 관리
processing_status: Dict[str, Dict] = {}  # 처리 상태 관리

# SubtitleProcessor 인스턴스 (싱글톤)
processor: Optional[SubtitleProcessor] = None


def get_processor() -> SubtitleProcessor:
    """SubtitleProcessor 인스턴스를 가져옵니다 (지연 초기화)"""
    global processor
    if processor is None:
        hf_token = os.environ.get("HF_TOKEN")
        
        # GPU 사용 가능 여부 확인
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        
        logger.info(f"SubtitleProcessor 초기화 중 (device: {device})")
        
        processor = SubtitleProcessor(
            whisper_model="base",  # 기본 모델 (필요시 변경)
            device=device,
            compute_type=compute_type,
            hf_token=hf_token
        )
        
        logger.info("SubtitleProcessor 초기화 완료")
    
    return processor


async def send_websocket_update(file_id: str, status: str, progress: int, message: str):
    """WebSocket을 통해 전처리 상태 업데이트 전송"""
    if file_id in websocket_connections:
        try:
            await websocket_connections[file_id].send_json({
                "status": status,
                "progress": progress,
                "message": message
            })
            logger.info(f"WebSocket 업데이트 전송: {file_id} - {message} ({progress}%)")
        except Exception as e:
            logger.error(f"WebSocket 전송 실패: {str(e)}")


async def preprocess_video_background(file_id: str, video_path: str, output_dir: str):
    """백그라운드에서 영상 전처리 수행"""
    try:
        logger.info(f"전처리 시작: {file_id}")
        
        # 상태 초기화
        processing_status[file_id] = {
            "status": "processing",
            "progress": 0,
            "message": "전처리 시작"
        }
        await send_websocket_update(file_id, "processing", 0, "전처리 시작")
        
        proc = get_processor()
        
        # 1단계: 오디오 추출
        await send_websocket_update(file_id, "processing", 20, "오디오 추출 중...")
        audio_path = os.path.join(output_dir, f"{file_id}_audio.wav")
        proc.audio_processor.extract_audio(video_path, audio_path)
        
        # 2단계: 음성 구간 탐지
        await send_websocket_update(file_id, "processing", 50, "음성 구간 탐지 중...")
        vad_segments = proc.vad.detect_voice_segments(audio_path)
        
        # 3단계: 전처리 (음소거 처리)
        await send_websocket_update(file_id, "processing", 80, "전처리 중...")
        preprocessed_audio_path = os.path.join(output_dir, f"{file_id}_preprocessed.wav")
        proc.vad.extract_speech_only_audio(audio_path, preprocessed_audio_path, vad_segments)
        
        # 전처리 완료
        processing_status[file_id] = {
            "status": "completed",
            "progress": 100,
            "message": "전처리 완료",
            "audio_path": audio_path,
            "preprocessed_audio_path": preprocessed_audio_path,
            "vad_segments": vad_segments
        }
        await send_websocket_update(file_id, "completed", 100, "전처리 완료")
        
        logger.info(f"전처리 완료: {file_id}")
        
    except Exception as e:
        error_msg = f"전처리 실패: {str(e)}"
        logger.error(error_msg)
        processing_status[file_id] = {
            "status": "error",
            "progress": 0,
            "message": error_msg
        }
        await send_websocket_update(file_id, "error", 0, error_msg)


@app.get("/")
async def root():
    """API 루트 엔드포인트"""
    return {
        "message": "Subtitle Maker X API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/api/upload",
            "video": "/api/video/{file_id}",
            "transcribe": "/api/transcribe/{file_id}",
            "download": "/api/download/{file_id}/{format}",
            "websocket": "/ws/{file_id}"
        }
    }


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    영상 파일 업로드
    업로드 후 백그라운드에서 전처리 시작
    """
    try:
        # 파일 ID 생성
        file_id = str(uuid.uuid4())
        
        # 파일 확장자 확인
        file_ext = Path(file.filename).suffix.lower()
        allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"지원하지 않는 파일 형식입니다. 허용된 형식: {', '.join(allowed_extensions)}"
            )
        
        # 파일 저장
        video_path = UPLOAD_DIR / f"{file_id}{file_ext}"
        
        logger.info(f"파일 업로드 시작: {file.filename} -> {file_id}")
        
        with open(video_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 파일 정보 저장
        file_storage[file_id] = {
            "original_filename": file.filename,
            "file_path": str(video_path),
            "upload_time": datetime.now().isoformat(),
            "status": "uploaded"
        }
        
        # 출력 디렉토리 생성
        output_dir = OUTPUT_DIR / file_id
        output_dir.mkdir(exist_ok=True)
        
        # 백그라운드에서 전처리 시작
        asyncio.create_task(preprocess_video_background(
            file_id,
            str(video_path),
            str(output_dir)
        ))
        
        logger.info(f"파일 업로드 완료: {file_id}")
        
        return JSONResponse(content={
            "fileId": file_id,
            "filename": file.filename,
            "message": "업로드 완료. 전처리가 시작되었습니다."
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"업로드 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"업로드 실패: {str(e)}")


@app.get("/api/video/{file_id}")
async def get_video(file_id: str):
    """업로드된 영상 파일 제공"""
    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    
    video_path = file_storage[file_id]["file_path"]
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="파일이 존재하지 않습니다.")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=file_storage[file_id]["original_filename"]
    )


@app.websocket("/ws/{file_id}")
async def websocket_endpoint(websocket: WebSocket, file_id: str):
    """전처리 상태 업데이트를 위한 WebSocket 연결"""
    await websocket.accept()
    websocket_connections[file_id] = websocket
    
    logger.info(f"WebSocket 연결: {file_id}")
    
    try:
        # 현재 상태 전송
        if file_id in processing_status:
            status = processing_status[file_id]
            await websocket.send_json({
                "status": status["status"],
                "progress": status["progress"],
                "message": status["message"]
            })
        else:
            await websocket.send_json({
                "status": "queued",
                "progress": 0,
                "message": "전처리 대기 중"
            })
        
        # 연결 유지
        while True:
            # 클라이언트로부터 메시지 대기 (연결 유지용)
            data = await websocket.receive_text()
            logger.debug(f"WebSocket 메시지 수신: {file_id} - {data}")
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket 연결 종료: {file_id}")
    except Exception as e:
        logger.error(f"WebSocket 오류: {str(e)}")
    finally:
        if file_id in websocket_connections:
            del websocket_connections[file_id]


@app.post("/api/transcribe/{file_id}")
async def transcribe_video(file_id: str):
    """
    영상에서 자막 생성
    전처리가 완료된 후 호출되어야 함
    """
    try:
        # 파일 존재 확인
        if file_id not in file_storage:
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
        
        # 전처리 상태 확인
        if file_id not in processing_status:
            raise HTTPException(
                status_code=400,
                detail="전처리가 아직 시작되지 않았습니다."
            )
        
        status = processing_status[file_id]
        
        if status["status"] == "processing":
            raise HTTPException(
                status_code=400,
                detail="전처리가 진행 중입니다. 완료될 때까지 기다려주세요."
            )
        
        if status["status"] == "error":
            raise HTTPException(
                status_code=500,
                detail=f"전처리 중 오류가 발생했습니다: {status['message']}"
            )
        
        logger.info(f"자막 생성 시작: {file_id}")
        
        # SubtitleProcessor 가져오기
        proc = get_processor()
        
        # WhisperX 모델 로드
        model = proc._load_whisper_model()
        
        # 전처리된 오디오 로드
        import whisperx
        preprocessed_audio_path = status["preprocessed_audio_path"]
        audio = whisperx.load_audio(preprocessed_audio_path)
        
        # 음성 인식
        logger.info(f"음성 인식 시작: {file_id}")
        result = model.transcribe(audio, batch_size=16)
        
        detected_language = result.get("language", "unknown")
        logger.info(f"음성 인식 완료: {file_id} (언어: {detected_language})")
        
        # 타임스탬프 정렬
        logger.info(f"타임스탬프 정렬 시작: {file_id}")
        proc._load_align_model(detected_language)
        
        result = whisperx.align(
            result["segments"],
            proc.align_model,
            proc.align_metadata,
            audio,
            proc.device,
            return_char_alignments=False
        )
        
        logger.info(f"타임스탬프 정렬 완료: {file_id}")
        
        # 화자 분리 (토큰이 있는 경우에만)
        if proc.hf_token:
            try:
                logger.info(f"화자 분리 시작: {file_id}")
                diarize_model = proc._load_diarize_model()
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                logger.info(f"화자 분리 완료: {file_id}")
            except Exception as e:
                logger.warning(f"화자 분리 실패 (계속 진행): {str(e)}")
        
        # 결과 저장
        segments = result.get("segments", [])
        
        # 자막 파일 생성
        output_dir = OUTPUT_DIR / file_id
        
        for fmt in ["srt", "vtt", "txt", "json"]:
            output_path = output_dir / f"subtitle.{fmt}"
            proc.export_subtitles(segments, str(output_path), format=fmt)
        
        # 파일 정보 업데이트
        file_storage[file_id]["segments"] = segments
        file_storage[file_id]["language"] = detected_language
        file_storage[file_id]["transcribed"] = True
        
        logger.info(f"자막 생성 완료: {file_id} ({len(segments)} 세그먼트)")
        
        return JSONResponse(content={
            "segments": segments,
            "language": detected_language,
            "message": "자막 생성이 완료되었습니다."
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"자막 생성 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"자막 생성 실패: {str(e)}")


@app.get("/api/download/{file_id}/{format}")
async def download_subtitle(file_id: str, format: str):
    """
    자막 파일 다운로드
    지원 형식: srt, vtt, txt, json
    """
    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    
    if not file_storage[file_id].get("transcribed", False):
        raise HTTPException(status_code=400, detail="아직 자막이 생성되지 않았습니다.")
    
    # 형식 검증
    allowed_formats = ["srt", "vtt", "txt", "json"]
    if format not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 형식입니다. 허용된 형식: {', '.join(allowed_formats)}"
        )
    
    # 파일 경로
    subtitle_path = OUTPUT_DIR / file_id / f"subtitle.{format}"
    
    if not subtitle_path.exists():
        raise HTTPException(status_code=404, detail="자막 파일을 찾을 수 없습니다.")
    
    # MIME 타입 설정
    media_types = {
        "srt": "application/x-subrip",
        "vtt": "text/vtt",
        "txt": "text/plain",
        "json": "application/json"
    }
    
    original_filename = Path(file_storage[file_id]["original_filename"]).stem
    
    return FileResponse(
        subtitle_path,
        media_type=media_types[format],
        filename=f"{original_filename}.{format}"
    )


@app.get("/api/status/{file_id}")
async def get_status(file_id: str):
    """파일의 처리 상태 조회"""
    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    
    status = processing_status.get(file_id, {
        "status": "not_started",
        "progress": 0,
        "message": "대기 중"
    })
    
    return JSONResponse(content={
        "fileId": file_id,
        "status": status["status"],
        "progress": status["progress"],
        "message": status["message"],
        "transcribed": file_storage[file_id].get("transcribed", False)
    })


@app.delete("/api/file/{file_id}")
async def delete_file(file_id: str):
    """업로드된 파일 및 관련 데이터 삭제"""
    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    
    try:
        # 파일 삭제
        video_path = file_storage[file_id]["file_path"]
        if os.path.exists(video_path):
            os.remove(video_path)
        
        # 출력 디렉토리 삭제
        output_dir = OUTPUT_DIR / file_id
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
        
        # 메모리에서 삭제
        del file_storage[file_id]
        
        if file_id in processing_status:
            del processing_status[file_id]
        
        if file_id in websocket_connections:
            await websocket_connections[file_id].close()
            del websocket_connections[file_id]
        
        logger.info(f"파일 삭제 완료: {file_id}")
        
        return JSONResponse(content={"message": "파일이 삭제되었습니다."})
        
    except Exception as e:
        logger.error(f"파일 삭제 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"파일 삭제 실패: {str(e)}")


if __name__ == "__main__":
    # 서버 실행
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 개발 모드: 코드 변경 시 자동 재시작
        log_level="info"
    )

