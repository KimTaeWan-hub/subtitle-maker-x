"""
자막 생성 통합 처리 모듈
프론트엔드에서 받은 영상을 처리하여 자막을 생성하는 전체 파이프라인
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import whisperx
import gc
import torch

from audio_extractor import AudioProcessor
from audio_detector import VoiceActivityDetector

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SubtitleProcessor:
    """영상에서 자막을 생성하는 통합 처리 클래스"""
    
    def __init__(
        self,
        whisper_model: str = "large-v2",
        device: str = "cuda",
        compute_type: str = "float16",
        sample_rate: int = 16000,
        vad_threshold: float = 0.5,
        hf_token: Optional[str] = None
    ):
        """
        Args:
            whisper_model: WhisperX 모델 이름 (기본값: "large-v2")
            device: 디바이스 ("cuda" 또는 "cpu", 기본값: "cuda")
            compute_type: 계산 타입 ("float16", "int8" 등, 기본값: "float16")
            sample_rate: 오디오 샘플 레이트 (기본값: 16000 Hz)
            vad_threshold: 음성 탐지 임계값 (0.0 ~ 1.0, 기본값: 0.5)
            hf_token: Hugging Face 토큰 (화자 분리에 필요)
        """
        self.whisper_model_name = whisper_model
        self.device = device
        self.compute_type = compute_type
        self.sample_rate = sample_rate
        self.hf_token = hf_token
        
        # 오디오 추출기 초기화
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        
        # 음성 구간 탐지기 초기화
        self.vad = VoiceActivityDetector(
            sample_rate=sample_rate,
            threshold=vad_threshold
        )
        
        # WhisperX 모델은 지연 로딩
        self.whisper_model = None
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None
        
        logger.info(
            f"SubtitleProcessor 초기화 완료 "
            f"(model: {whisper_model}, device: {device}, "
            f"compute_type: {compute_type}, sample_rate: {sample_rate})"
        )
    
    def _load_whisper_model(self):
        """WhisperX 모델을 로드합니다 (지연 로딩)"""
        if self.whisper_model is None:
            try:
                logger.info(f"WhisperX 모델 로딩 중: {self.whisper_model_name}")
                self.whisper_model = whisperx.load_model(
                    self.whisper_model_name,
                    self.device,
                    compute_type=self.compute_type
                )
                logger.info("WhisperX 모델 로드 완료")
            except Exception as e:
                logger.error(f"WhisperX 모델 로드 실패: {str(e)}")
                raise
        return self.whisper_model
    
    def _load_align_model(self, language_code: str):
        """정렬 모델을 로드합니다"""
        try:
            logger.info(f"정렬 모델 로딩 중 (언어: {language_code})")
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code=language_code,
                device=self.device
            )
            logger.info("정렬 모델 로드 완료")
        except Exception as e:
            logger.error(f"정렬 모델 로드 실패: {str(e)}")
            raise
    
    def _load_diarize_model(self):
        """화자 분리 모델을 로드합니다"""
        if self.diarize_model is None:
            if not self.hf_token:
                raise ValueError(
                    "화자 분리를 위해서는 Hugging Face 토큰이 필요합니다. "
                    "hf_token 파라미터를 설정해주세요."
                )
            
            try:
                logger.info("화자 분리 모델 로딩 중...")
                from whisperx.diarize import DiarizationPipeline
                self.diarize_model = DiarizationPipeline(
                    use_auth_token=self.hf_token,
                    device=self.device
                )
                logger.info("화자 분리 모델 로드 완료")
            except Exception as e:
                logger.error(f"화자 분리 모델 로드 실패: {str(e)}")
                raise
        return self.diarize_model
    
    def process_video(
        self,
        video_path: str,
        output_dir: str,
        language: Optional[str] = None,
        batch_size: int = 16,
        enable_diarization: bool = True,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100
    ) -> Dict:
        """
        영상 파일을 처리하여 자막을 생성합니다.
        
        전체 파이프라인:
        1. 영상에서 오디오 추출 (audio_extractor.py)
        2. 음성 구간 탐지 및 전처리 (audio_detector.py)
        3. WhisperX로 음성을 텍스트로 변환
        4. 단어 수준 타임스탬프 정렬
        5. 화자 분리 (선택적)
        
        Args:
            video_path: 입력 영상 파일 경로
            output_dir: 출력 디렉토리
            language: 언어 코드 (None이면 자동 감지, 예: "ko", "en", "ja")
            batch_size: WhisperX 배치 크기 (기본값: 16)
            enable_diarization: 화자 분리 활성화 여부 (기본값: True)
            min_speakers: 최소 화자 수 (None이면 자동)
            max_speakers: 최대 화자 수 (None이면 자동)
            min_speech_duration_ms: 최소 음성 구간 길이 (ms)
            min_silence_duration_ms: 음성 구간을 분리하는 최소 무음 길이 (ms)
            
        Returns:
            Dict: 처리 결과
                - segments: 자막 세그먼트 리스트 (타임스탬프, 텍스트, 화자 포함)
                - language: 감지된 언어
                - audio_path: 추출된 오디오 파일 경로
                - preprocessed_audio_path: 전처리된 오디오 파일 경로
                - vad_segments: 음성 구간 탐지 결과
                - statistics: 처리 통계 정보
        """
        try:
            logger.info(f"영상 처리 시작: {video_path}")
            
            # 출력 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)
            
            # 파일명 추출
            video_name = Path(video_path).stem
            
            # ========== 1단계: 오디오 추출 ==========
            logger.info("=" * 60)
            logger.info("1단계: 영상에서 오디오 추출")
            logger.info("=" * 60)
            
            audio_path = os.path.join(output_dir, f"{video_name}_audio.wav")
            self.audio_processor.extract_audio(video_path, audio_path)
            
            # ========== 2단계: 음성 구간 탐지 및 전처리 ==========
            logger.info("=" * 60)
            logger.info("2단계: 음성 구간 탐지 및 전처리")
            logger.info("=" * 60)
            
            # 음성 구간 탐지
            vad_segments = self.vad.detect_voice_segments(
                audio_path,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms
            )
            
            # 음성 통계 계산
            vad_statistics = self.vad.get_speech_statistics(audio_path, vad_segments)
            
            # 음성이 아닌 구간 음소거 처리 (전처리)
            preprocessed_audio_path = os.path.join(
                output_dir,
                f"{video_name}_preprocessed.wav"
            )
            self.vad.extract_speech_only_audio(
                audio_path,
                preprocessed_audio_path,
                vad_segments
            )
            
            # ========== 3단계: WhisperX 음성 인식 ==========
            logger.info("=" * 60)
            logger.info("3단계: WhisperX 음성 인식")
            logger.info("=" * 60)
            
            # WhisperX 모델 로드
            model = self._load_whisper_model()
            
            # 전처리된 오디오 로드
            audio = whisperx.load_audio(preprocessed_audio_path)
            
            # 음성 인식 수행
            logger.info("음성 인식 시작...")
            transcribe_options = {"batch_size": batch_size}
            if language:
                transcribe_options["language"] = language
            
            result = model.transcribe(audio, **transcribe_options)
            
            detected_language = result.get("language", "unknown")
            logger.info(f"음성 인식 완료 (감지된 언어: {detected_language})")
            logger.info(f"세그먼트 수: {len(result['segments'])}")
            
            # ========== 4단계: 단어 수준 타임스탬프 정렬 ==========
            logger.info("=" * 60)
            logger.info("4단계: 단어 수준 타임스탬프 정렬")
            logger.info("=" * 60)
            
            # 정렬 모델 로드
            self._load_align_model(detected_language)
            
            # 정렬 수행
            logger.info("타임스탬프 정렬 시작...")
            result = whisperx.align(
                result["segments"],
                self.align_model,
                self.align_metadata,
                audio,
                self.device,
                return_char_alignments=False
            )
            logger.info("타임스탬프 정렬 완료")
            
            # ========== 5단계: 화자 분리 (선택적) ==========
            if enable_diarization:
                logger.info("=" * 60)
                logger.info("5단계: 화자 분리")
                logger.info("=" * 60)
                
                try:
                    # 화자 분리 모델 로드
                    diarize_model = self._load_diarize_model()
                    
                    # 화자 분리 수행
                    logger.info("화자 분리 시작...")
                    diarize_options = {}
                    if min_speakers is not None:
                        diarize_options["min_speakers"] = min_speakers
                    if max_speakers is not None:
                        diarize_options["max_speakers"] = max_speakers
                    
                    diarize_segments = diarize_model(audio, **diarize_options)
                    
                    # 화자 레이블 할당
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    logger.info("화자 분리 완료")
                    
                    # 화자 수 로깅
                    speakers = set()
                    for segment in result.get("segments", []):
                        if "speaker" in segment:
                            speakers.add(segment["speaker"])
                    logger.info(f"감지된 화자 수: {len(speakers)}")
                    
                except Exception as e:
                    logger.warning(f"화자 분리 실패 (계속 진행): {str(e)}")
            
            # ========== 결과 정리 ==========
            logger.info("=" * 60)
            logger.info("처리 완료")
            logger.info("=" * 60)
            
            # 통계 정보
            statistics = {
                "total_segments": len(result.get("segments", [])),
                "vad_statistics": vad_statistics,
                "detected_language": detected_language,
            }
            
            if enable_diarization:
                speakers = set()
                for segment in result.get("segments", []):
                    if "speaker" in segment:
                        speakers.add(segment["speaker"])
                statistics["num_speakers"] = len(speakers)
                statistics["speakers"] = sorted(list(speakers))
            
            # 결과 반환
            final_result = {
                "segments": result.get("segments", []),
                "language": detected_language,
                "audio_path": audio_path,
                "preprocessed_audio_path": preprocessed_audio_path,
                "vad_segments": vad_segments,
                "statistics": statistics
            }
            
            logger.info(f"최종 세그먼트 수: {statistics['total_segments']}")
            logger.info(f"감지된 언어: {detected_language}")
            if enable_diarization and "num_speakers" in statistics:
                logger.info(f"화자 수: {statistics['num_speakers']}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"영상 처리 중 오류 발생: {str(e)}")
            raise
    
    def export_subtitles(
        self,
        segments: List[Dict],
        output_path: str,
        format: str = "srt"
    ) -> str:
        """
        세그먼트를 자막 파일로 내보냅니다.
        
        Args:
            segments: 자막 세그먼트 리스트
            output_path: 출력 파일 경로
            format: 자막 포맷 ("srt", "vtt", "txt", "json")
            
        Returns:
            str: 생성된 파일 경로
        """
        try:
            logger.info(f"자막 파일 생성 중: {output_path} (형식: {format})")
            
            if format == "srt":
                self._export_srt(segments, output_path)
            elif format == "vtt":
                self._export_vtt(segments, output_path)
            elif format == "txt":
                self._export_txt(segments, output_path)
            elif format == "json":
                self._export_json(segments, output_path)
            else:
                raise ValueError(f"지원하지 않는 형식: {format}")
            
            logger.info(f"자막 파일 생성 완료: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"자막 파일 생성 실패: {str(e)}")
            raise
    
    def _export_srt(self, segments: List[Dict], output_path: str):
        """SRT 형식으로 내보내기"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start = self._format_timestamp_srt(segment.get('start', 0))
                end = self._format_timestamp_srt(segment.get('end', 0))
                text = segment.get('text', '').strip()
                
                # 화자 정보 추가 (있는 경우)
                if 'speaker' in segment:
                    text = f"[{segment['speaker']}] {text}"
                
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
    
    def _export_vtt(self, segments: List[Dict], output_path: str):
        """VTT 형식으로 내보내기"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            
            for segment in segments:
                start = self._format_timestamp_vtt(segment.get('start', 0))
                end = self._format_timestamp_vtt(segment.get('end', 0))
                text = segment.get('text', '').strip()
                
                # 화자 정보 추가 (있는 경우)
                if 'speaker' in segment:
                    text = f"[{segment['speaker']}] {text}"
                
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
    
    def _export_txt(self, segments: List[Dict], output_path: str):
        """TXT 형식으로 내보내기"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for segment in segments:
                text = segment.get('text', '').strip()
                
                # 화자 정보 추가 (있는 경우)
                if 'speaker' in segment:
                    f.write(f"[{segment['speaker']}] {text}\n")
                else:
                    f.write(f"{text}\n")
    
    def _export_json(self, segments: List[Dict], output_path: str):
        """JSON 형식으로 내보내기"""
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
    
    def _format_timestamp_srt(self, seconds: float) -> str:
        """SRT 타임스탬프 형식으로 변환 (00:00:00,000)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _format_timestamp_vtt(self, seconds: float) -> str:
        """VTT 타임스탬프 형식으로 변환 (00:00:00.000)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    
    def cleanup(self):
        """메모리 정리"""
        logger.info("메모리 정리 중...")
        
        if self.whisper_model is not None:
            del self.whisper_model
            self.whisper_model = None
        
        if self.align_model is not None:
            del self.align_model
            self.align_model = None
        
        if self.diarize_model is not None:
            del self.diarize_model
            self.diarize_model = None
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("메모리 정리 완료")


# 테스트용 코드
if __name__ == "__main__":
    import sys
    
    # 사용 예시
    print("=" * 60)
    print("SubtitleProcessor 사용 예시")
    print("=" * 60)
    
    # 환경 변수에서 Hugging Face 토큰 가져오기
    hf_token = os.environ.get("HF_TOKEN")
    
    if not hf_token:
        print("\n⚠️  경고: HF_TOKEN 환경 변수가 설정되지 않았습니다.")
        print("화자 분리를 사용하려면 Hugging Face 토큰이 필요합니다.")
        print("export HF_TOKEN='your_token_here' 명령어로 설정하세요.\n")
    
    # 테스트 파일 경로
    test_video = "test_video.mp4"
    
    if len(sys.argv) > 1:
        test_video = sys.argv[1]
    
    if not os.path.exists(test_video):
        print(f"\n❌ 테스트 파일이 없습니다: {test_video}")
        print("\n사용법:")
        print(f"  python {__file__} <video_path>")
        print("\n예시:")
        print(f"  python {__file__} sample.mp4")
        sys.exit(1)
    
    try:
        # 프로세서 초기화
        processor = SubtitleProcessor(
            whisper_model="base",  # 테스트용으로 작은 모델 사용
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "int8",
            hf_token=hf_token
        )
        
        # 영상 처리
        output_dir = "output"
        result = processor.process_video(
            video_path=test_video,
            output_dir=output_dir,
            enable_diarization=bool(hf_token),  # 토큰이 있을 때만 화자 분리
            batch_size=8  # 테스트용으로 작은 배치 크기
        )
        
        print("\n" + "=" * 60)
        print("처리 결과")
        print("=" * 60)
        print(f"✓ 세그먼트 수: {result['statistics']['total_segments']}")
        print(f"✓ 감지된 언어: {result['statistics']['detected_language']}")
        print(f"✓ 음성 구간 비율: {result['statistics']['vad_statistics']['speech_ratio'] * 100:.1f}%")
        
        if 'num_speakers' in result['statistics']:
            print(f"✓ 화자 수: {result['statistics']['num_speakers']}")
        
        # 자막 파일 내보내기
        video_name = Path(test_video).stem
        
        srt_path = os.path.join(output_dir, f"{video_name}.srt")
        processor.export_subtitles(result['segments'], srt_path, format="srt")
        print(f"\n✓ SRT 파일 생성: {srt_path}")
        
        vtt_path = os.path.join(output_dir, f"{video_name}.vtt")
        processor.export_subtitles(result['segments'], vtt_path, format="vtt")
        print(f"✓ VTT 파일 생성: {vtt_path}")
        
        txt_path = os.path.join(output_dir, f"{video_name}.txt")
        processor.export_subtitles(result['segments'], txt_path, format="txt")
        print(f"✓ TXT 파일 생성: {txt_path}")
        
        json_path = os.path.join(output_dir, f"{video_name}.json")
        processor.export_subtitles(result['segments'], json_path, format="json")
        print(f"✓ JSON 파일 생성: {json_path}")
        
        # 메모리 정리
        processor.cleanup()
        
        print("\n✅ 모든 처리가 완료되었습니다!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

