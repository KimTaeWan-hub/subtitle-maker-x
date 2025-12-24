"""
비디오에서 오디오 추출 모듈
"""

import os
import logging
from pathlib import Path
import ffmpeg

# 로깅 설정 # 프로그램 실행 중 발생하는 이벤트와 오류를 기록하고 추적하기 위한 것
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """비디오에서 오디오를 추출하는 클래스"""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Args:
            sample_rate: 오디오 샘플 레이트 (기본값: 16000 Hz)
        """
        self.sample_rate = sample_rate
        logger.info(f"AudioProcessor 초기화 완료 (sample_rate: {sample_rate})")
    
    def extract_audio(self, video_path: str, output_path: str) -> str:
        """
        비디오 파일에서 오디오를 추출하여 WAV 파일로 저장
        
        Args:
            video_path: 입력 비디오 파일 경로
            output_path: 출력 오디오 파일 경로
            
        Returns:
            str: 생성된 오디오 파일 경로
            
        Raises:
            Exception: FFmpeg 실행 중 오류 발생 시
        """
        try:
            logger.info(f"오디오 추출 시작: {video_path}")
            
            # 출력 디렉토리가 없으면 생성
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"출력 디렉토리 생성: {output_dir}")
            
            # ffmpeg를 사용하여 오디오 추출
            # - acodec='pcm_s16le': 16비트 PCM 오디오 코덱
            # - ac=1: 모노 채널 (Whisper는 모노 오디오를 권장)
            # - ar=sample_rate: 샘플레이트 설정
            # - filter: loudnorm (EBU R128 표준 기반 볼륨 정규화) # Gemini가 추가
            stream = ffmpeg.input(video_path)
            
            # 오디오 필터 체인 구성: 볼륨 정규화 -> 포맷 변환 # Gemini가 추가
            stream = ffmpeg.filter(stream, 'loudnorm', I=-16, TP=-1.5, LRA=11)
            
            stream = ffmpeg.output(
                stream,
                output_path,
                acodec='pcm_s16le',  # 16-bit PCM
                ac=1,                # 모노
                ar=self.sample_rate  # 샘플 레이트
            )
            
            # 기존 파일이 있으면 덮어쓰기
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            
            logger.info(f"오디오 추출 완료: {output_path}")
            return output_path
            
        except ffmpeg.Error as e:
            error_message = e.stderr.decode('utf-8') if e.stderr else str(e)
            logger.error(f"FFmpeg 오류 발생: {error_message}")
            raise Exception(f"오디오 추출 실패: {error_message}")
        
        except Exception as e:
            logger.error(f"오디오 추출 중 예외 발생: {str(e)}")
            raise
    
    def get_audio_info(self, file_path: str) -> dict:
        """
        오디오 또는 비디오 파일의 정보를 가져옴
        
        Args:
            file_path: 파일 경로
            
        Returns:
            dict: 파일 정보 (duration, sample_rate, channels 등)
        """
        try:
            logger.info(f"파일 정보 조회: {file_path}")
            probe = ffmpeg.probe(file_path)
            
            # 오디오 스트림 찾기
            audio_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'audio'),
                None
            )
            
            if audio_stream is None:
                logger.warning(f"오디오 스트림을 찾을 수 없음: {file_path}")
                return {}
            
            info = {
                'duration': float(probe['format'].get('duration', 0)),
                'sample_rate': int(audio_stream.get('sample_rate', 0)),
                'channels': int(audio_stream.get('channels', 0)),
                'codec': audio_stream.get('codec_name', 'unknown'),
                'bit_rate': int(audio_stream.get('bit_rate', 0))
            }
            
            logger.info(f"파일 정보: {info}")
            return info
            
        except Exception as e:
            logger.error(f"파일 정보 조회 실패: {str(e)}")
            return {}


# 테스트용 코드
if __name__ == "__main__":
    processor = AudioProcessor()
    
    # 테스트 예제
    test_video = "test_video.mp4"
    test_output = "test_audio.wav"
    
    if os.path.exists(test_video):
        try:
            processor.extract_audio(test_video, test_output)
            print(f"✓ 오디오 추출 성공: {test_output}")
            
            # 추출된 오디오 정보 확인
            info = processor.get_audio_info(test_output)
            print(f"✓ 오디오 정보: {info}")
        except Exception as e:
            print(f"✗ 오류 발생: {e}")
    else:
        print(f"테스트 파일이 없습니다: {test_video}")
