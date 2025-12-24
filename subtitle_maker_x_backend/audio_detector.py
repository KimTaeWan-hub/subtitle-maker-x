"""
Silero VAD를 사용한 음성 구간 탐지 모듈
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
import torch

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """Silero VAD를 사용하여 오디오에서 음성 구간을 탐지하는 클래스"""
    
    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        """
        Args:
            sample_rate: 오디오 샘플 레이트 (기본값: 16000 Hz)
            threshold: 음성 탐지 임계값 (0.0 ~ 1.0, 기본값: 0.5)
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.model = None
        logger.info(f"VoiceActivityDetector 초기화 (sample_rate: {sample_rate}, threshold: {threshold})")
    
    def _load_model(self):
        """Silero VAD 모델을 로드합니다 (지연 로딩)"""
        if self.model is None:
            try:
                from silero_vad import load_silero_vad
                logger.info("Silero VAD 모델 로딩 중...")
                self.model = load_silero_vad()
                logger.info("Silero VAD 모델 로드 완료")
            except ImportError:
                raise ImportError(
                    "silero-vad 패키지가 설치되어 있지 않습니다. "
                    "'pip install silero-vad' 명령어로 설치해주세요."
                )
            except Exception as e:
                logger.error(f"모델 로드 실패: {str(e)}")
                raise
        return self.model
    
    def detect_voice_segments(
        self,
        audio_path: str,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float = float('inf'),
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        return_seconds: bool = True
    ) -> List[Dict[str, float]]:
        """
        오디오 파일에서 음성이 있는 구간을 탐지합니다.
        
        Args:
            audio_path: 오디오 파일 경로 (WAV 형식 권장)
            min_speech_duration_ms: 최소 음성 구간 길이 (ms, 기본값: 250ms)
            max_speech_duration_s: 최대 음성 구간 길이 (초, 기본값: 무제한)
            min_silence_duration_ms: 음성 구간을 분리하는 최소 무음 길이 (ms, 기본값: 100ms)
            speech_pad_ms: 음성 구간 앞뒤 패딩 (ms, 기본값: 30ms)
            return_seconds: True면 초 단위, False면 샘플 단위로 반환
            
        Returns:
            List[Dict[str, float]]: 음성 구간 리스트
                예: [{'start': 0.512, 'end': 3.264}, {'start': 5.120, 'end': 8.736}]
                
        Raises:
            FileNotFoundError: 오디오 파일이 존재하지 않을 때
            Exception: VAD 처리 중 오류 발생 시
        """
        try:
            # 파일 존재 확인
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")
            
            logger.info(f"음성 구간 탐지 시작: {audio_path}")
            
            # 모델 로드
            model = self._load_model()
            
            # 오디오 파일 읽기
            from silero_vad import read_audio, get_speech_timestamps
            wav = read_audio(audio_path, sampling_rate=self.sample_rate)
            
            logger.info(f"오디오 길이: {len(wav) / self.sample_rate:.2f}초")
            
            # 음성 타임스탬프 추출
            speech_timestamps = get_speech_timestamps(
                wav,
                model,
                threshold=self.threshold,
                sampling_rate=self.sample_rate,
                min_speech_duration_ms=min_speech_duration_ms,
                max_speech_duration_s=max_speech_duration_s,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_pad_ms=speech_pad_ms,
                return_seconds=return_seconds
            )
            
            logger.info(f"음성 구간 탐지 완료: {len(speech_timestamps)}개 구간 발견")
            
            # 탐지된 구간 로깅
            for i, segment in enumerate(speech_timestamps, 1):
                duration = segment['end'] - segment['start']
                logger.info(f"  구간 {i}: {segment['start']:.2f}s ~ {segment['end']:.2f}s (길이: {duration:.2f}s)")
            
            # 모델 상태 초기화
            model.reset_states()
            
            return speech_timestamps
            
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"음성 구간 탐지 실패: {str(e)}")
            raise Exception(f"음성 구간 탐지 중 오류 발생: {str(e)}")
    
    def get_speech_statistics(
        self,
        audio_path: str,
        speech_timestamps: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, float]:
        """
        오디오 파일의 음성 구간 통계를 계산합니다.
        
        Args:
            audio_path: 오디오 파일 경로
            speech_timestamps: 이미 계산된 음성 타임스탬프 (없으면 자동으로 계산)
            
        Returns:
            Dict[str, float]: 통계 정보
                - total_duration: 전체 오디오 길이 (초)
                - speech_duration: 음성 구간 총 길이 (초)
                - silence_duration: 무음 구간 총 길이 (초)
                - speech_ratio: 음성 비율 (0.0 ~ 1.0)
                - num_segments: 음성 구간 개수
        """
        try:
            # 음성 타임스탬프가 없으면 계산
            if speech_timestamps is None:
                speech_timestamps = self.detect_voice_segments(audio_path)
            
            # 오디오 파일 읽기
            from silero_vad import read_audio
            wav = read_audio(audio_path, sampling_rate=self.sample_rate)
            
            total_duration = len(wav) / self.sample_rate
            
            # 음성 구간 총 길이 계산
            speech_duration = sum(
                segment['end'] - segment['start']
                for segment in speech_timestamps
            )
            
            silence_duration = total_duration - speech_duration
            speech_ratio = speech_duration / total_duration if total_duration > 0 else 0.0
            
            statistics = {
                'total_duration': round(total_duration, 2),
                'speech_duration': round(speech_duration, 2),
                'silence_duration': round(silence_duration, 2),
                'speech_ratio': round(speech_ratio, 4),
                'num_segments': len(speech_timestamps)
            }
            
            logger.info(f"음성 통계: {statistics}")
            
            return statistics
            
        except Exception as e:
            logger.error(f"통계 계산 실패: {str(e)}")
            raise
    
    def extract_speech_only_audio(
        self,
        audio_path: str,
        output_path: str,
        speech_timestamps: Optional[List[Dict[str, float]]] = None
    ) -> str:
        """
        오디오에서 음성이 아닌 구간을 음소거 처리하여 저장합니다 (길이 유지).
        
        Args:
            audio_path: 입력 오디오 파일 경로
            output_path: 출력 오디오 파일 경로
            speech_timestamps: 이미 계산된 음성 타임스탬프 (없으면 자동으로 계산)
            
        Returns:
            str: 생성된 오디오 파일 경로
        """
        try:
            logger.info(f"음성 구간 음소거 처리 시작: {audio_path} -> {output_path}")
            
            # 음성 타임스탬프가 없으면 계산
            if speech_timestamps is None:
                speech_timestamps = self.detect_voice_segments(audio_path)
            
            # 오디오 파일 읽기
            from silero_vad import read_audio, save_audio
            import numpy as np
            wav = read_audio(audio_path, sampling_rate=self.sample_rate)
            
            # 원본 오디오 복사 (numpy array로 변환)
            processed_wav = wav.clone()
            
            # 음성이 아닌 구간을 0으로 만들기 (음소거)
            # 먼저 전체를 0으로 만들고, 음성 구간만 원본 값으로 채우기
            processed_wav[:] = 0.0
            
            for segment in speech_timestamps:
                start_sample = int(segment['start'] * self.sample_rate)
                end_sample = int(segment['end'] * self.sample_rate)
                
                # 범위 체크
                start_sample = max(0, start_sample)
                end_sample = min(len(wav), end_sample)
                
                # 음성 구간은 원본 오디오 유지
                processed_wav[start_sample:end_sample] = wav[start_sample:end_sample]
            
            # 출력 디렉토리가 없으면 생성
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 파일 저장
            save_audio(output_path, processed_wav, sampling_rate=self.sample_rate)
            
            # 통계 계산
            original_duration = len(wav) / self.sample_rate
            speech_duration = sum(segment['end'] - segment['start'] for segment in speech_timestamps)
            muted_duration = original_duration - speech_duration
            muted_ratio = 100 * (muted_duration / original_duration) if original_duration > 0 else 0
            
            logger.info(f"음소거 처리 완료: {output_path}")
            logger.info(f"전체 길이: {original_duration:.1f}초 (유지)")
            logger.info(f"음성 구간: {speech_duration:.1f}초, 음소거 구간: {muted_duration:.1f}초")
            logger.info(f"음소거 비율: {muted_ratio:.1f}%")
            
            return output_path
            
        except Exception as e:
            logger.error(f"음소거 처리 실패: {str(e)}")
            raise


# 테스트용 코드
if __name__ == "__main__":
    detector = VoiceActivityDetector(sample_rate=16000, threshold=0.5)
    
    # 테스트 예제
    test_audio = "test_audio.wav"
    
    if os.path.exists(test_audio):
        try:
            # 음성 구간 탐지
            segments = detector.detect_voice_segments(test_audio)
            print(f"\n✓ 음성 구간 탐지 완료: {len(segments)}개 구간")
            
            for i, seg in enumerate(segments, 1):
                print(f"  구간 {i}: {seg['start']:.2f}s ~ {seg['end']:.2f}s")
            
            # 통계 계산
            stats = detector.get_speech_statistics(test_audio, segments)
            print(f"\n✓ 음성 통계:")
            print(f"  전체 길이: {stats['total_duration']}초")
            print(f"  음성 길이: {stats['speech_duration']}초")
            print(f"  무음 길이: {stats['silence_duration']}초")
            print(f"  음성 비율: {stats['speech_ratio'] * 100:.1f}%")
            print(f"  음성 구간 수: {stats['num_segments']}개")
            
            # 음성만 추출 (옵션)
            # output_audio = "test_audio_speech_only.wav"
            # detector.extract_speech_only_audio(test_audio, output_audio, segments)
            # print(f"\n✓ 음성만 추출 완료: {output_audio}")
            
        except Exception as e:
            print(f"✗ 오류 발생: {e}")
    else:
        print(f"테스트 파일이 없습니다: {test_audio}")
        print("\n사용 예시:")
        print("  detector = VoiceActivityDetector(sample_rate=16000, threshold=0.5)")
        print("  segments = detector.detect_voice_segments('audio.wav')")
        print("  stats = detector.get_speech_statistics('audio.wav', segments)")
        print("  detector.extract_speech_only_audio('audio.wav', 'output.wav', segments)")

