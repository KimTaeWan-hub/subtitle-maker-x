import React, { useState, useEffect, useRef } from 'react';
import './PreprocessingStatus.css';

/**
 * 전처리 진행상황 표시 컴포넌트
 * WebSocket을 통해 실시간으로 전처리 진행상황을 표시합니다.
 */
const PreprocessingStatus = ({ status, progress, message }) => {
  const [estimatedTime, setEstimatedTime] = useState(null);
  const startTimeRef = useRef(null);
  const lastProgressRef = useRef(0);
  const lastUpdateRef = useRef(null);

  // 진행 상태 추적 및 예상 시간 계산
  useEffect(() => {
    if (status === 'processing') {
      // 처음 시작할 때 시작 시간 기록
      if (!startTimeRef.current && progress > 0) {
        startTimeRef.current = Date.now();
        lastProgressRef.current = progress;
        lastUpdateRef.current = Date.now();
      }
      
      // 진행률이 업데이트되면 예상 시간 재계산
      if (progress > lastProgressRef.current && progress > 5) {
        const now = Date.now();
        const elapsed = (now - startTimeRef.current) / 1000; // 초 단위
        const progressMade = progress - 0; // 0%부터 현재까지의 진행
        
        if (progressMade > 0) {
          const rate = progressMade / elapsed; // % per second
          const remaining = (100 - progress) / rate; // 남은 초
          
          // 남은 시간이 너무 크거나 음수가 아닌 경우에만 업데이트
          if (remaining > 0 && remaining < 600) { // 최대 10분
            setEstimatedTime(Math.ceil(remaining));
          }
        }
        
        lastProgressRef.current = progress;
        lastUpdateRef.current = now;
      }
    } else {
      // 상태가 완료/실패/대기 중이면 초기화
      startTimeRef.current = null;
      lastProgressRef.current = 0;
      lastUpdateRef.current = null;
      setEstimatedTime(null);
    }
  }, [status, progress]);

  if (!status || status === 'not_started') {
    return null;
  }

  // 시간 포맷팅 함수
  const formatTime = (seconds) => {
    if (!seconds || seconds < 0) return null;
    if (seconds < 60) return `약 ${seconds}초`;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (secs === 0) return `약 ${mins}분`;
    return `약 ${mins}분 ${secs}초`;
  };

  const getStatusText = () => {
    switch (status) {
      case 'queued':
        return '전처리 대기 중...';
      case 'processing':
        return '전처리 진행 중...';
      case 'completed':
        return '전처리 완료!';
      case 'failed':
        return '전처리 실패';
      default:
        return '전처리 상태 확인 중...';
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'queued':
        return '#ffa500';
      case 'processing':
        return '#2196F3';
      case 'completed':
        return '#4CAF50';
      case 'failed':
        return '#f44336';
      default:
        return '#9e9e9e';
    }
  };

  return (
    <div className="preprocessing-status">
      <div className="status-header">
        <div 
          className={`status-indicator ${status}`}
          style={{ backgroundColor: getStatusColor() }}
        />
        <h3>{getStatusText()}</h3>
      </div>
      
      {status === 'processing' && (
        <div className="progress-container">
          <div className="progress-bar">
            <div 
              className="progress-fill"
              style={{ 
                width: `${progress}%`,
                backgroundColor: getStatusColor()
              }}
            />
          </div>
          <div className="progress-info">
            <div className="progress-text">{progress}%</div>
            {estimatedTime && progress > 5 && progress < 100 && (
              <div className="time-remaining">
                <svg className="clock-icon" viewBox="0 0 24 24" width="16" height="16">
                  <path 
                    fill="currentColor" 
                    d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10 10-4.5 10-10S17.5 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm.5-13H11v6l5.25 3.15.75-1.23-4.5-2.67z"
                  />
                </svg>
                <span>{formatTime(estimatedTime)} 남음</span>
              </div>
            )}
          </div>
        </div>
      )}
      
      {message && (
        <div className="status-message">{message}</div>
      )}
      
      {status === 'completed' && (
        <div className="completion-message">
          <svg className="check-icon" viewBox="0 0 24 24">
            <path 
              fill="currentColor" 
              d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"
            />
          </svg>
          <span>이제 자막을 생성할 수 있습니다.</span>
        </div>
      )}
      
      {status === 'failed' && (
        <div className="error-message">
          <svg className="error-icon" viewBox="0 0 24 24">
            <path 
              fill="currentColor" 
              d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"
            />
          </svg>
          <span>파일을 다시 업로드해주세요.</span>
        </div>
      )}
    </div>
  );
};

export default PreprocessingStatus;

