import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import VideoUploader from './components/VideoUploader';
import VideoPlayer from './components/VideoPlayer';
import SubtitleEditor from './components/SubtitleEditor';
import SubtitleDownloader from './components/SubtitleDownloader';
import PreprocessingStatus from './components/PreprocessingStatus';

function App() {
  const [fileId, setFileId] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [segments, setSegments] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const videoPlayerRef = useRef(null);
  
  // 전처리 상태 관리
  const [preprocessingStatus, setPreprocessingStatus] = useState({
    status: 'not_started',
    progress: 0,
    message: ''
  });
  
  const wsRef = useRef(null);

  // WebSocket 연결 설정
  useEffect(() => {
    if (!fileId) return;

    // WebSocket 연결
    const ws = new WebSocket(`ws://localhost:8000/ws/${fileId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket 연결 성공:', fileId);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('전처리 상태 업데이트:', data);
        setPreprocessingStatus({
          status: data.status,
          progress: data.progress,
          message: data.message
        });
      } catch (error) {
        console.error('WebSocket 메시지 파싱 오류:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket 오류:', error);
    };

    ws.onclose = () => {
      console.log('WebSocket 연결 종료:', fileId);
    };

    // 컴포넌트 언마운트 시 WebSocket 종료
    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [fileId]);

  const handleUploadSuccess = (uploadedFileId, filename) => {
    setFileId(uploadedFileId);
    setVideoUrl(`http://localhost:8000/api/video/${uploadedFileId}`);
    setSegments([]); // 새 파일 업로드 시 자막 초기화
    
    // 전처리 상태 초기화
    setPreprocessingStatus({
      status: 'queued',
      progress: 0,
      message: '전처리 대기 중'
    });
  };

  const handleTranscribeSuccess = (transcribedSegments) => {
    setSegments(transcribedSegments);
    setIsProcessing(false);
  };

  const handleSubtitleUpdate = (updatedSegments) => {
    setSegments(updatedSegments);
  };

  const handleSegmentClick = (time) => {
    if (videoPlayerRef.current) {
      videoPlayerRef.current.seek(time);
    }
  };

  const handleTranscribe = async () => {
    if (!fileId) {
      alert('먼저 비디오를 업로드해주세요.');
      return;
    }

    // 전처리 상태 확인
    if (preprocessingStatus.status === 'processing') {
      alert('전처리가 진행 중입니다. 잠시 후 다시 시도해주세요.');
      return;
    }

    setIsProcessing(true);
    try {
      const response = await fetch(`http://localhost:8000/api/transcribe/${fileId}`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || '변환에 실패했습니다.');
      }
      
      const data = await response.json();
      handleTranscribeSuccess(data.segments);
    } catch (error) {
      console.error('변환 실패:', error);
      alert(error.message || '변환 중 오류가 발생했습니다.');
      setIsProcessing(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Subtitle Maker</h1>
        <p>영상을 업로드하고 자동으로 자막을 생성하세요</p>
      </header>

      <main className="App-main">
        <div className="main-layout">
          {/* 좌측: 영상 영역 */}
          <div className="left-panel">
            <div className="video-section">
              {!videoUrl ? (
                <VideoUploader onUploadSuccess={handleUploadSuccess} />
              ) : (
                <>
                  <VideoPlayer ref={videoPlayerRef} videoUrl={videoUrl} segments={segments} />
                  
                  {/* 전처리 진행상황 표시 */}
                  {fileId && (
                    <PreprocessingStatus
                      status={preprocessingStatus.status}
                      progress={preprocessingStatus.progress}
                      message={preprocessingStatus.message}
                    />
                  )}
                  
                  <div className="video-actions">
                    <button
                      className="btn btn-secondary"
                      onClick={() => {
                        // WebSocket 연결 종료
                        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                          wsRef.current.close();
                        }
                        setFileId(null);
                        setVideoUrl(null);
                        setSegments([]);
                        setPreprocessingStatus({
                          status: 'not_started',
                          progress: 0,
                          message: ''
                        });
                      }}
                    >
                      새 영상 업로드
                    </button>
                  </div>
                </>
              )}
            </div>
            {segments.length > 0 && (
              <div className="download-section-left">
                <SubtitleDownloader fileId={fileId} />
              </div>
            )}
          </div>

          {/* 우측: 자막 영역 */}
          <div className="right-panel">
            <div className="subtitle-section">
              {segments.length === 0 ? (
                <div className="subtitle-empty-state">
                  <h2>자막 생성</h2>
                  <p>좌측에서 영상을 업로드한 후, 아래 버튼을 클릭하여 자막을 생성하세요.</p>
                  <button
                    className="btn btn-primary"
                    onClick={handleTranscribe}
                    disabled={!fileId || isProcessing || preprocessingStatus.status === 'processing'}
                  >
                    {isProcessing 
                      ? '변환 중...' 
                      : preprocessingStatus.status === 'processing' 
                        ? '전처리 중...' 
                        : '자막 생성하기'
                    }
                  </button>
                  {preprocessingStatus.status === 'processing' && (
                    <p style={{ marginTop: '10px', color: '#666' }}>
                      전처리가 완료되면 자막을 생성할 수 있습니다.
                    </p>
                  )}
                </div>
              ) : (
                <SubtitleEditor
                  segments={segments}
                  fileId={fileId}
                  onUpdate={handleSubtitleUpdate}
                  onRegenerate={handleTranscribe}
                  isProcessing={isProcessing}
                  onSegmentClick={handleSegmentClick}
                />
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
