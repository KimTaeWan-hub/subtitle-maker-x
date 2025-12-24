import React, { useRef, useEffect, useState, forwardRef, useImperativeHandle } from 'react';
import './VideoPlayer.css';

const VideoPlayer = forwardRef(({ videoUrl, segments = [] }, ref) => {
  const videoRef = useRef(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [currentSubtitle, setCurrentSubtitle] = useState('');

  useImperativeHandle(ref, () => ({
    seek(time) {
      if (videoRef.current) {
        videoRef.current.currentTime = time;
      }
    }
  }));

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const updateTime = () => {
      setCurrentTime(video.currentTime);
    };

    video.addEventListener('timeupdate', updateTime);
    return () => video.removeEventListener('timeupdate', updateTime);
  }, []);

  useEffect(() => {
    // 현재 시간에 맞는 자막 찾기
    const activeSubtitle = segments.find(
      (seg) => currentTime >= seg.start_time && currentTime <= seg.end_time
    );
    setCurrentSubtitle(activeSubtitle ? activeSubtitle.text : '');
  }, [currentTime, segments]);

  return (
    <div className="video-player-container">
      <video
        ref={videoRef}
        src={videoUrl}
        controls
        className="video-player"
      />
      {currentSubtitle && (
        <div className="subtitle-overlay">
          {currentSubtitle}
        </div>
      )}
      <div className="video-info">
        <p>현재 시간: {formatTime(currentTime)}</p>
      </div>
    </div>
  );
});

const formatTime = (seconds) => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  const millis = Math.floor((seconds % 1) * 1000);
  
  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${millis.toString().padStart(3, '0')}`;
  }
  return `${minutes}:${secs.toString().padStart(2, '0')}.${millis.toString().padStart(3, '0')}`;
};

export default VideoPlayer;

