import React from 'react';
import './SubtitleDownloader.css';

const SubtitleDownloader = ({ fileId }) => {
  const handleDownload = async (format) => {
    try {
      const response = await fetch(`http://localhost:8000/api/download/${fileId}/${format}`);
      
      if (!response.ok) {
        throw new Error('다운로드 실패');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `subtitle.${format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('다운로드 오류:', error);
      alert('다운로드 중 오류가 발생했습니다.');
    }
  };

  return (
    <div className="subtitle-downloader">
      <h3>자막 다운로드</h3>
      <div className="download-buttons">
        <button
          className="btn btn-download"
          onClick={() => handleDownload('srt')}
        >
          SRT 다운로드
        </button>
        <button
          className="btn btn-download"
          onClick={() => handleDownload('vtt')}
        >
          VTT 다운로드
        </button>
        <button
          className="btn btn-download"
          onClick={() => handleDownload('txt')}
        >
          TXT 다운로드
        </button>
      </div>
    </div>
  );
};

export default SubtitleDownloader;

