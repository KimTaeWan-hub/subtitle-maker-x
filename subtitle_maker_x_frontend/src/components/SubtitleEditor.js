import React, { useState, useEffect, useRef } from 'react';
import './SubtitleEditor.css';

const SubtitleEditor = ({ segments, fileId, onUpdate, onRegenerate, isProcessing, onSegmentClick }) => {
  const [editedSegments, setEditedSegments] = useState(segments);
  const [saving, setSaving] = useState(false);
  const textareaRefs = useRef({});

  useEffect(() => {
    setEditedSegments(segments);
  }, [segments]);

  useEffect(() => {
    // 세그먼트가 변경되면 모든 textarea 높이 조절
    editedSegments.forEach((segment, index) => {
      const textarea = textareaRefs.current[index];
      if (textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = `${textarea.scrollHeight}px`;
      }
    });
  }, [editedSegments]);

  const handleTextChange = (index, newText) => {
    const updated = editedSegments.map((seg, i) =>
      i === index ? { ...seg, text: newText } : seg
    );
    setEditedSegments(updated);
    
    // textarea 높이 자동 조절
    const textarea = textareaRefs.current[index];
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${textarea.scrollHeight}px`;
    }
  };

  const handleTimeChange = (index, field, value) => {
    const updated = editedSegments.map((seg, i) =>
      i === index ? { ...seg, [field]: parseFloat(value) } : seg
    );
    setEditedSegments(updated);
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      const response = await fetch(`http://localhost:8000/api/subtitles/${fileId}/edit`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          segments: editedSegments,
        }),
      });

      if (!response.ok) {
        throw new Error('저장 실패');
      }

      const data = await response.json();
      onUpdate(data.segments);
      alert('자막이 성공적으로 저장되었습니다!');
    } catch (error) {
      console.error('저장 오류:', error);
      alert('저장 중 오류가 발생했습니다.');
    } finally {
      setSaving(false);
    }
  };

  const formatTime = (seconds) => {
    if (seconds === undefined || seconds === null) return '0:00.00';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = (seconds % 60).toFixed(2);
    
    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.padStart(5, '0')}`;
    }
    return `${minutes}:${secs.padStart(5, '0')}`;
  };

  const getSpeakerColor = (speaker) => {
    if (!speaker) return '#999';
    const colors = {
      'SPEAKER_00': '#667eea',
      'SPEAKER_01': '#f093fb',
      'SPEAKER_02': '#4facfe',
      'SPEAKER_03': '#43e97b',
      'SPEAKER_04': '#fa709a',
    };
    return colors[speaker] || '#999';
  };

  // 원본과 편집본을 비교하여 변경사항이 있는지 확인
  const hasChanges = () => {
    if (segments.length !== editedSegments.length) {
      return true;
    }
    
    // 각 세그먼트를 정확히 비교
    return segments.some((original, index) => {
      const edited = editedSegments[index];
      if (!edited) return true;
      
      const originalStart = original.start || original.start_time;
      const originalEnd = original.end || original.end_time;
      const editedStart = edited.start || edited.start_time;
      const editedEnd = edited.end || edited.end_time;
      
      return (
        Math.abs(originalStart - editedStart) > 0.001 ||
        Math.abs(originalEnd - editedEnd) > 0.001 ||
        original.text !== edited.text
      );
    });
  };

  return (
    <div className="subtitle-editor">
      <div className="editor-header">
        <h2>자막 편집</h2>
        <div className="editor-actions">
          {onRegenerate && (
            <button
              className="btn btn-secondary"
              onClick={onRegenerate}
              disabled={isProcessing}
            >
              {isProcessing ? '재생성 중...' : '자막 재생성'}
            </button>
          )}
          <button
            className="btn btn-primary"
            onClick={handleSave}
            disabled={saving || !hasChanges()}
          >
            {saving ? '저장 중...' : '저장하기'}
          </button>
        </div>
      </div>

      <div className="segments-list">
        {editedSegments.map((segment, index) => (
          <div 
            key={index} 
            className="segment-item"
            onClick={() => onSegmentClick && onSegmentClick(segment.start || segment.start_time)}
          >
            <div className="segment-header">
              <div className="segment-info">
                <span className="segment-id">#{index + 1}</span>
                {segment.speaker && (
                  <span 
                    className="speaker-badge"
                    style={{ backgroundColor: getSpeakerColor(segment.speaker) }}
                  >
                    {segment.speaker}
                  </span>
                )}
              </div>
              <div className="segment-times">
                <span className="time-display">
                  {formatTime(segment.start || segment.start_time)} → {formatTime(segment.end || segment.end_time)}
                </span>
                <span className="duration-display">
                  ({((segment.end || segment.end_time) - (segment.start || segment.start_time)).toFixed(2)}초)
                </span>
              </div>
            </div>
            <textarea
              ref={(el) => (textareaRefs.current[index] = el)}
              value={segment.text}
              onChange={(e) => handleTextChange(index, e.target.value)}
              className="segment-text"
              rows="1"
              placeholder="자막 텍스트를 입력하세요"
            />
          </div>
        ))}
      </div>
    </div>
  );
};

export default SubtitleEditor;

