// frontend/react_app/src/components/CaptionCard.tsx
import React, { useState } from 'react'
import { uploadImage, generateCaption } from '../services/api'
import { TaskCard } from './TaskCard'

interface CaptionResult {
  caption: string
  confidence: number
  model_used: string
  processing_time_ms: number
}

export const CaptionCard: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string>('')
  const [result, setResult] = useState<CaptionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>('')

  // Parameters
  const [maxLength, setMaxLength] = useState(50)
  const [numBeams, setNumBeams] = useState(5)
  const [temperature, setTemperature] = useState(1.0)

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      setError('')

      // Create preview
      const reader = new FileReader()
      reader.onload = (e) => {
        setPreview(e.target?.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleGenerate = async () => {
    if (!selectedFile) {
      setError('請先選擇圖片')
      return
    }

    setLoading(true)
    setError('')

    try {
      const result = await generateCaption(selectedFile, {
        max_length: maxLength,
        num_beams: numBeams,
        temperature: temperature
      })
      setResult(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : '生成失敗')
    } finally {
      setLoading(false)
    }
  }

  return (
    <TaskCard
      title="📸 BLIP-2 圖像描述生成"
      description="上傳圖片，AI 將自動生成詳細的圖像描述"
    >
      <div className="caption-interface">
        <div className="input-section">
          <div className="image-upload">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="file-input"
              id="image-upload"
            />
            <label htmlFor="image-upload" className="upload-button">
              {selectedFile ? '更換圖片' : '📁 選擇圖片'}
            </label>
          </div>

          {preview && (
            <div className="image-preview">
              <img src={preview} alt="Preview" className="preview-image" />
            </div>
          )}

          <div className="parameters">
            <div className="param-group">
              <label>最大長度: {maxLength}</label>
              <input
                type="range"
                min="10"
                max="200"
                step="5"
                value={maxLength}
                onChange={(e) => setMaxLength(Number(e.target.value))}
                className="slider"
              />
            </div>

            <div className="param-group">
              <label>束搜索數量: {numBeams}</label>
              <input
                type="range"
                min="1"
                max="10"
                value={numBeams}
                onChange={(e) => setNumBeams(Number(e.target.value))}
                className="slider"
              />
            </div>

            <div className="param-group">
              <label>溫度: {temperature}</label>
              <input
                type="range"
                min="0.1"
                max="2.0"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                className="slider"
              />
            </div>
          </div>

          <button
            onClick={handleGenerate}
            disabled={!selectedFile || loading}
            className="generate-button"
          >
            {loading ? '🔄 生成中...' : '🚀 生成描述'}
          </button>
        </div>

        <div className="result-section">
          {error && (
            <div className="error-message">
              ❌ {error}
            </div>
          )}

          {result && (
            <div className="result-card">
              <h3>生成結果</h3>
              <div className="caption-result">
                <p className="caption-text">"{result.caption}"</p>
                <div className="result-meta">
                  <span>信心度: {(result.confidence * 100).toFixed(1)}%</span>
                  <span>模型: {result.model_used}</span>
                  <span>處理時間: {result.processing_time_ms.toFixed(0)}ms</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </TaskCard>
  )
}
