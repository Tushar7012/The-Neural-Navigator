import { useState, useCallback, useEffect } from 'react'
import './App.css'

// Sample test images from the dataset
const SAMPLE_IMAGES = [
  { id: '000000', command: 'Go to the Green Square' },
  { id: '000001', command: 'Go to the Blue Triangle' },
  { id: '000002', command: 'Go to the Red Circle' },
  { id: '000003', command: 'Go to the Green Square' },
]

const COMMANDS = [
  { label: 'Red Circle', color: 'red', text: 'Go to the Red Circle' },
  { label: 'Blue Triangle', color: 'blue', text: 'Go to the Blue Triangle' },
  { label: 'Green Square', color: 'green', text: 'Go to the Green Square' },
]

// Simulated path prediction (in real app, this would call the backend)
const simulatePrediction = (command) => {
  const baseX = 64, baseY = 64
  const targets = {
    'Go to the Red Circle': { x: 95, y: 45 },
    'Go to the Blue Triangle': { x: 30, y: 100 },
    'Go to the Green Square': { x: 100, y: 90 },
  }
  
  const target = targets[command] || { x: 100, y: 100 }
  const path = []
  
  for (let i = 0; i < 10; i++) {
    const t = i / 9
    const x = baseX + (target.x - baseX) * t + (Math.random() - 0.5) * 5
    const y = baseY + (target.y - baseY) * t + (Math.random() - 0.5) * 5
    path.push({ x, y })
  }
  
  return path
}

function App() {
  const [image, setImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [command, setCommand] = useState('')
  const [selectedCommand, setSelectedCommand] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [isDragging, setIsDragging] = useState(false)

  const handleImageUpload = useCallback((file) => {
    if (file && file.type.startsWith('image/')) {
      setImage(file)
      const reader = new FileReader()
      reader.onloadend = () => {
        setImagePreview(reader.result)
      }
      reader.readAsDataURL(file)
      setResult(null)
    }
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    handleImageUpload(file)
  }, [handleImageUpload])

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback(() => {
    setIsDragging(false)
  }, [])

  const handleFileSelect = useCallback((e) => {
    const file = e.target.files[0]
    handleImageUpload(file)
  }, [handleImageUpload])

  const handleCommandSelect = useCallback((cmd) => {
    setSelectedCommand(cmd)
    setCommand(cmd.text)
  }, [])

  const handlePredict = useCallback(async () => {
    if (!imagePreview || !command) return
    
    setIsLoading(true)
    
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1500))
    
    const predictedPath = simulatePrediction(command)
    
    setResult({
      path: predictedPath,
      confidence: (0.85 + Math.random() * 0.1).toFixed(2),
      inferenceTime: (50 + Math.random() * 100).toFixed(0),
    })
    
    setIsLoading(false)
  }, [imagePreview, command])

  const loadSampleImage = useCallback(async (sample) => {
    // Create a sample image preview (in real app, would load from dataset)
    const canvas = document.createElement('canvas')
    canvas.width = 128
    canvas.height = 128
    const ctx = canvas.getContext('2d')
    
    // White background
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, 128, 128)
    
    // Draw shapes
    // Red circle
    ctx.fillStyle = '#ef4444'
    ctx.beginPath()
    ctx.arc(95, 45, 15, 0, Math.PI * 2)
    ctx.fill()
    
    // Blue triangle
    ctx.fillStyle = '#3b82f6'
    ctx.beginPath()
    ctx.moveTo(30, 85)
    ctx.lineTo(15, 115)
    ctx.lineTo(45, 115)
    ctx.closePath()
    ctx.fill()
    
    // Green square
    ctx.fillStyle = '#22c55e'
    ctx.fillRect(85, 75, 30, 30)
    
    setImagePreview(canvas.toDataURL())
    setCommand(sample.command)
    setSelectedCommand(COMMANDS.find(c => c.text === sample.command))
    setResult(null)
  }, [])

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <div className="logo-icon">🧭</div>
            <span className="logo-text">Neural Navigator</span>
          </div>
          <nav className="nav-links">
            <a href="https://github.com/Tushar7012/The-Neural-Navigator" className="nav-link" target="_blank" rel="noopener noreferrer">
              GitHub
            </a>
            <a href="#demo" className="nav-link">Demo</a>
            <a href="#docs" className="nav-link">Docs</a>
          </nav>
        </div>
      </header>

      {/* Hero */}
      <section className="hero">
        <h1 className="animate-fade-in">Smart GPS Navigation</h1>
        <p className="animate-fade-in" style={{ animationDelay: '0.1s' }}>
          A multi-modal AI that predicts navigation paths from map images and text commands.
          Powered by CNN + LSTM + Transformer architecture.
        </p>
        <div className="hero-badges animate-fade-in" style={{ animationDelay: '0.2s' }}>
          <span className="badge">
            <span className="badge-icon">🔥</span>
            PyTorch
          </span>
          <span className="badge">
            <span className="badge-icon">🧠</span>
            Multi-Modal
          </span>
          <span className="badge">
            <span className="badge-icon">⚡</span>
            10.5M Parameters
          </span>
          <span className="badge">
            <span className="badge-icon">📊</span>
            0.0109 MSE
          </span>
        </div>
      </section>

      {/* Main Content */}
      <main className="main" id="demo">
        <div className="grid">
          {/* Input Panel */}
          <div className="card animate-fade-in">
            <div className="card-header">
              <div className="card-icon">📤</div>
              <h2 className="card-title">Input</h2>
            </div>

            {/* Image Upload */}
            <div className="form-group">
              <label className="form-label">Map Image</label>
              <div
                className={`upload-area ${isDragging ? 'dragging' : ''} ${imagePreview ? 'has-image' : ''}`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => document.getElementById('file-input').click()}
              >
                {imagePreview ? (
                  <img src={imagePreview} alt="Preview" className="preview-image" />
                ) : (
                  <>
                    <div className="upload-icon">🖼️</div>
                    <p className="upload-text">Drop your map image here or click to browse</p>
                    <p className="upload-hint">Supports PNG, JPG (128×128 recommended)</p>
                  </>
                )}
                <input
                  type="file"
                  id="file-input"
                  accept="image/*"
                  onChange={handleFileSelect}
                  style={{ display: 'none' }}
                />
              </div>
            </div>

            {/* Command Selection */}
            <div className="form-group">
              <label className="form-label">Navigation Command</label>
              <div className="command-buttons">
                {COMMANDS.map((cmd) => (
                  <button
                    key={cmd.text}
                    className={`command-btn ${selectedCommand?.text === cmd.text ? 'active' : ''}`}
                    onClick={() => handleCommandSelect(cmd)}
                  >
                    <span className={`color-dot ${cmd.color}`}></span>
                    {cmd.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Custom Command */}
            <div className="form-group">
              <label className="form-label">Or enter custom command</label>
              <input
                type="text"
                className="form-input"
                placeholder="e.g., Go to the Red Circle"
                value={command}
                onChange={(e) => {
                  setCommand(e.target.value)
                  setSelectedCommand(null)
                }}
              />
            </div>

            {/* Predict Button */}
            <button
              className="btn-primary"
              onClick={handlePredict}
              disabled={!imagePreview || !command || isLoading}
            >
              {isLoading ? (
                <>
                  <span className="spinner" style={{ width: 20, height: 20, borderWidth: 2 }}></span>
                  Processing...
                </>
              ) : (
                <>
                  🚀 Predict Path
                </>
              )}
            </button>
          </div>

          {/* Result Panel */}
          <div className="result-card animate-fade-in" style={{ animationDelay: '0.1s' }}>
            <div className="result-header">
              <span className="result-title">
                📍 Predicted Path
              </span>
              {result && (
                <span className="result-status">Success</span>
              )}
            </div>
            <div className="result-body">
              <div className="result-canvas">
                {imagePreview && (
                  <img src={imagePreview} alt="Map" className="result-image" />
                )}
                
                {/* Path Overlay */}
                {result && (
                  <div className="path-overlay">
                    <svg viewBox="0 0 128 128" preserveAspectRatio="xMidYMid meet">
                      {/* Path line */}
                      <path
                        className="path-line"
                        d={`M ${result.path.map(p => `${p.x},${p.y}`).join(' L ')}`}
                      />
                      
                      {/* Path points */}
                      {result.path.map((point, i) => (
                        <circle
                          key={i}
                          cx={point.x}
                          cy={point.y}
                          r={i === 0 ? 5 : i === result.path.length - 1 ? 5 : 3}
                          className={`path-point ${i === 0 ? 'path-start' : i === result.path.length - 1 ? 'path-end' : ''}`}
                        />
                      ))}
                    </svg>
                  </div>
                )}
                
                {isLoading && (
                  <div className="loading-overlay">
                    <div className="spinner"></div>
                    <p className="loading-text">Running inference...</p>
                  </div>
                )}
                
                {!imagePreview && !isLoading && (
                  <div className="loading-overlay" style={{ background: 'var(--bg-dark)' }}>
                    <p className="loading-text">Upload an image to see predictions</p>
                  </div>
                )}
              </div>

              {/* Metrics */}
              {result && (
                <div className="metrics">
                  <div className="metric">
                    <div className="metric-value">{result.path.length}</div>
                    <div className="metric-label">Path Points</div>
                  </div>
                  <div className="metric">
                    <div className="metric-value">{result.confidence}</div>
                    <div className="metric-label">Confidence</div>
                  </div>
                  <div className="metric">
                    <div className="metric-value">{result.inferenceTime}ms</div>
                    <div className="metric-label">Inference</div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Sample Images */}
        <section className="samples-section animate-fade-in" style={{ animationDelay: '0.2s' }}>
          <h3 className="samples-title">Try Sample Images</h3>
          <div className="samples-grid">
            {SAMPLE_IMAGES.map((sample) => (
              <div
                key={sample.id}
                className="sample-card"
                onClick={() => loadSampleImage(sample)}
              >
                <div 
                  className="sample-image" 
                  style={{ 
                    background: 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '3rem'
                  }}
                >
                  🗺️
                </div>
                <div className="sample-info">
                  {sample.command}
                </div>
              </div>
            ))}
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>
          Built with ❤️ using PyTorch & React | 
          <a href="https://github.com/Tushar7012/The-Neural-Navigator" target="_blank" rel="noopener noreferrer"> View on GitHub</a>
        </p>
      </footer>
    </div>
  )
}

export default App
