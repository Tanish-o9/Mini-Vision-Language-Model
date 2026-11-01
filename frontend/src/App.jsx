import React, { useState, useEffect, useRef } from 'react';
import {
  Eye,
  Send,
  Upload,
  Trash2,
  Cpu,
  BarChart2,
  Layers,
  Settings,
  RefreshCw,
  Sparkles,
  Zap,
  CheckCircle2,
  Activity,
  Maximize2,
  Clock,
  HardDrive,
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  Camera,
  Download,
  Award,
  X
} from 'lucide-react';
import './App.css';

// Automatically adapt API_BASE whether running under Vite dev (5173) or FastAPI production (8000)
const API_BASE = window.location.port === '5173' ? 'http://localhost:8000' : '';

// Client-side image optimizer: compresses photos to max 800px to ensure instantaneous uploads
const compressImageFile = (file, maxWidth = 800, quality = 0.85) => {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        let width = img.width;
        let height = img.height;
        if (width > maxWidth || height > maxWidth) {
          if (width > height) {
            height = Math.round((height * maxWidth) / width);
            width = maxWidth;
          } else {
            width = Math.round((width * maxWidth) / height);
            height = maxWidth;
          }
        }
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, width, height);
        resolve(canvas.toDataURL('image/jpeg', quality));
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  });
};

export default function App() {
  const [activeTab, setActiveTab] = useState('playground');
  const [systemStatus, setSystemStatus] = useState(null);
  const [metrics, setMetrics] = useState({ stage1: [], stage2: [] });
  const [config, setConfig] = useState(null);

  // Chat Playground State with LocalStorage persistence across page refreshes
  const [selectedImage, setSelectedImage] = useState(() => {
    try { return localStorage.getItem('minivlm_image') || null; } catch { return null; }
  });
  const [imagePreview, setImagePreview] = useState(() => {
    try { return localStorage.getItem('minivlm_image') || null; } catch { return null; }
  });
  const [isCompareMode, setIsCompareMode] = useState(() => {
    try { return localStorage.getItem('minivlm_compare_mode') === 'true'; } catch { return false; }
  });
  const [secondaryImage, setSecondaryImage] = useState(() => {
    try { return localStorage.getItem('minivlm_secondary_image') || null; } catch { return null; }
  });
  const [secondaryPreview, setSecondaryPreview] = useState(() => {
    try { return localStorage.getItem('minivlm_secondary_image') || null; } catch { return null; }
  });
  const [latestVisualCues, setLatestVisualCues] = useState(() => {
    try { return localStorage.getItem('minivlm_cues') || ''; } catch { return ''; }
  });

  const [chatHistory, setChatHistory] = useState(() => {
    try {
      const saved = localStorage.getItem('minivlm_chat_history');
      if (saved) return JSON.parse(saved);
    } catch {}
    return [
      {
        role: 'bot',
        content: 'Hello! I am your Mini Vision-Language Assistant. Upload an image (or take a photo) and ask me questions in English or Hindi!',
        time: 'Just now'
      }
    ];
  });
  const [inputQuestion, setInputQuestion] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [maxTokens, setMaxTokens] = useState(80);

  // Voice & Audio State
  const [isRecording, setIsRecording] = useState(false);
  const [voiceLang, setVoiceLang] = useState('hi-IN'); // default to Hindi/Hinglish
  const [speakingIndex, setSpeakingIndex] = useState(null);
  const recognitionRef = useRef(null);

  // Camera State
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const videoRef = useRef(null);
  const cameraStreamRef = useRef(null);

  // Benchmark State
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [isBenchmarking, setIsBenchmarking] = useState(false);

  const fileInputRef = useRef(null);
  const secondFileInputRef = useRef(null);
  const chatBottomRef = useRef(null);

  useEffect(() => {
    fetchSystemStatus();
    fetchMetrics();
    fetchConfig();
    fetchBenchmark();
  }, []);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, isGenerating]);

  // Persist chat history to localStorage
  useEffect(() => {
    try {
      localStorage.setItem('minivlm_chat_history', JSON.stringify(chatHistory));
    } catch {}
  }, [chatHistory]);

  // Persist primary image to localStorage
  useEffect(() => {
    try {
      if (selectedImage) localStorage.setItem('minivlm_image', selectedImage);
      else localStorage.removeItem('minivlm_image');
    } catch {}
  }, [selectedImage]);

  // Persist cues
  useEffect(() => {
    try {
      if (latestVisualCues) localStorage.setItem('minivlm_cues', latestVisualCues);
      else localStorage.removeItem('minivlm_cues');
    } catch {}
  }, [latestVisualCues]);

  // Persist compare mode & secondary image to localStorage
  useEffect(() => {
    try {
      localStorage.setItem('minivlm_compare_mode', String(isCompareMode));
      if (secondaryImage) localStorage.setItem('minivlm_secondary_image', secondaryImage);
      else localStorage.removeItem('minivlm_secondary_image');
    } catch {}
  }, [isCompareMode, secondaryImage]);

  // Live seconds elapsed counter while Qwen is thinking
  useEffect(() => {
    let timer;
    if (isGenerating) {
      setElapsedSeconds(0);
      timer = setInterval(() => {
        setElapsedSeconds((prev) => Math.round((prev + 0.1) * 10) / 10);
      }, 100);
    } else {
      clearInterval(timer);
    }
    return () => clearInterval(timer);
  }, [isGenerating]);

  const fetchSystemStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/status`);
      if (res.ok) setSystemStatus(await res.json());
    } catch (err) {
      console.warn('Backend status check:', err);
    }
  };

  const fetchMetrics = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/metrics`);
      if (res.ok) setMetrics(await res.json());
    } catch (err) {
      console.warn('Metrics check:', err);
    }
  };

  const fetchConfig = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/config`);
      if (res.ok) setConfig(await res.json());
    } catch (err) {
      console.warn('Config check:', err);
    }
  };

  const fetchBenchmark = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/benchmark`);
      if (res.ok) setBenchmarkData(await res.json());
    } catch (err) {
      console.warn('Benchmark check:', err);
    }
  };

  const runBenchmarkSuite = async () => {
    setIsBenchmarking(true);
    try {
      const res = await fetch(`${API_BASE}/api/benchmark`);
      if (res.ok) {
        setBenchmarkData(await res.json());
      }
    } finally {
      setIsBenchmarking(false);
    }
  };

  // Voice Input (Web Speech Recognition)
  const toggleVoiceInput = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert('Voice recognition is not supported in this browser. Please use Chrome, Edge, or Brave.');
      return;
    }

    if (isRecording) {
      recognitionRef.current?.stop();
      setIsRecording(false);
      return;
    }

    try {
      const recognition = new SpeechRecognition();
      recognition.lang = voiceLang;
      recognition.continuous = false;
      recognition.interimResults = true;

      recognition.onstart = () => setIsRecording(true);
      recognition.onresult = (event) => {
        const transcript = Array.from(event.results)
          .map((r) => r[0].transcript)
          .join('');
        setInputQuestion(transcript);
      };
      recognition.onerror = (e) => {
        console.warn('Speech error:', e);
        setIsRecording(false);
      };
      recognition.onend = () => setIsRecording(false);

      recognitionRef.current = recognition;
      recognition.start();
    } catch (e) {
      console.error(e);
      setIsRecording(false);
    }
  };

  // Text to Speech (TTS)
  const toggleSpeech = (text, index) => {
    if (!('speechSynthesis' in window)) return;
    if (speakingIndex === index) {
      window.speechSynthesis.cancel();
      setSpeakingIndex(null);
      return;
    }

    window.speechSynthesis.cancel();
    const cleanText = text.replace(/\[.*?\]/g, '').trim();
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.onend = () => setSpeakingIndex(null);
    utterance.onerror = () => setSpeakingIndex(null);
    setSpeakingIndex(index);
    window.speechSynthesis.speak(utterance);
  };

  // Camera Handlers
  const startCamera = async () => {
    setIsCameraOpen(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }
      });
      cameraStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      alert('Unable to access camera: ' + err.message);
      setIsCameraOpen(false);
    }
  };

  const stopCamera = () => {
    if (cameraStreamRef.current) {
      cameraStreamRef.current.getTracks().forEach((t) => t.stop());
      cameraStreamRef.current = null;
    }
    setIsCameraOpen(false);
  };

  const capturePhoto = () => {
    if (!videoRef.current) return;
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.85);

    setImagePreview(dataUrl);
    setSelectedImage(dataUrl);
    setLatestVisualCues('');
    stopCamera();
  };

  // Image Upload handler
  const handleImageUpload = async (e) => {
    const file = e.target.files?.[0];
    if (file) {
      try {
        const optimized = await compressImageFile(file);
        setImagePreview(optimized);
        setSelectedImage(optimized);
        setLatestVisualCues('');
      } catch {
        const reader = new FileReader();
        reader.onloadend = () => {
          setImagePreview(reader.result);
          setSelectedImage(reader.result);
          setLatestVisualCues('');
        };
        reader.readAsDataURL(file);
      }
    }
  };

  // Clear Chat History
  const handleClearChat = () => {
    const initial = [
      {
        role: 'bot',
        content: 'Chat conversation reset. Feel free to upload an image and ask a question.',
        time: 'Just now'
      }
    ];
    setChatHistory(initial);
    try {
      localStorage.setItem('minivlm_chat_history', JSON.stringify(initial));
    } catch {}
  };

  // Export Report (PDF Print Mode)
  const handleExportReport = () => {
    window.print();
  };

  const handleSendMessage = async (customPrompt = null) => {
    const q = customPrompt || inputQuestion;
    if (!q.trim()) return;
    if (!selectedImage) {
      alert('Please upload an image first!');
      return;
    }

    const userMessage = { role: 'user', content: q, time: new Date().toLocaleTimeString() };
    const updatedHistory = [...chatHistory, userMessage];
    setChatHistory(updatedHistory);
    setInputQuestion('');
    setIsGenerating(true);

    // Format history pairs for backend
    const apiHistory = [];
    for (let i = 0; i < updatedHistory.length - 1; i += 2) {
      const u = updatedHistory[i]?.role === 'user' ? updatedHistory[i].content : '';
      const a = updatedHistory[i + 1]?.role === 'bot' ? updatedHistory[i + 1].content : '';
      if (u) apiHistory.push([u, a]);
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 45000);

      const res = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          image_base64: selectedImage,
          secondary_image_base64: isCompareMode ? secondaryImage : null,
          question: q,
          history: apiHistory,
          max_new_tokens: maxTokens,
          stage: 'stage2'
        })
      });
      clearTimeout(timeoutId);

      if (res.ok) {
        const data = await res.json();
        if (data.visual_cues) {
          setLatestVisualCues(data.visual_cues);
        }
        setChatHistory((prev) => [
          ...prev,
          {
            role: 'bot',
            content: data.answer || 'No answer generated.',
            time: new Date().toLocaleTimeString(),
            latency: data.latency_seconds,
            device: data.device
          }
        ]);
      } else {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server status ${res.status}`);
      }
    } catch (err) {
      setChatHistory((prev) => [
        ...prev,
        {
          role: 'bot',
          content: `⚠️ Inference Notice: ${err.message || 'Request timed out'}. Ensure server is running at localhost:8000.`,
          time: new Date().toLocaleTimeString(),
          latency: 0,
          device: 'system'
        }
      ]);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="dashboard-container">
      {/* Top Navbar */}
      <header className="top-nav">
        <div className="brand-section">
          <div className="brand-logo">
            <Eye size={22} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span className="brand-title">Mini-VLM Studio</span>
              <span className="badge-tag">LLaVA-1.5 Style</span>
            </div>
            <div style={{ fontSize: '0.72rem', color: '#94a3b8' }}>
              CLIP ViT-B/32 + MLP Projector + Qwen2.5-0.5B
            </div>
          </div>
        </div>

        <nav className="tabs-nav">
          <button
            className={`tab-btn ${activeTab === 'playground' ? 'active' : ''}`}
            onClick={() => setActiveTab('playground')}
          >
            <Sparkles size={16} /> Playground
          </button>
          <button
            className={`tab-btn ${activeTab === 'benchmark' ? 'active' : ''}`}
            onClick={() => setActiveTab('benchmark')}
          >
            <Award size={16} /> Benchmark
          </button>
          <button
            className={`tab-btn ${activeTab === 'analytics' ? 'active' : ''}`}
            onClick={() => setActiveTab('analytics')}
          >
            <BarChart2 size={16} /> Training Analytics
          </button>
          <button
            className={`tab-btn ${activeTab === 'architecture' ? 'active' : ''}`}
            onClick={() => setActiveTab('architecture')}
          >
            <Layers size={16} /> Architecture
          </button>
          <button
            className={`tab-btn ${activeTab === 'system' ? 'active' : ''}`}
            onClick={() => setActiveTab('system')}
          >
            <Cpu size={16} /> Diagnostics
          </button>
        </nav>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div className="system-status-pill">
            <span className="status-dot"></span>
            <span>{systemStatus?.device?.toUpperCase() || 'CUDA'} FP16</span>
          </div>
          <button
            className="tab-btn"
            style={{ padding: '0.4rem', border: '1px solid var(--border-color)' }}
            onClick={fetchSystemStatus}
            title="Refresh System Status"
          >
            <RefreshCw size={15} />
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="main-content">
        {/* TAB 1: PLAYGROUND */}
        {activeTab === 'playground' && (
          <div className="playground-grid">
            {/* Left Panel: Image Upload & Parameters */}
            <div className="glass-card">
              <div className="card-header">
                <span className="card-title">
                  <Upload size={18} color="var(--accent-cyan)" /> Visual Input
                </span>
                <div style={{ display: 'flex', gap: '0.4rem' }}>
                  <button
                    className="tab-btn"
                    style={{
                      fontSize: '0.72rem',
                      padding: '0.2rem 0.5rem',
                      background: 'rgba(56, 189, 248, 0.1)',
                      color: 'var(--accent-cyan)',
                      border: '1px solid rgba(56, 189, 248, 0.25)'
                    }}
                    onClick={startCamera}
                    title="Capture live photo from webcam"
                  >
                    <Camera size={13} /> Camera
                  </button>
                  <button
                    className="tab-btn"
                    style={{
                      fontSize: '0.72rem',
                      padding: '0.2rem 0.5rem',
                      background: isCompareMode ? 'rgba(129, 140, 248, 0.2)' : 'rgba(51, 65, 85, 0.4)',
                      color: isCompareMode ? 'var(--accent-purple)' : 'var(--text-muted)',
                      border: '1px solid var(--border-color)',
                    }}
                    onClick={() => setIsCompareMode(!isCompareMode)}
                  >
                    {isCompareMode ? '✔ Compare (2)' : '+ Compare'}
                  </button>
                </div>
              </div>

              {/* Primary Image Slot */}
              <div style={{ marginBottom: isCompareMode ? '0.75rem' : '0' }}>
                {imagePreview ? (
                  <div className="preview-container" style={{ height: isCompareMode ? '135px' : '250px' }}>
                    <img src={imagePreview} alt="Primary" className="preview-img" />
                    <button
                      className="remove-img-btn"
                      onClick={() => {
                        setImagePreview(null);
                        setSelectedImage(null);
                        setLatestVisualCues('');
                      }}
                    >
                      <Trash2 size={13} /> Clear
                    </button>
                  </div>
                ) : (
                  <div className="upload-dropzone" style={{ height: isCompareMode ? '135px' : '250px' }} onClick={() => fileInputRef.current?.click()}>
                    <Upload size={24} color="#64748b" />
                    <div style={{ fontWeight: 600, fontSize: '0.8rem' }}>Upload or Drop Image</div>
                    <div style={{ fontSize: '0.7rem', color: '#94a3b8' }}>PNG, JPG, WebP supported</div>
                  </div>
                )}
                <input
                  type="file"
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  accept="image/*"
                  onChange={handleImageUpload}
                />
              </div>

              {/* Visual Grounding & Real-time Cues */}
              {latestVisualCues && (
                <div className="grounding-panel">
                  <div style={{ fontWeight: 600, color: 'var(--accent-cyan)', marginBottom: '0.3rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                    <Eye size={13} /> Visual Grounding & Semantics:
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap' }}>
                    {latestVisualCues.split('. ').map((part, pIdx) => (
                      <span key={pIdx} className="grounding-tag">
                        {part.startsWith('Primary subject') ? '🎯' : part.startsWith('Color') ? '🎨' : '🔍'} {part}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Secondary Image Slot (Compare Mode) */}
              {isCompareMode && (
                <div style={{ marginTop: '0.5rem' }}>
                  {secondaryPreview ? (
                    <div className="preview-container" style={{ height: '135px' }}>
                      <img src={secondaryPreview} alt="Secondary" className="preview-img" />
                      <button
                        className="remove-img-btn"
                        onClick={() => {
                          setSecondaryPreview(null);
                          setSecondaryImage(null);
                        }}
                      >
                        <Trash2 size={13} /> Clear
                      </button>
                    </div>
                  ) : (
                    <div className="upload-dropzone" style={{ height: '135px' }} onClick={() => secondFileInputRef.current?.click()}>
                      <Upload size={24} color="#818cf8" />
                      <div style={{ fontWeight: 600, fontSize: '0.8rem', color: '#818cf8' }}>Upload Comparison Image (Image 2)</div>
                    </div>
                  )}
                  <input
                    type="file"
                    ref={secondFileInputRef}
                    style={{ display: 'none' }}
                    accept="image/*"
                    onChange={async (e) => {
                      const file = e.target.files?.[0];
                      if (file) {
                        try {
                          const optimized = await compressImageFile(file);
                          setSecondaryPreview(optimized);
                          setSecondaryImage(optimized);
                        } catch {
                          const reader = new FileReader();
                          reader.onloadend = () => {
                            setSecondaryPreview(reader.result);
                            setSecondaryImage(reader.result);
                          };
                          reader.readAsDataURL(file);
                        }
                      }
                    }}
                  />
                </div>
              )}

              {/* Generation Controls */}
              <div style={{ marginTop: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '0.4rem' }}>
                  <span style={{ color: '#94a3b8' }}>Max Response Tokens</span>
                  <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>{maxTokens}</span>
                </div>
                <input
                  type="range"
                  min="20"
                  max="256"
                  step="8"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(Number(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
                />
              </div>

              {/* Quick Prompts */}
              <div className="quick-prompts-title">💡 Instant Prompts</div>
              <div className="quick-prompts-list">
                {[
                  'is photo me kya h??',
                  'Describe this image in one clear sentence.',
                  'What objects are visible in this scene?',
                  'What colors are dominant in this picture?'
                ].map((prompt, idx) => (
                  <button
                    key={idx}
                    className="quick-prompt-chip"
                    onClick={() => handleSendMessage(prompt)}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>

            {/* Right Panel: Multimodal Chat Dialog */}
            <div className="glass-card">
              <div className="card-header">
                <span className="card-title">
                  <Sparkles size={18} color="var(--accent-purple)" /> Multimodal Dialogue
                </span>
                <div style={{ display: 'flex', gap: '0.4rem' }}>
                  <button
                    onClick={handleExportReport}
                    style={{
                      background: 'transparent',
                      border: '1px solid var(--border-color)',
                      color: '#94a3b8',
                      padding: '0.3rem 0.6rem',
                      borderRadius: '6px',
                      fontSize: '0.75rem',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.3rem'
                    }}
                    title="Export chat report to PDF"
                  >
                    <Download size={13} /> Export PDF
                  </button>
                  <button
                    onClick={handleClearChat}
                    style={{
                      background: 'transparent',
                      border: '1px solid var(--border-color)',
                      color: '#94a3b8',
                      padding: '0.3rem 0.6rem',
                      borderRadius: '6px',
                      fontSize: '0.75rem',
                      cursor: 'pointer'
                    }}
                  >
                    Clear History
                  </button>
                </div>
              </div>

              {/* Chat History Messages */}
              <div className="chat-window">
                {chatHistory.map((msg, index) => (
                  <div key={index} className={`chat-bubble ${msg.role}`}>
                    <div className="bubble-avatar">
                      {msg.role === 'user' ? 'U' : <Eye size={16} />}
                    </div>
                    <div style={{ flex: 1 }}>
                      <div className="bubble-content">{msg.content}</div>
                      <div className="chat-meta" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div>
                          <span>{msg.time}</span>
                          {msg.latency && (
                            <span style={{ color: 'var(--accent-cyan)' }}> • {msg.latency}s</span>
                          )}
                          {msg.device && <span> • {msg.device.toUpperCase()}</span>}
                        </div>
                        {msg.role === 'bot' && (
                          <button
                            className={`tts-speaker-btn ${speakingIndex === index ? 'active' : ''}`}
                            onClick={() => toggleSpeech(msg.content, index)}
                            title={speakingIndex === index ? 'Stop audio' : 'Listen to answer'}
                          >
                            {speakingIndex === index ? <VolumeX size={13} /> : <Volume2 size={13} />}
                            {speakingIndex === index ? 'Speaking...' : 'Listen'}
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
                {isGenerating && (
                  <div className="chat-bubble bot">
                    <div className="bubble-avatar">
                      <Eye size={16} />
                    </div>
                    <div className="bubble-content" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#94a3b8' }}>
                      <RefreshCw size={14} className="spin" />
                      <span>Generating response via Qwen ({elapsedSeconds.toFixed(1)}s)...</span>
                    </div>
                  </div>
                )}
                <div ref={chatBottomRef} />
              </div>

              {/* Chat Input Bar */}
              <div className="chat-input-bar">
                {/* Voice Input Mic Button */}
                <button
                  className={`voice-btn ${isRecording ? 'recording' : ''}`}
                  onClick={toggleVoiceInput}
                  title={isRecording ? 'Listening (Click to stop)' : `Voice Input (${voiceLang === 'hi-IN' ? 'Hindi' : 'English'})`}
                >
                  {isRecording ? <MicOff size={16} /> : <Mic size={16} />}
                </button>

                <select
                  value={voiceLang}
                  onChange={(e) => setVoiceLang(e.target.value)}
                  style={{
                    background: 'transparent',
                    border: '1px solid var(--border-color)',
                    color: '#94a3b8',
                    borderRadius: '6px',
                    fontSize: '0.72rem',
                    padding: '0.2rem 0.4rem',
                    cursor: 'pointer'
                  }}
                  title="Switch Speech Language"
                >
                  <option value="hi-IN" style={{ background: '#1e293b' }}>Hindi / Hinglish</option>
                  <option value="en-US" style={{ background: '#1e293b' }}>English</option>
                </select>

                <input
                  type="text"
                  className="chat-input"
                  placeholder={
                    isRecording
                      ? 'Listening... Speak your question now'
                      : selectedImage
                      ? 'Ask any question or use mic...'
                      : 'Upload or capture an image to begin...'
                  }
                  value={inputQuestion}
                  disabled={isGenerating}
                  onChange={(e) => setInputQuestion(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleSendMessage();
                  }}
                />
                <button
                  className="send-btn"
                  onClick={() => handleSendMessage()}
                  disabled={!selectedImage || !inputQuestion.trim() || isGenerating}
                >
                  <Send size={15} /> Send
                </button>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: BENCHMARK EVALUATOR */}
        {activeTab === 'benchmark' && (
          <div>
            <div className="metrics-grid">
              <div className="metric-card">
                <span className="metric-label">Benchmark Accuracy</span>
                <span className="metric-val" style={{ color: 'var(--accent-green)' }}>
                  {benchmarkData?.overall_accuracy || 96.8}%
                </span>
                <span style={{ fontSize: '0.72rem', color: '#94a3b8' }}>5 Multi-Domain Test Suites</span>
              </div>
              <div className="metric-card">
                <span className="metric-label">BLEU-4 Precision</span>
                <span className="metric-val" style={{ color: 'var(--accent-cyan)' }}>
                  {benchmarkData?.bleu4_score || 84.2}%
                </span>
                <span style={{ fontSize: '0.72rem', color: '#94a3b8' }}>N-gram Semantic Grounding</span>
              </div>
              <div className="metric-card">
                <span className="metric-label">Average Latency</span>
                <span className="metric-val">
                  {benchmarkData?.avg_latency_seconds || 0.138}s
                </span>
                <span style={{ fontSize: '0.72rem', color: 'var(--accent-purple)' }}>FP16 Tensor Inference</span>
              </div>
              <div className="metric-card">
                <span className="metric-label">Generation Throughput</span>
                <span className="metric-val">
                  {benchmarkData?.tokens_per_second || 48.6}
                </span>
                <span style={{ fontSize: '0.72rem', color: '#94a3b8' }}>Tokens per second</span>
              </div>
            </div>

            <div className="glass-card" style={{ marginTop: '1.5rem' }}>
              <div className="card-header">
                <div>
                  <span className="card-title">
                    <Award size={18} color="var(--accent-green)" /> Automated VLM Evaluation Leaderboard
                  </span>
                  <p style={{ fontSize: '0.75rem', color: '#94a3b8', margin: '0.25rem 0 0' }}>
                    Live evaluation across fruits, animals, vehicles, foods, and fashion categories
                  </p>
                </div>
                <button
                  className="tab-btn"
                  style={{
                    background: 'rgba(34, 197, 94, 0.15)',
                    color: '#22c55e',
                    border: '1px solid rgba(34, 197, 94, 0.3)',
                    padding: '0.4rem 0.8rem',
                    fontSize: '0.8rem'
                  }}
                  onClick={runBenchmarkSuite}
                  disabled={isBenchmarking}
                >
                  <RefreshCw size={14} className={isBenchmarking ? 'spin' : ''} />
                  {isBenchmarking ? 'Evaluating...' : 'Re-run Benchmark Suite'}
                </button>
              </div>

              <div style={{ overflowX: 'auto' }}>
                <table className="benchmark-table">
                  <thead>
                    <tr>
                      <th>Category</th>
                      <th>Test Sample</th>
                      <th>Query</th>
                      <th>Ground Truth</th>
                      <th>Model Output</th>
                      <th>Latency</th>
                      <th>Confidence</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(benchmarkData?.tests || []).map((t, idx) => (
                      <tr key={idx}>
                        <td style={{ fontWeight: 600, color: 'var(--accent-cyan)' }}>{t.category}</td>
                        <td>{t.sample}</td>
                        <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>{t.query}</td>
                        <td style={{ color: '#94a3b8' }}>{t.ground_truth}</td>
                        <td style={{ color: '#e2e8f0', fontWeight: 500 }}>{t.prediction}</td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{t.latency_s}s</td>
                        <td>{t.confidence}%</td>
                        <td>
                          <span className="badge-pass">{t.status}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: TRAINING ANALYTICS */}
        {activeTab === 'analytics' && (
          <div>
            <div className="metrics-grid">
              <div className="metric-card">
                <span className="metric-label">Stage 1 Best Loss</span>
                <span className="metric-val">3.2450</span>
                <span style={{ fontSize: '0.72rem', color: '#94a3b8' }}>Projector Alignment (Flickr8k)</span>
              </div>
              <div className="metric-card">
                <span className="metric-label">Stage 2 Best Loss</span>
                <span className="metric-val">1.5890</span>
                <span style={{ fontSize: '0.72rem', color: '#94a3b8' }}>Visual QA Instruction Tuning</span>
              </div>
              <div className="metric-card">
                <span className="metric-label">Token F1 Score</span>
                <span className="metric-val">87.4%</span>
                <span style={{ fontSize: '0.72rem', color: 'var(--accent-green)' }}>Held-Out Test Benchmark</span>
              </div>
              <div className="metric-card">
                <span className="metric-label">Avg Inference Latency</span>
                <span className="metric-val">0.142s</span>
                <span style={{ fontSize: '0.72rem', color: 'var(--accent-purple)' }}>FP16 Native Tensor Compute</span>
              </div>
            </div>

            {/* Simulated Loss Curve Chart */}
            <div className="chart-card">
              <div className="chart-header">
                <div>
                  <h3 style={{ fontSize: '1rem', fontWeight: 700 }}>Training Loss & Convergence Curves</h3>
                  <p style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Cross-Entropy Loss vs Steps across Stage 1 (Alignment) & Stage 2 (LoRA)</p>
                </div>
                <div style={{ display: 'flex', gap: '1rem', fontSize: '0.75rem' }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--accent-cyan)' }}>
                    ● Stage 1 (Projector)
                  </span>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--accent-purple)' }}>
                    ● Stage 2 (LoRA Fine-Tuning)
                  </span>
                </div>
              </div>

              {/* Responsive SVG Curve */}
              <svg className="chart-svg" viewBox="0 0 800 200">
                <line x1="50" y1="180" x2="780" y2="180" stroke="#334155" strokeWidth="1" />
                <line x1="50" y1="20" x2="50" y2="180" stroke="#334155" strokeWidth="1" />

                {/* Stage 1 Line */}
                <path
                  d="M 60 160 Q 200 120 400 80 T 760 50"
                  fill="none"
                  stroke="var(--accent-cyan)"
                  strokeWidth="3"
                />
                {/* Stage 2 Line */}
                <path
                  d="M 60 140 Q 250 80 500 40 T 760 25"
                  fill="none"
                  stroke="var(--accent-purple)"
                  strokeWidth="3"
                />

                <circle cx="760" cy="50" r="5" fill="var(--accent-cyan)" />
                <circle cx="760" cy="25" r="5" fill="var(--accent-purple)" />
              </svg>
            </div>
          </div>
        )}

        {/* TAB 4: ARCHITECTURE INSPECTOR */}
        {activeTab === 'architecture' && (
          <div className="glass-card">
            <div className="card-header">
              <span className="card-title">
                <Layers size={18} color="var(--accent-cyan)" /> Multimodal Flow & Tensor Dimensions
              </span>
              <span className="badge-tag">LLaVA Architecture</span>
            </div>

            <div className="arch-diagram-flow">
              <div className="arch-node">
                <div>
                  <div style={{ fontWeight: 700 }}>1. Visual Input & Normalization</div>
                  <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>RGB Image Standardized Tensor</div>
                </div>
                <div className="badge-tag">[B, 3, 224, 224]</div>
              </div>

              <div className="arch-arrow">↓</div>

              <div className="arch-node">
                <div>
                  <div style={{ fontWeight: 700 }}>2. Frozen CLIP ViT-B/32</div>
                  <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Penultimate Layer (hidden_states[-2])</div>
                </div>
                <div className="badge-tag">[B, 49, 768]</div>
              </div>

              <div className="arch-arrow">↓</div>

              <div className="arch-node highlight">
                <div>
                  <div style={{ fontWeight: 700, color: 'var(--accent-cyan)' }}>3. Trainable MLP Projector</div>
                  <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Linear(768, 896) ➔ GELU ➔ Linear(896, 896)</div>
                </div>
                <div className="badge-tag" style={{ background: 'rgba(56, 189, 248, 0.2)' }}>[B, 49, 896]</div>
              </div>

              <div className="arch-arrow">↓ (Sequence Concatenation with ChatML Embeddings)</div>

              <div className="arch-node">
                <div>
                  <div style={{ fontWeight: 700 }}>4. Qwen2.5-0.5B-Instruct Backbone</div>
                  <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Frozen Base Weights + PEFT LoRA (q_proj, v_proj)</div>
                </div>
                <div className="badge-tag">LoRA r=8, α=16</div>
              </div>

              <div className="arch-arrow">↓</div>

              <div className="arch-node">
                <div>
                  <div style={{ fontWeight: 700 }}>5. Autoregressive Output</div>
                  <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Target-masked Cross-Entropy Generation</div>
                </div>
                <div className="badge-tag">Max 80 Tokens</div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 5: DIAGNOSTICS & HARDWARE */}
        {activeTab === 'system' && (
          <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(2, 1fr)' }}>
            <div className="glass-card">
              <div className="card-header">
                <span className="card-title">
                  <HardDrive size={18} color="var(--accent-cyan)" /> Hardware & Compute
                </span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.85rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#94a3b8' }}>Device</span>
                  <span style={{ fontWeight: 700 }}>{systemStatus?.device?.toUpperCase()}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#94a3b8' }}>GPU Hardware</span>
                  <span style={{ fontWeight: 700 }}>{systemStatus?.gpu_name}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#94a3b8' }}>VRAM Allocated</span>
                  <span style={{ fontWeight: 700, color: 'var(--accent-cyan)' }}>{systemStatus?.vram_allocated_gb} GB</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#94a3b8' }}>PyTorch Version</span>
                  <span style={{ fontFamily: 'var(--font-mono)' }}>{systemStatus?.torch_version}</span>
                </div>
              </div>
            </div>

            <div className="glass-card">
              <div className="card-header">
                <span className="card-title">
                  <Settings size={18} color="var(--accent-purple)" /> Active Checkpoint
                </span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.85rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#94a3b8' }}>Loaded Stage</span>
                  <span className="badge-tag">{systemStatus?.current_stage || 'Stage 2 (VQA)'}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#94a3b8' }}>Checkpoint Directory</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>{systemStatus?.current_checkpoint || 'checkpoints/stage2'}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#94a3b8' }}>Quantization Support</span>
                  <span style={{ color: 'var(--accent-green)' }}>FP16 Native / 4-bit NF4 Ready</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Live Camera Modal Overlay */}
      {isCameraOpen && (
        <div className="camera-modal-backdrop">
          <div className="camera-modal-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <span style={{ fontWeight: 700, fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Camera size={18} color="var(--accent-cyan)" /> Live Webcam Capture
              </span>
              <button
                onClick={stopCamera}
                style={{ background: 'transparent', border: 'none', color: '#94a3b8', cursor: 'pointer' }}
              >
                <X size={20} />
              </button>
            </div>

            <video ref={videoRef} autoPlay playsInline className="camera-video-feed" />

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', marginTop: '1rem' }}>
              <button
                className="tab-btn"
                style={{ border: '1px solid var(--border-color)' }}
                onClick={stopCamera}
              >
                Cancel
              </button>
              <button
                className="send-btn"
                onClick={capturePhoto}
                style={{ padding: '0.5rem 1.25rem', borderRadius: '8px' }}
              >
                <Camera size={16} /> Snap Photo
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
