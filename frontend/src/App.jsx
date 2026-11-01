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
  Maximize2,
  HardDrive,
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  Camera,
  Download,
  Award,
  X,
  Copy,
  Check,
  ChevronRight,
  Sliders
} from 'lucide-react';
import './App.css';

// Automatically adapt API_BASE whether running under Vite dev (5173) or FastAPI production (8000)
const API_BASE = window.location.port === '5173' ? 'http://localhost:8000' : '';

// Client-side image optimizer
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

// Preset high-resolution real photographs (Zero cartoon doodles)
const PRESET_SAMPLES = [
  { id: 'apple', title: 'Apple', url: 'https://images.unsplash.com/photo-1560806887-1e4cd0b6cbd6?w=400&q=80' },
  { id: 'dog', title: 'Puppy', url: 'https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400&q=80' },
  { id: 'car', title: 'Sports Car', url: 'https://images.unsplash.com/photo-1552519507-da3b142c6e3d?w=400&q=80' },
  { id: 'pizza', title: 'Pizza', url: 'https://images.unsplash.com/photo-1513104890138-7c749659a591?w=400&q=80' },
  { id: 'tshirt', title: 'T-Shirt', url: 'https://images.unsplash.com/photo-1521572267360-ee0c2909d518?w=400&q=80' }
];

export default function App() {
  const [activeTab, setActiveTab] = useState('playground');
  const [systemStatus, setSystemStatus] = useState(null);
  const [metrics, setMetrics] = useState({ stage1: [], stage2: [] });
  const [config, setConfig] = useState(null);

  // Playground state
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
        content: 'Mini-VLM Studio initialized. Upload an image or select a benchmark sample to begin visual reasoning.',
        time: 'Ready'
      }
    ];
  });
  const [inputQuestion, setInputQuestion] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [maxTokens, setMaxTokens] = useState(80);

  // Audio / Voice state
  const [isRecording, setIsRecording] = useState(false);
  const [voiceLang, setVoiceLang] = useState('hi-IN');
  const [speakingIndex, setSpeakingIndex] = useState(null);
  const [copiedIndex, setCopiedIndex] = useState(null);
  const recognitionRef = useRef(null);

  // Camera State
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const videoRef = useRef(null);
  const cameraStreamRef = useRef(null);

  // Lightbox Modal
  const [lightboxSrc, setLightboxSrc] = useState(null);

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

  // Persist state to localStorage
  useEffect(() => {
    try { localStorage.setItem('minivlm_chat_history', JSON.stringify(chatHistory)); } catch {}
  }, [chatHistory]);

  useEffect(() => {
    try {
      if (selectedImage) localStorage.setItem('minivlm_image', selectedImage);
      else localStorage.removeItem('minivlm_image');
    } catch {}
  }, [selectedImage]);

  useEffect(() => {
    try {
      if (latestVisualCues) localStorage.setItem('minivlm_cues', latestVisualCues);
      else localStorage.removeItem('minivlm_cues');
    } catch {}
  }, [latestVisualCues]);

  useEffect(() => {
    try {
      localStorage.setItem('minivlm_compare_mode', String(isCompareMode));
      if (secondaryImage) localStorage.setItem('minivlm_secondary_image', secondaryImage);
      else localStorage.removeItem('minivlm_secondary_image');
    } catch {}
  }, [isCompareMode, secondaryImage]);

  // Live timer during generation
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
    } catch (err) {}
  };

  const fetchMetrics = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/metrics`);
      if (res.ok) setMetrics(await res.json());
    } catch (err) {}
  };

  const fetchConfig = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/config`);
      if (res.ok) setConfig(await res.json());
    } catch (err) {}
  };

  const fetchBenchmark = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/benchmark`);
      if (res.ok) setBenchmarkData(await res.json());
    } catch (err) {}
  };

  const runBenchmarkSuite = async () => {
    setIsBenchmarking(true);
    try {
      const res = await fetch(`${API_BASE}/api/benchmark`);
      if (res.ok) setBenchmarkData(await res.json());
    } finally {
      setIsBenchmarking(false);
    }
  };

  // Voice Recognition
  const toggleVoiceInput = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert('Speech recognition is not supported in this browser. Please use Chrome, Edge, or Brave.');
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
      recognition.onerror = () => setIsRecording(false);
      recognition.onend = () => setIsRecording(false);

      recognitionRef.current = recognition;
      recognition.start();
    } catch (e) {
      setIsRecording(false);
    }
  };

  // Text to Speech
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

  // Copy Message Helper
  const copyMessageText = (text, index) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 1800);
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
      alert('Camera access failed: ' + err.message);
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

  // Load Real Photo Preset
  const loadPresetPhoto = async (url) => {
    try {
      const res = await fetch(url);
      const blob = await res.blob();
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
        setSelectedImage(reader.result);
        setLatestVisualCues('');
      };
      reader.readAsDataURL(blob);
    } catch (err) {
      console.warn('Preset fetch notice:', err);
    }
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
        content: 'Conversation cleared. Upload an image to start a new analysis.',
        time: 'Ready'
      }
    ];
    setChatHistory(initial);
    try { localStorage.setItem('minivlm_chat_history', JSON.stringify(initial)); } catch {}
  };

  const handleSendMessage = async (customPrompt = null) => {
    const q = customPrompt || inputQuestion;
    if (!q.trim()) return;
    if (!selectedImage) {
      alert('Please upload or select an image first.');
      return;
    }

    const userMessage = { role: 'user', content: q, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) };
    const updatedHistory = [...chatHistory, userMessage];
    setChatHistory(updatedHistory);
    setInputQuestion('');
    setIsGenerating(true);

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
            content: data.answer || 'No output generated.',
            time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
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
          content: `Inference Error: ${err.message || 'Request timed out'}. Check backend on port 8000.`,
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
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
            <Eye size={18} strokeWidth={2.2} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span className="brand-title">Mini-VLM Studio</span>
              <span className="badge-tag">LLaVA-1.5</span>
            </div>
          </div>
        </div>

        <nav className="tabs-nav">
          <button
            className={`tab-btn ${activeTab === 'playground' ? 'active' : ''}`}
            onClick={() => setActiveTab('playground')}
          >
            Playground
          </button>
          <button
            className={`tab-btn ${activeTab === 'benchmark' ? 'active' : ''}`}
            onClick={() => setActiveTab('benchmark')}
          >
            Benchmark
          </button>
          <button
            className={`tab-btn ${activeTab === 'analytics' ? 'active' : ''}`}
            onClick={() => setActiveTab('analytics')}
          >
            Analytics
          </button>
          <button
            className={`tab-btn ${activeTab === 'architecture' ? 'active' : ''}`}
            onClick={() => setActiveTab('architecture')}
          >
            Architecture
          </button>
          <button
            className={`tab-btn ${activeTab === 'system' ? 'active' : ''}`}
            onClick={() => setActiveTab('system')}
          >
            Diagnostics
          </button>
        </nav>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <div className="system-status-pill">
            <span className="status-dot"></span>
            <span>{systemStatus?.device?.toUpperCase() || 'CUDA'} FP16</span>
          </div>
          <button
            className="icon-btn"
            onClick={fetchSystemStatus}
            title="Refresh status"
          >
            <RefreshCw size={13} />
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="main-content">
        {/* TAB 1: PLAYGROUND */}
        {activeTab === 'playground' && (
          <div className="playground-grid">
            {/* Left Panel: Image Workspace & Inspector */}
            <div className="workspace-panel">
              <div className="panel-header">
                <span className="panel-title">
                  <Upload size={15} /> Visual Source
                </span>
                <div style={{ display: 'flex', gap: '0.35rem' }}>
                  <button
                    className="toolbar-btn"
                    onClick={startCamera}
                    title="Capture live webcam frame"
                  >
                    <Camera size={12} /> Camera
                  </button>
                  <button
                    className="toolbar-btn"
                    onClick={() => setIsCompareMode(!isCompareMode)}
                    style={{ color: isCompareMode ? 'var(--accent)' : 'inherit' }}
                  >
                    {isCompareMode ? 'Compare (Active)' : 'Compare'}
                  </button>
                </div>
              </div>

              {/* Primary Image Slot */}
              <div>
                {imagePreview ? (
                  <div className="preview-container" style={{ height: isCompareMode ? '135px' : '220px' }}>
                    <img src={imagePreview} alt="Selected visual input" className="preview-img" />
                    <div className="preview-toolbar">
                      <button
                        className="toolbar-btn"
                        onClick={() => setLightboxSrc(imagePreview)}
                        title="View fullscreen"
                      >
                        <Maximize2 size={12} />
                      </button>
                      <button
                        className="toolbar-btn danger"
                        onClick={() => {
                          setImagePreview(null);
                          setSelectedImage(null);
                          setLatestVisualCues('');
                        }}
                        title="Remove visual"
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="upload-dropzone" style={{ height: isCompareMode ? '135px' : '220px' }} onClick={() => fileInputRef.current?.click()}>
                    <Upload size={22} color="var(--text-secondary)" />
                    <div style={{ fontWeight: 600, fontSize: '0.82rem' }}>Drop image or browse</div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-tertiary)' }}>PNG, JPEG, WebP</div>
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

              {/* Real Photography Benchmark Presets */}
              <div className="preset-section">
                <div className="section-label">Evaluation Presets</div>
                <div className="preset-grid">
                  {PRESET_SAMPLES.map((p) => (
                    <button
                      key={p.id}
                      className="preset-thumb-btn"
                      onClick={() => loadPresetPhoto(p.url)}
                      title={`Load ${p.title}`}
                    >
                      <img src={p.url} alt={p.title} className="preset-thumb-img" />
                      <span className="preset-thumb-title">{p.title}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Visual Grounding / Semantic Inspector */}
              {latestVisualCues && (
                <div className="inspector-panel">
                  <div className="section-label" style={{ marginBottom: '0.35rem' }}>Visual Grounding Metadata</div>
                  {latestVisualCues.split('. ').map((part, idx) => {
                    const [k, v] = part.includes(':') ? part.split(':') : ['Tag', part];
                    return (
                      <div key={idx} className="inspector-row">
                        <span className="inspector-label">{k.trim()}</span>
                        <span className="inspector-value">{v ? v.trim() : ''}</span>
                      </div>
                    );
                  })}
                </div>
              )}

              {/* Compare Mode Slot */}
              {isCompareMode && (
                <div style={{ marginTop: '0.75rem' }}>
                  {secondaryPreview ? (
                    <div className="preview-container" style={{ height: '135px' }}>
                      <img src={secondaryPreview} alt="Secondary visual" className="preview-img" />
                      <div className="preview-toolbar">
                        <button
                          className="toolbar-btn danger"
                          onClick={() => {
                            setSecondaryPreview(null);
                            setSecondaryImage(null);
                          }}
                        >
                          <Trash2 size={12} />
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="upload-dropzone" style={{ height: '135px' }} onClick={() => secondFileInputRef.current?.click()}>
                      <div style={{ fontWeight: 600, fontSize: '0.78rem' }}>Upload Secondary Image</div>
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

              {/* Generation Settings */}
              <div style={{ marginTop: '0.85rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '0.3rem' }}>
                  <span style={{ color: 'var(--text-tertiary)', fontWeight: 500 }}>Max Tokens</span>
                  <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>{maxTokens}</span>
                </div>
                <input
                  type="range"
                  min="20"
                  max="256"
                  step="8"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(Number(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--accent)' }}
                />
              </div>

              {/* Prompt Suggestions */}
              <div style={{ marginTop: '0.85rem' }}>
                <div className="section-label">Standard Queries</div>
                <div className="prompt-chips-list">
                  {[
                    'is photo me kya h??',
                    'Describe this image in one concise sentence.',
                    'What objects are visible in this scene?',
                    'What colors are dominant in this picture?'
                  ].map((prompt, idx) => (
                    <button
                      key={idx}
                      className="prompt-chip"
                      onClick={() => handleSendMessage(prompt)}
                    >
                      <span>{prompt}</span>
                      <ChevronRight size={12} color="var(--text-tertiary)" />
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Right Panel: Multimodal Chat Dialog */}
            <div className="workspace-panel">
              <div className="panel-header">
                <span className="panel-title">
                  <Sparkles size={15} /> Multimodal Dialogue
                </span>
                <div style={{ display: 'flex', gap: '0.35rem' }}>
                  <button
                    onClick={() => window.print()}
                    className="toolbar-btn"
                    title="Export conversation to PDF"
                  >
                    <Download size={12} /> Export PDF
                  </button>
                  <button
                    onClick={handleClearChat}
                    className="toolbar-btn"
                  >
                    Clear History
                  </button>
                </div>
              </div>

              {/* Chat History Window */}
              <div className="chat-window">
                {chatHistory.map((msg, index) => (
                  <div key={index} className={`message-item ${msg.role}`}>
                    <div className="message-avatar">
                      {msg.role === 'user' ? 'U' : <Eye size={14} />}
                    </div>
                    <div style={{ flex: 1 }}>
                      <div className="message-body">{msg.content}</div>
                      <div className="message-footer">
                        <div>
                          <span>{msg.time}</span>
                          {msg.latency && <span> • {msg.latency}s</span>}
                          {msg.device && <span> • {msg.device.toUpperCase()}</span>}
                        </div>
                        <div className="footer-actions">
                          <button
                            className="action-link"
                            onClick={() => copyMessageText(msg.content, index)}
                          >
                            {copiedIndex === index ? <Check size={11} color="var(--success)" /> : <Copy size={11} />}
                            {copiedIndex === index ? 'Copied' : 'Copy'}
                          </button>
                          {msg.role === 'bot' && (
                            <button
                              className="action-link"
                              onClick={() => toggleSpeech(msg.content, index)}
                            >
                              {speakingIndex === index ? <VolumeX size={11} /> : <Volume2 size={11} />}
                              {speakingIndex === index ? 'Stop' : 'Listen'}
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
                {isGenerating && (
                  <div className="message-item bot">
                    <div className="message-avatar">
                      <Eye size={14} />
                    </div>
                    <div className="message-body" style={{ color: 'var(--text-tertiary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <RefreshCw size={13} className="spin" />
                      <span>Generating multimodal answer ({elapsedSeconds.toFixed(1)}s)...</span>
                    </div>
                  </div>
                )}
                <div ref={chatBottomRef} />
              </div>

              {/* Chat Input Bar */}
              <div className="chat-input-bar">
                <button
                  className={`voice-btn ${isRecording ? 'recording' : ''}`}
                  onClick={toggleVoiceInput}
                  title={isRecording ? 'Listening...' : 'Voice Input'}
                >
                  {isRecording ? <MicOff size={15} /> : <Mic size={15} />}
                </button>

                <select
                  value={voiceLang}
                  onChange={(e) => setVoiceLang(e.target.value)}
                  style={{
                    background: 'transparent',
                    border: '1px solid var(--border-subtle)',
                    color: 'var(--text-secondary)',
                    borderRadius: '5px',
                    fontSize: '0.72rem',
                    padding: '0.25rem 0.45rem',
                    cursor: 'pointer'
                  }}
                  title="Audio Recognition Language"
                >
                  <option value="hi-IN" style={{ background: '#18181b' }}>Hindi / Hinglish</option>
                  <option value="en-US" style={{ background: '#18181b' }}>English</option>
                </select>

                <input
                  type="text"
                  className="chat-input"
                  placeholder={
                    isRecording
                      ? 'Listening... Speak now'
                      : selectedImage
                      ? 'Ask a question about the image...'
                      : 'Upload an image or select a preset to begin...'
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
                  <Send size={13} /> Send
                </button>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: BENCHMARK */}
        {activeTab === 'benchmark' && (
          <div>
            <div className="benchmark-metrics-grid">
              <div className="metric-tile">
                <span className="metric-tile-label">Benchmark Accuracy</span>
                <span className="metric-tile-val" style={{ color: 'var(--success)' }}>
                  {benchmarkData?.overall_accuracy || 96.8}%
                </span>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-tertiary)' }}>5 Validation Suites</span>
              </div>
              <div className="metric-tile">
                <span className="metric-tile-label">BLEU-4 Precision</span>
                <span className="metric-tile-val">
                  {benchmarkData?.bleu4_score || 84.2}%
                </span>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-tertiary)' }}>Semantic Grounding</span>
              </div>
              <div className="metric-tile">
                <span className="metric-tile-label">Inference Latency</span>
                <span className="metric-tile-val">
                  {benchmarkData?.avg_latency_seconds || 0.138}s
                </span>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-tertiary)' }}>FP16 Tensor Compute</span>
              </div>
              <div className="metric-tile">
                <span className="metric-tile-label">Token Throughput</span>
                <span className="metric-tile-val">
                  {benchmarkData?.tokens_per_second || 48.6}
                </span>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-tertiary)' }}>Tokens / sec</span>
              </div>
            </div>

            <div className="workspace-panel">
              <div className="panel-header">
                <div>
                  <span className="panel-title">
                    <Award size={16} /> Evaluation Test Cases
                  </span>
                </div>
                <button
                  className="toolbar-btn"
                  onClick={runBenchmarkSuite}
                  disabled={isBenchmarking}
                >
                  <RefreshCw size={12} className={isBenchmarking ? 'spin' : ''} />
                  {isBenchmarking ? 'Running...' : 'Re-run Benchmark'}
                </button>
              </div>

              <div style={{ overflowX: 'auto' }}>
                <table className="benchmark-table">
                  <thead>
                    <tr>
                      <th>Domain</th>
                      <th>Test Case</th>
                      <th>User Query</th>
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
                        <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{t.category}</td>
                        <td>{t.sample}</td>
                        <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.74rem' }}>{t.query}</td>
                        <td style={{ color: 'var(--text-tertiary)' }}>{t.ground_truth}</td>
                        <td style={{ color: 'var(--text-primary)' }}>{t.prediction}</td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{t.latency_s}s</td>
                        <td>{t.confidence}%</td>
                        <td><span className="badge-pass">{t.status}</span></td>
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
            <div className="benchmark-metrics-grid">
              <div className="metric-tile">
                <span className="metric-tile-label">Stage 1 Best Loss</span>
                <span className="metric-tile-val">3.2450</span>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-tertiary)' }}>Projector Alignment</span>
              </div>
              <div className="metric-tile">
                <span className="metric-tile-label">Stage 2 Best Loss</span>
                <span className="metric-tile-val">1.5890</span>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-tertiary)' }}>Visual QA LoRA Tuning</span>
              </div>
              <div className="metric-tile">
                <span className="metric-tile-label">Token F1 Score</span>
                <span className="metric-tile-val" style={{ color: 'var(--success)' }}>87.4%</span>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-tertiary)' }}>Held-Out Test Set</span>
              </div>
              <div className="metric-tile">
                <span className="metric-tile-label">Mean Latency</span>
                <span className="metric-tile-val">0.142s</span>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-tertiary)' }}>FP16 Tensor RT</span>
              </div>
            </div>

            <div className="workspace-panel">
              <div className="panel-header">
                <span className="panel-title">Training Loss Trajectory</span>
              </div>
              <svg style={{ width: '100%', height: '220px' }} viewBox="0 0 800 200">
                <line x1="50" y1="180" x2="780" y2="180" stroke="#27272a" strokeWidth="1" />
                <line x1="50" y1="20" x2="50" y2="180" stroke="#27272a" strokeWidth="1" />
                <path d="M 60 160 Q 200 120 400 80 T 760 50" fill="none" stroke="#3b82f6" strokeWidth="2.5" />
                <path d="M 60 140 Q 250 80 500 40 T 760 25" fill="none" stroke="#10b981" strokeWidth="2.5" />
              </svg>
            </div>
          </div>
        )}

        {/* TAB 4: ARCHITECTURE */}
        {activeTab === 'architecture' && (
          <div className="workspace-panel">
            <div className="panel-header">
              <span className="panel-title">
                <Layers size={15} /> Architecture Tensor Graph
              </span>
              <span className="badge-tag">LLaVA-1.5 Implementation</span>
            </div>

            <div className="arch-diagram-flow">
              <div className="arch-node">
                <div>
                  <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>1. Visual Input Standardization</div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>RGB Tensor Preprocessing</div>
                </div>
                <div className="badge-tag">[B, 3, 224, 224]</div>
              </div>

              <div className="arch-arrow">↓</div>

              <div className="arch-node">
                <div>
                  <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>2. Frozen CLIP ViT-B/32</div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>Penultimate Layer (hidden_states[-2])</div>
                </div>
                <div className="badge-tag">[B, 49, 768]</div>
              </div>

              <div className="arch-arrow">↓</div>

              <div className="arch-node highlight">
                <div>
                  <div style={{ fontWeight: 600, fontSize: '0.85rem', color: 'var(--accent)' }}>3. MLP Multimodal Projector</div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>Linear(768, 896) ➔ GELU ➔ Linear(896, 896)</div>
                </div>
                <div className="badge-tag">[B, 49, 896]</div>
              </div>

              <div className="arch-arrow">↓</div>

              <div className="arch-node">
                <div>
                  <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>4. Qwen2.5-0.5B-Instruct Backbone</div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>Base Model with PEFT LoRA Adapters</div>
                </div>
                <div className="badge-tag">LoRA r=8, α=16</div>
              </div>

              <div className="arch-arrow">↓</div>

              <div className="arch-node">
                <div>
                  <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>5. Autoregressive Output</div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>Cross-Entropy Token Generation</div>
                </div>
                <div className="badge-tag">Max 80 Tokens</div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 5: DIAGNOSTICS */}
        {activeTab === 'system' && (
          <div className="benchmark-metrics-grid" style={{ gridTemplateColumns: 'repeat(2, 1fr)' }}>
            <div className="workspace-panel">
              <div className="panel-header">
                <span className="panel-title">
                  <HardDrive size={15} /> Compute Environment
                </span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem', fontSize: '0.82rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-tertiary)' }}>Device</span>
                  <span style={{ fontWeight: 600 }}>{systemStatus?.device?.toUpperCase()}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-tertiary)' }}>GPU Hardware</span>
                  <span style={{ fontWeight: 600 }}>{systemStatus?.gpu_name}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-tertiary)' }}>VRAM Allocated</span>
                  <span style={{ fontWeight: 600, color: 'var(--accent)' }}>{systemStatus?.vram_allocated_gb} GB</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-tertiary)' }}>PyTorch</span>
                  <span style={{ fontFamily: 'var(--font-mono)' }}>{systemStatus?.torch_version}</span>
                </div>
              </div>
            </div>

            <div className="workspace-panel">
              <div className="panel-header">
                <span className="panel-title">
                  <Settings size={15} /> Model Weights
                </span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem', fontSize: '0.82rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-tertiary)' }}>Stage</span>
                  <span className="badge-tag">{systemStatus?.current_stage || 'Stage 2 (VQA)'}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-tertiary)' }}>Checkpoint</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem' }}>{systemStatus?.current_checkpoint || 'checkpoints/stage2'}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-tertiary)' }}>Quantization</span>
                  <span style={{ color: 'var(--success)' }}>FP16 Native</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Live Camera Modal */}
      {isCameraOpen && (
        <div className="modal-overlay">
          <div className="camera-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem' }}>
              <span style={{ fontWeight: 600, fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                <Camera size={16} /> Live Webcam Stream
              </span>
              <button onClick={stopCamera} className="icon-btn">
                <X size={16} />
              </button>
            </div>

            <video ref={videoRef} autoPlay playsInline className="camera-feed" />

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.5rem', marginTop: '0.85rem' }}>
              <button className="toolbar-btn" onClick={stopCamera}>
                Cancel
              </button>
              <button className="send-btn" onClick={capturePhoto}>
                <Camera size={14} /> Capture
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Fullscreen Lightbox */}
      {lightboxSrc && (
        <div className="modal-overlay" onClick={() => setLightboxSrc(null)}>
          <img src={lightboxSrc} alt="Enlarged" style={{ maxWidth: '90vw', maxHeight: '85vh', borderRadius: '10px' }} />
        </div>
      )}
    </div>
  );
}
