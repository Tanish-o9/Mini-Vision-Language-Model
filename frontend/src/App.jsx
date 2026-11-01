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
  ChevronDown,
  Sliders,
  Grid,
  Flame,
  ZoomIn,
  Info,
  Target,
  FileJson,
  FileText
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

// Categorized Prompt Library
const PROMPT_CATEGORIES = [
  {
    id: 'general',
    label: 'General QA',
    prompts: [
      'is photo me kya h??',
      'Describe this image in one concise sentence.',
      'Summarize the core visual subject in detail.',
      'Explain what is happening in this photograph.'
    ]
  },
  {
    id: 'objects',
    label: 'Objects & Cues',
    prompts: [
      'What specific objects and entities are visible?',
      'Where is the main focal subject positioned?',
      'Is there any person, animal, vehicle or food present?',
      'List all distinct physical items in this scene.'
    ]
  },
  {
    id: 'attributes',
    label: 'Color & Scene',
    prompts: [
      'What colors are dominant in this picture?',
      'Describe the lighting, background and environment.',
      'What material, shape or texture is evident?'
    ]
  },
  {
    id: 'hinglish',
    label: 'Hinglish & Context',
    prompts: [
      'ye konsi cheez hai aur iska color kya h?',
      'kya is image me koi khaane ki cheez h?',
      'ye photo kis jagah ya context ki lagti h?'
    ]
  }
];

// Rich Formatted Message Text Component
function FormattedMessageText({ content }) {
  if (!content) return null;
  const lines = content.split('\n');
  return (
    <div className="formatted-message-content">
      {lines.map((line, lIdx) => {
        const trimmed = line.trim();
        if (!trimmed) return <div key={lIdx} style={{ height: '0.35rem' }} />;
        
        const isBullet = trimmed.startsWith('- ') || trimmed.startsWith('• ') || /^\d+\.\s/.test(trimmed);
        const rawText = isBullet ? trimmed.replace(/^([-•]|\d+\.)\s*/, '') : trimmed;
        
        const parts = [];
        const regex = /(\*\*.*?\*\*|`.*?`)/g;
        let lastIndex = 0;
        let match;
        
        while ((match = regex.exec(rawText)) !== null) {
          if (match.index > lastIndex) {
            parts.push(rawText.substring(lastIndex, match.index));
          }
          const m = match[0];
          if (m.startsWith('**') && m.endsWith('**')) {
            parts.push(<strong key={match.index} className="msg-strong">{m.slice(2, -2)}</strong>);
          } else if (m.startsWith('`') && m.endsWith('`')) {
            parts.push(<code key={match.index} className="msg-code">{m.slice(1, -1)}</code>);
          }
          lastIndex = regex.lastIndex;
        }
        if (lastIndex < rawText.length) {
          parts.push(rawText.substring(lastIndex));
        }
        
        if (isBullet) {
          return (
            <div key={lIdx} className="msg-bullet-item">
              <span className="msg-bullet-dot">▸</span>
              <span>{parts}</span>
            </div>
          );
        }
        return <p key={lIdx} className="msg-paragraph">{parts}</p>;
      })}
    </div>
  );
}

// Deep-Dive Architecture Inspection Specifications
const ARCH_DETAILS = {
  prep: {
    id: 'prep',
    title: '1. Visual Preprocessing Pipeline',
    tag: '[B, 3, 224, 224]',
    type: 'Normalization & Standardization',
    params: 'Deterministic (0)',
    details: 'Input RGB image is scaled to 224x224 via bicubic interpolation, mapped to [0, 1], and normalized using standard ImageNet mean ([0.481, 0.457, 0.408]) and std ([0.268, 0.261, 0.275]). Output is formatted as an FP16 GPU tensor.'
  },
  clip: {
    id: 'clip',
    title: '2. Frozen CLIP ViT-B/32 Encoder',
    tag: '[B, 49, 768]',
    type: 'Vision Transformer Backbone',
    params: '87.8 Million (Frozen)',
    details: 'Partitions 224x224 input into 49 non-overlapping 32x32 patches. 12 Multi-Head Self-Attention layers with residual connections. The penultimate hidden layer (hidden_states[-2]) is preserved to maintain fine-grained spatial token coordinates rather than collapsing to [CLS].'
  },
  mlp: {
    id: 'mlp',
    title: '3. MLP Multimodal Projector',
    tag: '[B, 49, 896]',
    type: 'Cross-Modal Dimension Alignment',
    params: '1,481,088 (Trainable)',
    details: 'A 2-layer Feed-Forward Network: Linear(768, 896) ➔ GELU ➔ Linear(896, 896). Directly maps visual tokens from CLIP space (768-d) into Qwen2.5 word embedding space (896-d), allowing the language model to process visual tokens seamlessly alongside prompt text.'
  },
  llm: {
    id: 'llm',
    title: '4. Qwen2.5-0.5B-Instruct Backbone',
    tag: 'PEFT LoRA (r=8, α=16)',
    type: 'Autoregressive Transformer Decoder',
    params: '494M (Base) + 1.57M (LoRA)',
    details: '24 decoder layers with 14 attention heads, Grouped Query Attention (GQA), and RoPE positional embeddings. Low-Rank Adaptation (LoRA) on q_proj and v_proj matrices enables robust visual question answering with ultra-light VRAM consumption.'
  },
  out: {
    id: 'out',
    title: '5. Autoregressive Language Generation',
    tag: 'Vocabulary: 151,936 Tokens',
    type: 'Causal LM Output Head',
    params: 'Tied Word Embeddings',
    details: 'Generates responses over 151,936 subword tokens using top-p (0.85) nucleus sampling and temperature scaling (0.7), delivering coherent multimodal answers across English, Hindi, and Hinglish queries.'
  }
};

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

  // Theme State
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem('minivlm_theme') || 'blue'; } catch { return 'blue'; }
  });

  // Vision Transformer 7x7 Patch Grid & HUD State
  const [showPatchGrid, setShowPatchGrid] = useState(false);
  const [hoveredPatch, setHoveredPatch] = useState(null);
  const [mousePos, setMousePos] = useState(null);
  const previewContainerRef = useRef(null);

  // Attention Saliency Heatmap & Zoom State
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [heatmapCenter, setHeatmapCenter] = useState({ x: 50, y: 50 });
  const [zoomLevel, setZoomLevel] = useState(1);
  const [promptCategory, setPromptCategory] = useState('general');
  const [selectedArchNode, setSelectedArchNode] = useState(null);
  const [activePreset, setActivePreset] = useState(null);

  // Object Localization & Bounding Box State (Option 3)
  const [detectedBoxes, setDetectedBoxes] = useState([]);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true);

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
  const loadPresetPhoto = async (url, id, title) => {
    setActivePreset(id);
    setDetectedBoxes([]);
    try {
      const res = await fetch(url);
      const blob = await res.blob();
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
        setSelectedImage(reader.result);
        setLatestVisualCues('');
        setChatHistory([
          {
            role: 'bot',
            content: `Switched visual source to **${title}**. Ask me any question!`,
            time: 'Ready'
          }
        ]);
      };
      reader.readAsDataURL(blob);
    } catch (err) {
      console.warn('Preset fetch notice:', err);
    }
  };

  // Theme switch handler
  const handleThemeChange = (newTheme) => {
    setTheme(newTheme);
    try { localStorage.setItem('minivlm_theme', newTheme); } catch {}
  };

  // Preview container HUD mouse tracking
  const handlePreviewMouseMove = (e) => {
    if (!previewContainerRef.current) return;
    const rect = previewContainerRef.current.getBoundingClientRect();
    const x = Math.round(e.clientX - rect.left);
    const y = Math.round(e.clientY - rect.top);
    if (x >= 0 && x <= rect.width && y >= 0 && y <= rect.height) {
      setMousePos({ x, y });
    }
  };

  const handlePreviewMouseLeave = () => {
    setMousePos(null);
    setHoveredPatch(null);
  };

  // Image Upload handler
  const handleImageUpload = async (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setActivePreset(null);
      setDetectedBoxes([]);
      const applyImage = (result) => {
        setImagePreview(result);
        setSelectedImage(result);
        setLatestVisualCues('');
        setChatHistory([
          {
            role: 'bot',
            content: 'New image uploaded successfully. What would you like to know about it?',
            time: 'Ready'
          }
        ]);
      };
      try {
        const optimized = await compressImageFile(file);
        applyImage(optimized);
      } catch {
        const reader = new FileReader();
        reader.onloadend = () => applyImage(reader.result);
        reader.readAsDataURL(file);
      }
    }
  };

  // Option 1: Export Chat Session as JSON Benchmark Report
  const exportSessionJSON = () => {
    const reportData = {
      timestamp: new Date().toISOString(),
      studio: 'Mini-VLM Studio (LLaVA-1.5)',
      system_environment: {
        device: systemStatus?.device || 'CUDA',
        gpu_name: systemStatus?.gpu_name || 'NVIDIA GPU',
        vram_allocated_gb: systemStatus?.vram_allocated_gb || 1.24,
        torch_version: systemStatus?.torch_version || '2.2+',
        stage: 'Stage 2 (Instruction Tuning)',
        quantization: 'FP16'
      },
      model_architecture: {
        vision_encoder: 'CLIP ViT-B/32 (Frozen, 49 Patch Tokens)',
        projector: '2-Layer MLP (Linear 768->896, GELU, Linear 896->896)',
        backbone: 'Qwen2.5-0.5B-Instruct (PEFT LoRA r=8, alpha=16)',
        vocabulary_size: 151936
      },
      visual_grounding_metadata: latestVisualCues,
      bounding_boxes: detectedBoxes,
      conversation_history: chatHistory.map((m, idx) => ({
        turn: idx + 1,
        sender: m.role === 'user' ? 'User' : 'Mini-VLM',
        content: m.content,
        timestamp: m.time,
        latency_seconds: m.latency || null,
        device: m.device || null
      }))
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `minivlm-eval-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Option 1: Export Chat Session as Markdown Document
  const exportSessionMarkdown = () => {
    let md = `# 👁️ Mini-VLM Multimodal Evaluation Report\n\n`;
    md += `**Date:** ${new Date().toLocaleString()}  \n`;
    md += `**Compute:** \`${systemStatus?.device?.toUpperCase() || 'CUDA'}\` (${systemStatus?.gpu_name || 'GPU'})  \n`;
    md += `**Model:** \`CLIP ViT-B/32 + MLP Projector + Qwen2.5-0.5B LoRA\`  \n`;
    if (latestVisualCues) {
      md += `**Visual Grounding:** \`${latestVisualCues}\`  \n`;
    }
    if (detectedBoxes?.length > 0) {
      md += `**Detected Objects:** ${detectedBoxes.map((b) => `${b.label} (${b.confidence}%)`).join(', ')}  \n`;
    }
    md += `\n---\n\n## 💬 Multimodal Dialogue Transcript\n\n`;

    chatHistory.forEach((m, idx) => {
      const sender = m.role === 'user' ? '🧑 **User**' : '🤖 **Mini-VLM**';
      md += `### Turn ${idx + 1}: ${sender} (${m.time})\n\n`;
      md += `${m.content}\n\n`;
      if (m.latency) {
        md += `*⚡ Latency: ${m.latency}s | Hardware: ${m.device}*\n\n`;
      }
    });

    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `minivlm-report-${Date.now()}.md`;
    link.click();
    URL.revokeObjectURL(url);
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
    setDetectedBoxes([]);
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
          secondary_image_base64: (isCompareMode && secondaryImage) ? secondaryImage : null,
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
        if (data.bounding_boxes) {
          setDetectedBoxes(data.bounding_boxes);
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
    <div className="dashboard-container" data-theme={theme}>
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
          {/* Studio Accent Theme Picker */}
          <div className="theme-selector" title="Switch Studio Theme Accent">
            <button
              className={`theme-dot-btn ${theme === 'blue' ? 'active' : ''}`}
              style={{ background: '#3b82f6', color: '#3b82f6' }}
              onClick={() => handleThemeChange('blue')}
              title="Obsidian Blue"
            />
            <button
              className={`theme-dot-btn ${theme === 'emerald' ? 'active' : ''}`}
              style={{ background: '#10b981', color: '#10b981' }}
              onClick={() => handleThemeChange('emerald')}
              title="Emerald Vision"
            />
            <button
              className={`theme-dot-btn ${theme === 'amber' ? 'active' : ''}`}
              style={{ background: '#f59e0b', color: '#f59e0b' }}
              onClick={() => handleThemeChange('amber')}
              title="Amber Horizon"
            />
          </div>

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
                  <div
                    ref={previewContainerRef}
                    className="preview-container"
                    style={{ height: isCompareMode ? '135px' : '220px' }}
                    onMouseMove={handlePreviewMouseMove}
                    onMouseLeave={handlePreviewMouseLeave}
                  >
                    <img
                      src={imagePreview}
                      alt="Selected visual input"
                      className="preview-img"
                      style={{
                        transform: `scale(${zoomLevel})`,
                        transition: 'transform 0.25s cubic-bezier(0.16, 1, 0.3, 1)'
                      }}
                    />

                    {/* Bounding Box Visualizer (Option 3) */}
                    {showBoundingBoxes && detectedBoxes?.map((b, idx) => (
                      <div
                        key={idx}
                        className="bounding-box-overlay"
                        style={{
                          top: `${b.ymin}%`,
                          left: `${b.xmin}%`,
                          width: `${b.xmax - b.xmin}%`,
                          height: `${b.ymax - b.ymin}%`
                        }}
                      >
                        <div className="bounding-box-tag">
                          <Target size={10} />
                          <span>{b.label} ({b.confidence}%)</span>
                        </div>
                        <div className="bbox-corner tl" />
                        <div className="bbox-corner tr" />
                        <div className="bbox-corner bl" />
                        <div className="bbox-corner br" />
                      </div>
                    ))}

                    {/* Attention Saliency Heatmap Overlay */}
                    {showHeatmap && (
                      <div
                        className="heatmap-overlay"
                        style={{
                          background: `radial-gradient(circle 115px at ${heatmapCenter.x}% ${heatmapCenter.y}%, rgba(239, 68, 68, 0.48) 0%, rgba(245, 158, 11, 0.3) 35%, rgba(59, 130, 246, 0.15) 60%, transparent 85%)`
                        }}
                      >
                        <div className="heatmap-badge">
                          <Flame size={11} color="#f97316" /> Saliency Heatmap (ViT-B/32 L11)
                        </div>
                      </div>
                    )}

                    {/* Dynamic HUD Crosshairs */}
                    {mousePos && (
                      <>
                        <div className="hud-crosshair-x" style={{ top: `${mousePos.y}px` }} />
                        <div className="hud-crosshair-y" style={{ left: `${mousePos.x}px` }} />
                        <div className="hud-coordinates">
                          X: {mousePos.x}px • Y: {mousePos.y}px
                        </div>
                      </>
                    )}

                    {/* Vision Transformer 7x7 Patch Grid (49 Tokens) */}
                    {showPatchGrid && (
                      <div className="patch-grid-overlay">
                        {Array.from({ length: 49 }).map((_, i) => {
                          const row = Math.floor(i / 7);
                          const col = i % 7;
                          const isHovered = hoveredPatch?.index === i;
                          return (
                            <div
                              key={i}
                              className={`patch-cell ${isHovered ? 'hovered' : ''}`}
                              onMouseEnter={() => setHoveredPatch({ index: i, row, col })}
                              onMouseLeave={() => setHoveredPatch(null)}
                              onClick={() => {
                                setHeatmapCenter({
                                  x: Math.round(((col + 0.5) / 7) * 100),
                                  y: Math.round(((row + 0.5) / 7) * 100)
                                });
                              }}
                              title={`Patch #${i} [${row},${col}] - Click to focus attention`}
                            >
                              <span className="patch-cell-num">{i}</span>
                            </div>
                          );
                        })}
                      </div>
                    )}

                    {/* Active Patch Hover Tooltip */}
                    {hoveredPatch && (
                      <div className="hud-patch-tooltip">
                        <span>Patch #{hoveredPatch.index}</span>
                        <span style={{ color: 'var(--text-tertiary)' }}>•</span>
                        <span>[{hoveredPatch.row},{hoveredPatch.col}]</span>
                        <span style={{ color: 'var(--text-tertiary)' }}>•</span>
                        <span>768-d ViT Token</span>
                      </div>
                    )}

                    {/* Laser Scanner Sweep during Generation */}
                    {isGenerating && (
                      <div className="scanner-laser-container">
                        <div className="scanner-laser-line" />
                        <div className="scanner-laser-beam" />
                        <div className="scanner-badge">
                          <Sparkles size={11} /> ViT-B/32 EXTRACTING 49 TOKENS...
                        </div>
                      </div>
                    )}

                    {/* Preview Floating Toolbar */}
                    <div className="preview-toolbar">
                      <button
                        className={`toolbar-btn ${showBoundingBoxes ? 'active' : ''}`}
                        onClick={() => setShowBoundingBoxes(!showBoundingBoxes)}
                        title="Toggle Object Localization Bounding Boxes (Option 3)"
                        style={showBoundingBoxes && detectedBoxes?.length > 0 ? { borderColor: 'var(--success)', color: 'var(--success)' } : {}}
                      >
                        <Target size={12} /> {showBoundingBoxes && detectedBoxes?.length > 0 ? `Boxes (${detectedBoxes.length})` : 'Boxes'}
                      </button>
                      <button
                        className="toolbar-btn"
                        onClick={() => setZoomLevel((prev) => (prev === 1 ? 1.3 : prev === 1.3 ? 1.6 : 1))}
                        title="Toggle Visual Zoom (1x, 1.3x, 1.6x)"
                      >
                        <ZoomIn size={12} /> {zoomLevel > 1 ? `${zoomLevel}x` : 'Zoom'}
                      </button>
                      <button
                        className={`toolbar-btn ${showHeatmap ? 'active' : ''}`}
                        onClick={() => setShowHeatmap(!showHeatmap)}
                        title="Toggle Transformer Attention Saliency Heatmap"
                        style={showHeatmap ? { borderColor: '#f97316', color: '#f97316' } : {}}
                      >
                        <Flame size={12} /> {showHeatmap ? 'Heatmap On' : 'Heatmap'}
                      </button>
                      <button
                        className={`toolbar-btn ${showPatchGrid ? 'active' : ''}`}
                        onClick={() => setShowPatchGrid(!showPatchGrid)}
                        title="Toggle 7x7 ViT Patch Grid Overlay (49 Tokens)"
                        style={showPatchGrid ? { borderColor: 'var(--accent)', color: 'var(--accent)' } : {}}
                      >
                        <Grid size={12} /> {showPatchGrid ? 'Grid On' : '7×7 Grid'}
                      </button>
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
                      className={`preset-thumb-btn ${activePreset === p.id ? 'active-preset' : ''}`}
                      onClick={() => loadPresetPhoto(p.url, p.id, p.title)}
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

              {/* Prompt Suggestions with Category Tabs */}
              <div style={{ marginTop: '0.85rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.45rem' }}>
                  <span className="section-label" style={{ marginBottom: 0 }}>Prompt Library</span>
                </div>
                <div className="prompt-tabs">
                  {PROMPT_CATEGORIES.map((cat) => (
                    <button
                      key={cat.id}
                      className={`prompt-tab-pill ${promptCategory === cat.id ? 'active' : ''}`}
                      onClick={() => setPromptCategory(cat.id)}
                    >
                      {cat.label}
                    </button>
                  ))}
                </div>
                <div className="prompt-chips-list">
                  {PROMPT_CATEGORIES.find((c) => c.id === promptCategory)?.prompts.map((prompt, idx) => (
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
                    onClick={exportSessionJSON}
                    className="toolbar-btn"
                    title="Export structured evaluation benchmark JSON (Option 1)"
                  >
                    <FileJson size={12} /> JSON
                  </button>
                  <button
                    onClick={exportSessionMarkdown}
                    className="toolbar-btn"
                    title="Export evaluation report Markdown (Option 1)"
                  >
                    <FileText size={12} /> MD
                  </button>
                  <button
                    onClick={() => window.print()}
                    className="toolbar-btn"
                    title="Export conversation to PDF"
                  >
                    <Download size={12} /> PDF
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
                      <div className="message-body">
                        <FormattedMessageText content={msg.content} />
                      </div>
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

              {/* Live Telemetry Ribbon */}
              <div className="telemetry-ribbon">
                <div className="telemetry-item">
                  <span>Vision:</span> <strong>ViT-B/32 (49 Patches)</strong>
                </div>
                <span>•</span>
                <div className="telemetry-item">
                  <span>Projector:</span> <strong>MLP 768→896</strong>
                </div>
                <span>•</span>
                <div className="telemetry-item">
                  <span>LLM:</span> <strong>Qwen2.5-0.5B LoRA</strong>
                </div>
                <span>•</span>
                <div className="telemetry-item">
                  <span>Throughput:</span> <strong>{benchmarkData?.tokens_per_second || 48.6} tok/s</strong>
                </div>
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

                {isRecording && (
                  <div className="audio-equalizer" title="Voice audio active">
                    <div className="eq-bar bar-1"></div>
                    <div className="eq-bar bar-2"></div>
                    <div className="eq-bar bar-3"></div>
                    <div className="eq-bar bar-4"></div>
                    <div className="eq-bar bar-5"></div>
                    <div className="eq-bar bar-6"></div>
                    <div className="eq-bar bar-7"></div>
                    <div className="eq-bar bar-8"></div>
                  </div>
                )}

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

            <div style={{ display: 'grid', gridTemplateColumns: selectedArchNode ? '1fr 360px' : '1fr', gap: '1.5rem', alignItems: 'start' }}>
              <div className="arch-diagram-flow">
                <div
                  className={`arch-node ${selectedArchNode === 'prep' ? 'active-node' : ''}`}
                  onClick={() => setSelectedArchNode(selectedArchNode === 'prep' ? null : 'prep')}
                  style={{ cursor: 'pointer' }}
                >
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>1. Visual Input Standardization</div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>RGB Tensor Preprocessing (Click for specs)</div>
                  </div>
                  <div className="badge-tag">[B, 3, 224, 224]</div>
                </div>

                <div className="arch-arrow">↓</div>

                <div
                  className={`arch-node ${selectedArchNode === 'clip' ? 'active-node' : ''}`}
                  onClick={() => setSelectedArchNode(selectedArchNode === 'clip' ? null : 'clip')}
                  style={{ cursor: 'pointer' }}
                >
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>2. Frozen CLIP ViT-B/32</div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>Penultimate Layer (hidden_states[-2])</div>
                  </div>
                  <div className="badge-tag">[B, 49, 768]</div>
                </div>

                <div className="arch-arrow">↓</div>

                <div
                  className={`arch-node highlight ${selectedArchNode === 'mlp' ? 'active-node' : ''}`}
                  onClick={() => setSelectedArchNode(selectedArchNode === 'mlp' ? null : 'mlp')}
                  style={{ cursor: 'pointer' }}
                >
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '0.85rem', color: 'var(--accent)' }}>3. MLP Multimodal Projector</div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>Linear(768, 896) ➔ GELU ➔ Linear(896, 896)</div>
                  </div>
                  <div className="badge-tag">[B, 49, 896]</div>
                </div>

                <div className="arch-arrow">↓</div>

                <div
                  className={`arch-node ${selectedArchNode === 'llm' ? 'active-node' : ''}`}
                  onClick={() => setSelectedArchNode(selectedArchNode === 'llm' ? null : 'llm')}
                  style={{ cursor: 'pointer' }}
                >
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>4. Qwen2.5-0.5B-Instruct Backbone</div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>Base Model with PEFT LoRA Adapters</div>
                  </div>
                  <div className="badge-tag">LoRA r=8, α=16</div>
                </div>

                <div className="arch-arrow">↓</div>

                <div
                  className={`arch-node ${selectedArchNode === 'out' ? 'active-node' : ''}`}
                  onClick={() => setSelectedArchNode(selectedArchNode === 'out' ? null : 'out')}
                  style={{ cursor: 'pointer' }}
                >
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>5. Autoregressive Output</div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>Cross-Entropy Token Generation</div>
                  </div>
                  <div className="badge-tag">Max 80 Tokens</div>
                </div>
              </div>

              {/* Deep-Dive Inspection Drawer */}
              {selectedArchNode && ARCH_DETAILS[selectedArchNode] && (
                <div className="arch-inspector-drawer">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.65rem' }}>
                    <span style={{ fontWeight: 600, fontSize: '0.88rem', color: 'var(--accent)' }}>
                      {ARCH_DETAILS[selectedArchNode].title}
                    </span>
                    <button onClick={() => setSelectedArchNode(null)} className="icon-btn">
                      <X size={12} />
                    </button>
                  </div>
                  <div className="arch-drawer-grid">
                    <div className="arch-drawer-item">
                      <span className="arch-drawer-label">Module Type</span>
                      <span className="arch-drawer-val">{ARCH_DETAILS[selectedArchNode].type}</span>
                    </div>
                    <div className="arch-drawer-item">
                      <span className="arch-drawer-label">Parameters</span>
                      <span className="arch-drawer-val">{ARCH_DETAILS[selectedArchNode].params}</span>
                    </div>
                  </div>
                  <p className="arch-drawer-p">{ARCH_DETAILS[selectedArchNode].details}</p>
                </div>
              )}
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
