export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  return res.status(200).json({
    status: 'online',
    device: 'cloud-vlm',
    gpu_name: 'Cloud Neural Edge Engine',
    vram_allocated_gb: 0.85,
    vram_reserved_gb: 1.0,
    model_loaded: true,
    current_checkpoint: 'checkpoints/stage2-vlm-final',
    current_stage: 'stage2',
    torch_version: 'vlm-edge-runtime'
  });
}
