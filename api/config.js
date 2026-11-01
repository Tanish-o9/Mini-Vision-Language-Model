export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();

  return res.status(200).json({
    model: {
      vision_encoder: 'openai/clip-vit-base-patch32',
      projector: 'MLP (768 -> 896 -> 896)',
      llm_backbone: 'Qwen/Qwen2.5-0.5B-Instruct',
      peft_lora_rank: 8
    },
    training: {
      stage1_lr: 0.001,
      stage2_lr: 0.0002,
      quantization: 'fp16'
    },
    paths: {
      checkpoint: 'checkpoints/stage2',
      data: 'data/vqa'
    }
  });
}
