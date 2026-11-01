export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();

  return res.status(200).json({
    stage1: {
      train_loss: 0.384,
      val_loss: 0.412,
      alignment_score: 92.4
    },
    stage2: {
      train_loss: 0.245,
      val_loss: 0.288,
      vqa_accuracy: 94.6,
      bleu4: 38.5,
      meteor: 41.2
    }
  });
}
