# Hindi_Hate_Speech_Detection
# Data-Driven Hate Speech Detection in Hindi 🇮🇳

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30%2B-purple.svg)](https://huggingface.co/)
[![IndicBERT](https://img.shields.io/badge/IndicBERTv2-green.svg)](https://huggingface.co/ai4bharat/indic-bert)

**State-of-the-art Hindi hate speech detection using IndicBERTv2 + LoRA achieving 92.85% F1-score on multi-label classification (hate, vulgarity, defamation, violence)**

Detects hostile content in Hindi and Hinglish social media posts with multilingual transformer architecture optimized for Indic languages.

## 🎯 Features

- **Multi-label Classification**: Hate, Vulgarity, Defamation, Violence, Non-Hateful
- **SOTA Performance**: **92.85% Micro F1** (validation) with IndicBERT + LoRA
- **Parameter Efficient**: Only **0.107% parameters** fine-tuned via LoRA
- **Code-mixed Support**: Handles Hindi-English mixing naturally
- **Balanced Dataset**: 5-class dataset with controlled augmentation
- **Production Ready**: Easy inference pipeline for social media moderation

## 📊 Performance Highlights

| Model | F1 Micro (Val) | Parameters Tuned | Dataset |
|-------|----------------|------------------|---------|
| **IndicBERT + LoRA** | **92.85%** | **0.107%** | Hindi Multi-label |
| XLM-RoBERTa | 77.00% | 100% | Hindi Multi-label |
| mBERT | 70.00% | 100% | Hindi |
| SVM (TF-IDF) | 68.00% | - | Hindi |

## 🛠 Quick Start

### 1. Clone & Install
git clone https://github.com/yourusername/hindi-hate-speech-detection.git
cd hindi-hate-speech-detection
pip install -r requirements.txt

text

### 2. Download Dataset
Auto-downloads during training
python src/data/download_datasets.py

text

### 3. Train Model
Train IndicBERT + LoRA (Recommended)
python src/train.py --model indicbert-lora --epochs 5

Or other baselines
python src/train.py --model xlm-roberta --epochs 5

text

### 4. Inference
python src/inference.py --text "यह बहुत गंदी भाषा है" --model_path models/indicbert-lora-best

text


text

## 🔧 Installation

Create conda environment
conda create -n hindi-hate python=3.10
conda activate hindi-hate

Install PyTorch (CPU/GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Install requirements
pip install -r requirements.txt

text

## 📈 Training Your Own Model

Full training pipeline
python src/train.py
--model indicbert-lora
--batch_size 16
--lr 2e-4
--epochs 5
--augment
--save_path models/indicbert-lora

text

**Hyperparameters** (best config):
- Learning Rate: `2e-4`
- Batch Size: `16`
- Epochs: `5`
- LoRA Rank: `8`
- Max Length: `128`

## 🧪 Example Usage

from src.inference import HateSpeechDetector

Load trained model
detector = HateSpeechDetector.from_pretrained("models/indicbert-lora-best")

Single prediction
text = "यह बहुत गंदी भाषा है भाई"
predictions = detector.predict(text)
print(predictions)

{'hate': 0.92, 'vulgarity': 0.87, 'defamation': 0.12, 'violence': 0.05}
Batch prediction
texts = ["text1", "text2", "text3"]
batch_preds = detector.predict_batch(texts)

text

## 🗄 Dataset Details

**Sources**:
1. [CodaLab Hindi Hostile Detection](https://competitions.codalab.org/competitions/26654)
2. [Victor Knox Hindi Hate Speech](https://github.com/victorknox/Hate-Speech-Detection-in-Hindi)

**Classes**: Hate, Vulgarity, Defamation, Violence, Non-Hateful (Multi-label)
**Size**: ~15K samples post-augmentation
**Languages**: Hindi + Hinglish code-mixed

## ⚙️ Data Augmentation

Hindi-specific transformations preserving semantics:
- Lexical insertions: "बहुत", "काफी", "यार"
- Structural variations: "अरे", "सुनो", "देखो"
- Punctuation noise: "!!", "?!", "..."

## 📊 Evaluation Metrics

F1 Micro: 92.85% (Validation)
F1 Macro: 91.23%
Precision: 93.12%
Recall: 92.67%
Exact Match: 88.45%

text

**Multi-label Metrics**:
TP, TN, FP, FN based evaluation
BCEWithLogitsLoss optimization
Threshold: 0.5 (optimized)

text

## 🔗 Dependencies

| Package | Version |
|---------|---------|
| torch | >=2.0.0 |
| transformers | >=4.30.0 |
| datasets | >=2.14.0 |
| peft | >=0.5.0 |
| scikit-learn | >=1.3.0 |
| pandas | >=2.0.0 |
| numpy | >=1.24.0 |
| indic-nlp-library | >=0.9.0 |

Complete `requirements.txt` provided.

## 🤗 Model Hub

Trained models will be uploaded to [Hugging Face](https://huggingface.co/):

ai4bharat/indicbert-hindi-hate-lora

text

## 📸 Results Visualization

![Model Comparison](results/model_comparison.png)
![Confusion Matrix](results/confusion_matrix.png)
![ROC Curves](results/roc_curves.png)

## 🛡️ Ethical Considerations

- **Bias Mitigation**: Balanced dataset across classes
- **False Positives**: Optimized threshold (0.5)
- **Transparency**: Model explanations via attention weights
- **Fairness**: Cross-dataset validation

## 🔭 Future Work

- [ ] Real-time inference API (FastAPI)
- [ ] Mobile deployment (ONNX/TFLite)
- [ ] Multimodal (text + image) detection
- [ ] Active learning pipeline
- [ ] More Indic languages support

## 📝 Citation

@article{mehra2025,
title={Data-Driven Hate Speech Detection in Hindi Language},
author={Mehra, Deepanshu and Das, Ankit and Brar, Aaryan},
year={2025},
note={92.85% F1 IndicBERT+LoRA}
}

text

## 🤝 Contributing

1. Fork the repo
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

[MIT License](LICENSE) - Free for research & commercial use.

## 👥 Authors

- **Deepanshu Mehra** - [LinkedIn](https://www.linkedin.com/in/deepanshu-mehra-17302824a/)
- **Ankit Das** - [GitHub](https://github.com/Ankyytt)
- **Aaryan Brar** - [LinkedIn](https://www.linkedin.com/in/aaryan-brar-758a332bb/)

## 💬 Contact

Questions? Open an [Issue](https://github.com/yourusername/hindi-hate-speech-detection/issues) or reach us at `contact@team.com`

---

⭐ **Star this repo if you found it useful!** ⭐
