```markdown
# 🔎 Verity Lens — AI-Powered Fake News Detector

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow%2FKeras-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP%20Pipeline-154f3c?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Data%20Processing-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

**Verity Lens** is an end-to-end Deep Learning and Natural Language Processing (NLP) web application designed to detect misinformation and classify English news articles as **Real** or **Fake** with confidence scoring. Built on a hybrid **Separable CNN + Multi-Head Attention + Bidirectional LSTM (Bi-LSTM)** architecture and deployed via **Streamlit**, the system analyzes linguistic patterns, semantic relationships, and bidirectional sentence context rather than relying on superficial source tags.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture & Workflow](#-system-architecture--workflow)
- [NLP Preprocessing Pipeline](#-nlp-preprocessing-pipeline)
- [Deep Learning Model Architecture](#-deep-learning-model-architecture)
- [Data Leakage & Bias Mitigation](#-data-leakage--bias-mitigation)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Troubleshooting](#-troubleshooting)
- [Future Roadmap](#-future-roadmap)
- [Project Timeline & Author](#-project-timeline--author)

---

## 🌟 Overview

With the rapid spread of digital misinformation, automated fake news detection requires models capable of capturing both local phrase structures and long-range contextual dependencies. **Verity Lens** solves this by combining:

1. **Rigorous Text Normalization:** Cleaning raw articles using NLTK (punctuation stripping, stop-word filtering, and WordNet lemmatization).
2. **Bias-Free Training:** Eliminating dataset artifacts (such as news agency watermarks) so the neural network learns genuine semantic cues.
3. **Hybrid Deep Neural Architecture:** Combining dilated `SeparableConv1D` blocks, Self-Attention, `MultiHeadAttention`, and four stacked `Bidirectional LSTM` layers.
4. **Interactive Web Interface:** Providing instant, real-time classification (`Real` vs. `Fake`) with probability confidence scores through a responsive Streamlit UI.

---

## ✨ Key Features

- **Real-Time News Classification:** Paste any English news headline or full article to receive an instant `Real News` or `Fake News` verdict.
- **Confidence Probability Scoring:** Displays exact model confidence percentages derived from the Softmax output distribution.
- **Hybrid Attention-BiLSTM Network:** Captures multi-scale n-gram features via dilated separable convolutions and contextual dependencies in both forward and backward directions.
- **Bias-Mitigated Dataset Pipeline:** Removes publisher-specific leakage (`Reuters`) to generalize accurately across external news outlets (e.g., *The Daily Star*, international media).
- **Resource-Cached Inference:** Uses `@st.cache_resource` in Streamlit to load the `.keras` model, `tokenizer.pkl`, and NLTK corpora once in memory for fast subsequent predictions.

---

## 🔄 System Architecture & Workflow

```text
┌─────────────────────────┐
│   Raw User News Input   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  NLTK Text Preprocessing│  ──► Lowercase ──► Remove Punctuation ──► Word Tokenize
│                         │  ──► Remove English Stopwords ──► WordNet Lemmatization
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Tokenization & Padding  │  ──► Keras Tokenizer (tokenizer.pkl)
│                         │  ──► Post-Padding / Truncating (maxlen = 120)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Hybrid Deep Learning   │  ──► Embedding (128-dim)
│         Model           │  ──► 4x [SeparableConv1D + Attention/MHA + Bi-LSTM]
│ (Sentiment_model_.keras)│  ──► Flatten ──► Dense (128, 64) + LeakyReLU ──► Softmax
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Streamlit Web UI      │  ──► Class 0: 🚨 Fake News (Confidence %)
│                         │  ──► Class 1: ✅ Real News (Confidence %)
└─────────────────────────┘
```

---

## 🧹 NLP Preprocessing Pipeline

Every input text—both during training and live inference—passes through an identical transformation pipeline to guarantee feature consistency:

| Step | Operation | Tool / Library | Purpose |
| :--- | :--- | :--- | :--- |
| **1** | **Lowercasing** | Python `str.lower()` | Standardizes vocabulary casing (`News` $\rightarrow$ `news`). |
| **2** | **Punctuation Removal** | `string.punctuation` | Strips punctuation marks and special symbols. |
| **3** | **Tokenization** | `nltk.tokenize.word_tokenize` | Splits cleaned strings into individual word tokens (`punkt`, `punkt_tab`). |
| **4** | **Stop-Word Removal** | `nltk.corpus.stopwords` | Filters out high-frequency, low-information English words (*the, is, at, which*). |
| **5** | **Lemmatization** | `nltk.stem.WordNetLemmatizer` | Reduces inflected words to their base dictionary lemma (*running* $\rightarrow$ *run*). |
| **6** | **Integer Encoding** | `keras.preprocessing.text.Tokenizer` | Maps words to integer indices with `<OOV>` handling for unseen vocabulary. |
| **7** | **Sequence Padding** | `pad_sequences` | Enforces a uniform input vector length of **`120` tokens** (`post` padding & truncation). |

---

## 🧠 Deep Learning Model Architecture

The core classifier (`Sentiment_model_.keras`) is a custom multi-stage neural network engineered to extract hierarchical textual representations:

### Layer-by-Layer Specification

| Stage | Layer Components | Key Hyperparameters |
| :--- | :--- | :--- |
| **Input & Embedding** | `Input` $\rightarrow$ `Embedding` | `input_shape=(120,)`, `output_dim=128` |
| **Feature Block 1** | `SeparableConv1D` $\rightarrow$ `BatchNorm` $\rightarrow$ `Dropout` $\rightarrow$ `MaxPooling1D` $\rightarrow$ `Attention` $\rightarrow$ `MultiHeadAttention` $\rightarrow$ `Concatenate` $\rightarrow$ `Bidirectional(LSTM)` | Filters: `32`, Kernel: `3`, Multiplier: `3`, Pool: `2`, MHA Heads: `4` (`key_dim=16`), Bi-LSTM Units: `64` |
| **Feature Block 2** | `SeparableConv1D` (Dilated) $\rightarrow$ `BatchNorm` $\rightarrow$ `Dropout` $\rightarrow$ `MaxPooling1D` $\rightarrow$ `Attention` $\rightarrow$ `MultiHeadAttention` $\rightarrow$ `Concatenate` $\rightarrow$ `Bidirectional(LSTM)` | Filters: `64`, Kernel: `3`, Dilation: `2`, Multiplier: `4`, MHA Heads: `6` (`key_dim=32`), Bi-LSTM Units: `64` |
| **Feature Block 3** | `SeparableConv1D` (Dilated) $\rightarrow$ `BatchNorm` $\rightarrow$ `Dropout` $\rightarrow$ `MaxPooling1D` $\rightarrow$ `Bidirectional(LSTM)` | Filters: `128`, Kernel: `3`, Dilation: `3`, Multiplier: `5`, Bi-LSTM Units: `64` |
| **Feature Block 4** | `SeparableConv1D` (Dilated) $\rightarrow$ `BatchNorm` $\rightarrow$ `Dropout` $\rightarrow$ `MaxPooling1D` $\rightarrow$ `Bidirectional(LSTM)` | Filters: `256`, Kernel: `3`, Dilation: `4`, Multiplier: `6`, Bi-LSTM Units: `64` |
| **Classification Head** | `Flatten` $\rightarrow$ `Dense` + `LeakyReLU` + `Dropout` $\rightarrow$ `Dense` + `LeakyReLU` + `Dropout` $\rightarrow$ `Dense` | Dense Units: `128` $\rightarrow$ `64` $\rightarrow$ `2`, Regularization: `L2(1e-2)`, Output Activation: `softmax` |

### Training Configuration

- **Dataset:** ISOT Fake News Dataset (`cleaned_data.csv`)
- **Train / Test Split:** `80%` Training / `20%` Testing (`random_state=42`, `shuffle=True`)
- **Validation Split:** `20%` of training set during fitting
- **Optimizer:** `Adam(learning_rate=0.001)`
- **Loss Function:** `sparse_categorical_crossentropy`
- **Batch Size:** `32`
- **Regularization:** `L2(1e-4)` across convolutional/recurrent layers, `L2(1e-2)` on dense layers, and `Dropout(0.2)` throughout.

---

## 🛡️ Data Leakage & Bias Mitigation

### The Problem
During initial testing with real-world articles from external news portals (such as *The Daily Star*), the baseline model exhibited a bias toward classifying legitimate articles as **Fake**. Investigation revealed **Data Leakage** inherent to the ISOT Fake News Dataset: nearly all authentic (`True`) training samples began with agency headers such as `"WASHINGTON (Reuters) -"`, whereas `Fake` samples lacked them. Consequently, the network learned a shortcut heuristic based on the presence of the token `"reuters"`.

### The Solution
1. **Watermark Stripping:** Applied case-insensitive regex/string cleaning to purge all occurrences of `"reuters"` and `"reuter"` from `cleaned_data.csv` prior to tokenization.
2. **Tokenizer Synchronization:** Regenerated and serialized a fresh `tokenizer.pkl` directly from the de-biased corpus so word indices aligned strictly with the retrained weights.
3. **Model Retraining:** Retrained the Attention-BiLSTM model from scratch on the cleaned corpus, forcing the network to evaluate linguistic structure, tone, and semantic coherence rather than publisher metadata.

---

## 📁 Project Structure

```text
Fake_News_Project/
│
├── .venv/                        # Python virtual environment
├── app.py                        # Main Streamlit web application & inference pipeline
├── model.ipynb                   # Jupyter Notebook for preprocessing, training & model export
├── data_exploration.ipynb        # Exploratory Data Analysis (EDA) on the news dataset
├── cleaned_data.csv              # Preprocessed and bias-cleaned dataset
├── Sentiment_model_.keras        # Trained Keras deep learning model weights & architecture
├── tokenizer.pkl                 # Serialized Keras Tokenizer fitted on cleaned training text
├── fake_news_model.tflite        # Experimental TensorFlow Lite model export
└── List of Fake news 1 US.txt    # Sample news articles for local testing & validation
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- **Python:** Version `3.10` – `3.13`
- **OS:** Windows 10/11, macOS, or Linux

### 2. Clone or Navigate to the Project Directory
```bash
cd "Fake_News_Project"
```

### 3. Create & Activate a Virtual Environment
**Windows (Command Prompt / PowerShell):**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Required Dependencies
```bash
pip install streamlit tensorflow numpy pandas scikit-learn nltk
```

### 5. Download NLTK Corpora
`app.py` automatically downloads required NLTK packages on first launch, or you can pre-install them manually:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
```

---

## 🚀 Usage Guide

### Running the Web Application
1. Ensure your virtual environment (`.venv`) is active.
2. Launch the Streamlit server from the project root directory:
   ```bash
   streamlit run app.py
   ```
3. Open your browser to **`http://localhost:8501`**.
4. Paste any English news article or headline into the **"Enter News Text"** field and click **"Analyze"**.

### Retraining the Model (Optional)
If you update `cleaned_data.csv` and wish to retrain the model and regenerate `tokenizer.pkl`:
1. Open `model.ipynb` in VS Code or Jupyter Notebook.
2. Execute the final end-to-end training cell to clean text, fit and save `tokenizer.pkl`, train the Attention-BiLSTM network, and export `Sentiment_model_.keras`.
3. Restart the Streamlit server (`Ctrl + C`, then `streamlit run app.py`) to load the updated artifacts.

---

## 🛠️ Troubleshooting

| Issue / Error | Root Cause | Resolution |
| :--- | :--- | :--- |
| `LookupError: Resource 'punkt_tab' not found` | Newer NLTK versions require `punkt_tab` alongside `punkt` for `word_tokenize`. | Ensure `nltk.download('punkt_tab')` is included in `download_nltk()` in `app.py`. |
| `RuntimeError: Select TensorFlow op(s) ... FlexTensorListReserve` | `.tflite` runtime lacks Flex delegate ops required by `Bidirectional(LSTM)` layers. | Load the native `.keras` model (`load_model("Sentiment_model_.keras")`) directly in `app.py`. |
| `EOFError: Ran out of input` on `tokenizer.pkl` | `tokenizer.pkl` was saved empty or corrupted when `tokenizer` was undefined in notebook memory. | Run the complete tokenizer generation and `pickle.dump` cell in `model.ipynb` before launching Streamlit. |
| `ERR_CONNECTION_REFUSED` on `localhost:8501` | The Streamlit local server process stopped or was not started. | Run `streamlit run app.py` inside the active `.venv` terminal. |

---

## 🔮 Future Roadmap

- [ ] **Multilingual Support (Bengali Fake News Detection):** Extend preprocessing and embeddings (e.g., BanglaBERT / multilingual transformers) to support Bengali news portals.
- [ ] **Live URL Scraping:** Allow users to paste a direct article URL to automatically extract headlines and body text via `BeautifulSoup` or `newspaper3k`.
- [ ] **Explainable AI (XAI):** Integrate LIME or Attention weight heatmaps in the Streamlit UI to highlight specific words driving the model's `Fake` or `Real` decision.
- [ ] **Cloud Deployment:** Containerize with Docker and deploy on Streamlit Community Cloud or Hugging Face Spaces.

---

## 👨‍💻 Project Timeline & Author

- **Project Name:** Verity Lens — Fake News Detector
- **Domain:** Artificial Intelligence, Natural Language Processing & Deep Learning
- **Development Period:** October 2026
- **Developer:** **Meraj Islam**  Ashraful Alam & Kamrul Hasan — *Department of Computer Science and Engineering, Northern University Bangladesh*

```
