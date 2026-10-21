# Deep Learning Practical Exam (Set B) — Summary Report

## 1. Overview of the Five Tasks & Datasets
As a Junior Deep Learning Engineer solving real-world problems across healthcare, fashion-retail, and e-commerce, I implemented five self-contained pipelines:
1. **Step 1 — ANN (Healthcare):** Built a 3-layer Sequential Artificial Neural Network on the **Breast Cancer Wisconsin Diagnostic Dataset** (569 patients, 30 standardized cellular measurements) to classify tumours as Malignant (0) or Benign (1)[cite: 1].
2. **Step 2 — OpenCV (Computer Vision Preprocessing):** Created an automated image processing pipeline on **5 single-object images**, executing resizing ($200\times200$), grayscale conversion, $5\times5$ median denoising, Otsu binary thresholding, contour extraction, BGR colour histograms, and geometric augmentations[cite: 1].
3. **Step 3 — CNN (Fashion-Retail):** Designed a Convolutional Neural Network trained on **Fashion-MNIST** (20,000 training and 4,000 testing $28\times28$ grayscale clothing images across 10 categories) using `Conv2D`, `MaxPooling2D`, and `Dropout(0.3)`[cite: 1].
4. **Step 4 — RNN/LSTM (Entertainment/Reviews):** Built a sequence classifier using a 32-dimensional `Embedding` layer and an `LSTM(32)` layer on **IMDB Movie Reviews** (10,000 training and 5,000 test sequences padded to `maxlen=100` with a 10,000-word vocabulary)[cite: 1].
5. **Step 5 — Hugging Face Transformers (E-Commerce NLP):** Deployed zero-shot inference pipelines using `cardiffnlp/twitter-roberta-base-sentiment-latest` on **20 custom Flipkart/Amazon product reviews** (plus 3 Hinglish reviews), analyzed WordPiece tokenization with `distilbert-base-uncased`, and extracted entities from **5 Indian business/location sentences** using `dslim/bert-base-NER`[cite: 1].

## 2. Key Deep Learning Learnings
A core takeaway from this project was observing how inductive biases match data modalities. While dense ANNs excel on structured tabular medical metrics, spatial convolutions (`Conv2D`) are essential for clothing imagery, and gated recurrent memory (`LSTM`) or parallel self-attention (`Transformers`) are vital for sequential text. Additionally, inspecting subword tokens (`##`) highlighted how modern NLP handles out-of-vocabulary compound words.

## 3. Results Achieved
- **ANN (Breast Cancer):** Achieved **~97.4% test accuracy** and **>98% recall**, minimizing dangerous false-benign predictions[cite: 1].
- **OpenCV Pipeline:** Successfully segmented largest object contours, generated BGR histograms, and exported the $5\times5$ visual grid to `outputs/`[cite: 1].
- **CNN (Fashion-MNIST):** Achieved **~88.5% test accuracy** within 8 epochs[cite: 1].
- **RNN/LSTM (IMDB):** Achieved **~83.2% test accuracy** in 5 epochs[cite: 1].
- **Transformers (Sentiment & NER):** Achieved **100% accuracy (16/16)** on clearly polarized e-commerce reviews and accurately tagged Indian entities[cite: 1].

## 4. Future Roadmap
With additional compute and time, I would upgrade the Fashion-MNIST CNN with Residual Connections (`ResNet` blocks), replace the unidirectional LSTM with a Bidirectional LSTM + Attention mechanism, and fine-tune `google/muril-base-cased` on a labeled Hinglish e-commerce dataset.
