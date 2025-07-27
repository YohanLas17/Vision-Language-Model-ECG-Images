# ⚡ Vision-Language Model for ECG Diagnostics  
**Master Research Project – Technion (AIM Lab)**  

---

## 🚀 Quick Overview  
I developed and fine-tuned **SmolVLM-256M**, a **256M parameter lightweight Vision-Language Model**, achieving **competitive performance** in **ECG diagnostic reasoning** compared to leading proprietary models like **GPT-4o** and **Gemini**.

- **Benchmarked on ECGBench (CODE15 dataset)** with metrics: **F1, AUC, Hamming Loss (HL), ANLS**.  
- Demonstrated that **smaller open-source VLMs can rival larger proprietary models** when fine-tuned for domain-specific medical tasks.  
- Designed and implemented a **full fine-tuning and evaluation pipeline** tailored to ECG images and medical prompts.

---

## 📊 Key Results – CODE15  

| Model                   | F1    | AUC   | HL    |
|-------------------------|-------|-------|-------|
| GPT-4o                  | 24.9  | 59.9  | 15.7  |
| Gemini 1.5 Pro          | 20.0  | 56.7  | 15.9  |
| Claude 3.5 Sonnet       | 20.3  | 58.3  | 17.8  |
| LLaVA-Med               | 27.0  | 69.2  | 33.4  |
| MiniCPM-V-2.6           | 25.3  | 56.6  | 22.0  |
| **PULSE-7B**            | **85.4** | **90.7** | **5.0**  |
| **SmolVLM-256M (Ours)** | 27.5  | 65.3  | 16.5  |

**ANLS (SmolVLM-256M): 68.7**

**💡 Key Insight:**  
Our **SmolVLM-256M** achieves **upper-range open-source performance** while remaining **efficient and lightweight**.

---

## 🧠 Sample Predictions (CODE15)

### Example 1 – Correct Prediction  
![ECG Example 1](images/example1.png)

**Prompt:**  
*The ECG image above belongs to a 54-year-old patient. Identify the cardiac rhythm and classify whether it is NORMAL or ABNORMAL.*  
**Ground Truth:** ABNORMAL – Atrial Fibrillation  
**Prediction (Ours):** ABNORMAL – Atrial Fibrillation  

---

### Example 2 – Diagnostic Report  
![ECG Example 2](images/example2.png)

**Prompt:**  
*Evaluate the ECG and write a comprehensive diagnostic report.*  
**Ground Truth:** Normal sinus rhythm, no arrhythmia or conduction issues.  
**Prediction (Ours):** Sinus rhythm, low QRS voltages (may suggest mild conduction abnormalities).  

---

### Example 3 – Incorrect Prediction  
![ECG Example 3](images/example3.png)

**Prompt:**  
*Classify the ECG condition (NORMAL or ABNORMAL) and specify the rhythm.*  
**Ground Truth:** ABNORMAL – Ventricular Tachycardia  
**Prediction (Ours):** NORMAL – Sinus Rhythm  

---

## 👤 About Me  
I am a Master’s student in **Engineering & Artificial Intelligence** at **ECE Paris**, currently completing my final year at the **Technion – Israel Institute of Technology (AIM Lab)**.  
**Focus areas:**  
- **Vision-Language Models (VLMs) & Multimodal AI**  
- **Deep Learning for Healthcare**  
- **Model Efficiency & Domain Adaptation**

---

## 📚 References  
- **PULSE:** [PULSE: A Unified Benchmark for ECG Reasoning with VLMs](https://arxiv.org/abs/2410.19008)  
- **SmolVLM-256M-Instruct:** Hugging Face  
- **GPT-4o:** OpenAI  
