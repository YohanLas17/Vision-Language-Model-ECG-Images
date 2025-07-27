# Vision-Language Model for ECG Image Diagnostic  
**Master Research Project – Technion - Israel Institute of Technology (AIM Lab)**  

---

## 📖 Project Overview  
This repository presents my **Master’s research project** on **Vision-Language Models (VLMs)** applied to **electrocardiogram (ECG) diagnostics**.  
The project builds upon the **[PULSE benchmark](https://arxiv.org/abs/2410.19008)** and evaluates **SmolVLM-256M-Instruct**, a lightweight open-source VLM, fine-tuned for **ECG reasoning and classification tasks**.

---

## 🎯 Objectives  
- Fine-tune **SmolVLM-256M-Instruct (256M parameters)** for ECG-specific reasoning tasks.  
- Benchmark its performance against **state-of-the-art proprietary and open-source models** such as **PULSE-7B, GPT-4o, Gemini, and Claude**.  
- Focus on the **ECGBench benchmark**, with particular attention to the **CODE15 dataset**, evaluating **F1, AUC, Hamming Loss (HL)**, and **ANLS** for our model.

---

## 📊 Results – CODE15 (In-Domain)

| Model                   | F1    | AUC   | HL    |
|-------------------------|-------|-------|-------|
| GPT-4o                  | 24.9  | 59.9  | 15.7  |
| Gemini 1.5 Pro          | 20.0  | 56.7  | 15.9  |
| Claude 3.5 Sonnet       | 20.3  | 58.3  | 17.8  |
| LLaVA-Med               | 27.0  | 69.2  | 33.4  |
| MiniCPM-V-2.6           | 25.3  | 56.6  | 22.0  |
| **PULSE-7B**            | **85.4** | **90.7** | **5.0**  |
| **SmolVLM-256M (Ours)** | 27.5  | 65.3  | 16.5  |

**ANLS (SmolVLM-256M only): 68.7**

**Key Insight:**  
Our **SmolVLM-256M** achieves performance in the **upper range of open-source models**, especially in terms of ANLS, while being **lightweight and efficient**.  
**PULSE-7B** remains the **clear leader**, but our model demonstrates strong adaptability for its size.

---

## 🧠 Examples from CODE15 Dataset  

### **Example 1 – Correct Prediction**  
**Image (ECG strip):**  
![image.png](attachment:7111abc1-3666-4ebc-8dd9-3b90a8aff3e1:image.png)

**Prompt:** 
 The ECG image above belongs to a 54-year-old patient. Identify the cardiac rhythm and classify whether it is NORMAL or ABNORMAL.
**Ground Truth Answer:**
ABNORMAL – Atrial Fibrillation
**Prediction by SmolVLM-256M (Ours):**
ABNORMAL – Atrial Fibrillation

### **Example 2 – Open-Ended Diagnostic Report**

**Image (ECG strip):** 
![image.png](attachment:60f6364c-ee9d-46ca-8272-1ab62c5208fa:image.png)

**Prompt:** 
Evaluate the ECG and write a comprehensive diagnostic report, including the final diagnosis.
**Ground Truth Answer:**
The electrocardiogram (ECG) reveals a normal sinus rhythm, characterized by a regular heartbeat with a rate within the normal range. The ECG tracing exhibits a normal P-wave, QRS complex, and T-wave morphology, with no signs of arrhythmia, conduction abnormalities, or ischemic changes.
**Prediction by SmolVLM-256M (Ours):**
The electrocardiogram (ECG) reveals a sinus rhythm, indicating a normal heart rate and rhythm. The ECG also shows poor R wave progression, which is likely a normal variant, but may be a sign of a subtle abnormality in the heart's electrical conduction system. Additionally, the ECG demonstrates low QRS voltages in the precordial leads, which may suggest a decrease in the heart's electrical activity.


### **Example 3 – Incorrect Prediction**

**Image (ECG strip):**
![download.png](attachment:90b92c5c-222a-41ef-9785-19b1d8c87723:download.png)

**Prompt:** 
Classify the ECG condition (NORMAL or ABNORMAL) and specify the rhythm.
**Ground Truth Answer:**
ABNORMAL – Ventricular Tachycardia
**Prediction by SmolVLM-256M (Ours):**
NORMAL – Sinus Rhythm


## Research Context
This project takes inspiration from the PULSE benchmark, which sets a new standard for ECG reasoning with vision-language models.
My work focuses on adapting lightweight VLMs for medical diagnostic tasks, showing that smaller models like SmolVLM can achieve competitive results with optimized fine-tuning.


## About Me
I am a Master’s student in Engineering & Artificial Intelligence at École Centrale d’Électronique (ECE Paris), currently completing my final year at the Technion – Israel Institute of Technology.
I focus on:

Multimodal AI & Vision-Language Models

Deep Learning for Healthcare Applications

Model Efficiency and Domain Adaptation

## References
PULSE: "PULSE: A Unified Benchmark for ECG Reasoning with Vision-Language Models" – arXiv:2410.19008

SmolVLM: SmolVLM-256M-Instruct – Hugging Face

GPT-4o: OpenAI


