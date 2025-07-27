\documentclass[11pt,a4paper]{article}

% ---- PACKAGES ----
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}        % Police moderne
\usepackage{geometry}       % Marges
\geometry{margin=2.5cm}
\usepackage{setspace}       % Interligne
\usepackage{hyperref}       % Liens cliquables
\usepackage{graphicx}       % Images
\usepackage{booktabs}       % Tableaux jolis
\usepackage{titlesec}       % Design sections
\usepackage{enumitem}       % Listes custom
\usepackage{xcolor}         % Couleurs

% ---- STYLING ----
\titleformat{\section}{\large\bfseries}{\thesection.}{0.5em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection.}{0.5em}{}
\renewcommand{\baselinestretch}{1.15}
\setlength{\parskip}{6pt}

% ---- DOCUMENT ----
\begin{document}

\begin{center}
    {\LARGE \textbf{Vision-Language Model for ECG Diagnostics}}\\[6pt]
    \textbf{Master Research Project – Technion, Israel Institute of Technology (AIM Lab)} \\
    \rule{0.9\textwidth}{0.4pt}
\end{center}

\section{Project Overview}
This research project focuses on \textbf{Vision-Language Models (VLMs)} applied to \textbf{electrocardiogram (ECG) diagnostics}.  
The work builds upon the \href{https://arxiv.org/abs/2410.19008}{PULSE benchmark} and evaluates \textbf{SmolVLM-256M-Instruct}, a lightweight open-source VLM fine-tuned for ECG reasoning and classification tasks.

\section{Objectives}
\begin{itemize}[leftmargin=*]
    \item Fine-tune \textbf{SmolVLM-256M-Instruct (256M parameters)} for ECG-specific reasoning tasks.
    \item Benchmark performance against \textbf{state-of-the-art proprietary and open-source models} (PULSE-7B, GPT-4o, Gemini, Claude).
    \item Focus on the \textbf{ECGBench benchmark}, particularly the \textbf{CODE15 dataset}, using metrics such as F1-score, AUC, Hamming Loss (HL), and ANLS.
\end{itemize}

\section{Results – CODE15 (In-Domain)}
\begin{center}
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{F1} & \textbf{AUC} & \textbf{HL} \\
\midrule
GPT-4o                  & 24.9  & 59.9  & 15.7 \\
Gemini 1.5 Pro          & 20.0  & 56.7  & 15.9 \\
Claude 3.5 Sonnet       & 20.3  & 58.3  & 17.8 \\
LLaVA-Med               & 27.0  & 69.2  & 33.4 \\
MiniCPM-V-2.6           & 25.3  & 56.6  & 22.0 \\
\textbf{PULSE-7B}       & \textbf{85.4} & \textbf{90.7} & \textbf{5.0} \\
\textbf{SmolVLM-256M (Ours)} & 27.5  & 65.3  & 16.5 \\
\bottomrule
\end{tabular}
\end{center}

\noindent \textbf{ANLS (SmolVLM-256M):} 68.7

\noindent \textit{Key Insight:} Our \textbf{SmolVLM-256M} performs among the upper range of open-source models while remaining \textbf{lightweight and efficient}. PULSE-7B remains the leader, but our model demonstrates strong adaptability for its size.

\section{Sample Predictions (CODE15)}

\subsection*{Example 1 – Correct Prediction}
\includegraphics[width=\textwidth]{images/example1.png}\\
\textbf{Prompt:} The ECG image above belongs to a 54-year-old patient. Identify the cardiac rhythm and classify whether it is NORMAL or ABNORMAL.\\
\textbf{Ground Truth:} ABNORMAL – Atrial Fibrillation\\
\textbf{Prediction (Ours):} ABNORMAL – Atrial Fibrillation.

\subsection*{Example 2 – Diagnostic Report}
\includegraphics[width=\textwidth]{images/example2.png}\\
\textbf{Prompt:} Evaluate the ECG and write a comprehensive diagnostic report.\\
\textbf{Ground Truth:} Normal sinus rhythm, no arrhythmia or conduction issues.\\
\textbf{Prediction (Ours):} Sinus rhythm, low QRS voltages (may suggest mild conduction abnormalities).

\subsection*{Example 3 – Incorrect Prediction}
\includegraphics[width=\textwidth]{images/example3.png}\\
\textbf{Prompt:} Classify the ECG condition (NORMAL or ABNORMAL) and specify the rhythm.\\
\textbf{Ground Truth:} ABNORMAL – Ventricular Tachycardia\\
\textbf{Prediction (Ours):} NORMAL – Sinus Rhythm.

\section{About Me}
I am a Master's student in \textbf{Engineering \& Artificial Intelligence} at \textbf{ECE Paris}, currently completing my final year at the \textbf{Technion – Israel Institute of Technology (AIM Lab)}.  
\textbf{Focus areas:}
\begin{itemize}[leftmargin=*]
    \item Vision-Language Models (VLMs) \& Multimodal AI
    \item Deep Learning for Healthcare Applications
    \item Model Efficiency \& Domain Adaptation
\end{itemize}

\section{References}
\begin{itemize}[leftmargin=*]
    \item PULSE: ``PULSE: A Unified Benchmark for ECG Reasoning with Vision-Language Models'' – \href{https://arxiv.org/abs/2410.19008}{arXiv:2410.19008}.
    \item SmolVLM: SmolVLM-256M-Instruct – Hugging Face.
\end{itemize}

\end{document}
