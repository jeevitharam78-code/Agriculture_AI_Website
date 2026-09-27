\# 🌾 Agriculture AI Assistant \& Plant Disease Detection



An AI-powered web application that combines a fine-tuned Agriculture Domain Expert LLM with an image-based Plant Disease Detection system in a single Flask website.



The project is designed to provide farmers, students, and agriculture users with intelligent agriculture-related information and plant disease analysis through an easy-to-use web interface.



\---



\## 🚀 Project Overview



Agriculture involves many decisions related to crops, soil, irrigation, fertilizers, pests, diseases, and farming practices. Access to reliable agricultural information can be difficult, especially when users need quick answers or assistance identifying plant diseases.



This project combines two AI modules into one web application:



\### 🌾 1. Agriculture AI Assistant



A domain-specific conversational AI assistant based on:



\- Google Gemma 4 E2B Instruct

\- Agriculture-domain LoRA fine-tuning

\- KisanVaani agriculture question-answer dataset

\- PEFT

\- 4-bit quantization

\- GPU acceleration using CUDA



The assistant can answer agriculture-related questions such as:



\- Crop cultivation

\- Irrigation

\- Fertilizers

\- Soil management

\- Pest management

\- Plant care

\- Crop-related problems

\- General agricultural practices



\### 🌿 2. Plant Disease Detection



The second module identifies plant diseases from uploaded leaf images.



It uses:



\- EfficientNet-B0

\- PyTorch

\- Trained plant disease classification model

\- Image preprocessing

\- Disease information database



The system provides:



\- Predicted disease

\- Confidence

\- Disease description

\- Treatment information

\- Prevention suggestions



\---



\# 🏗️ System Architecture



```text

&#x20;                   USER

&#x20;                     │

&#x20;                     ▼

&#x20;            ┌─────────────────┐

&#x20;            │  Flask Website  │

&#x20;            └────────┬────────┘

&#x20;                     │

&#x20;         ┌───────────┴───────────┐

&#x20;         │                       │

&#x20;         ▼                       ▼

&#x20;┌──────────────────┐    ┌────────────────────┐

&#x20;│ Agriculture AI   │    │ Plant Disease      │

&#x20;│ Assistant        │    │ Detection           │

&#x20;└────────┬─────────┘    └──────────┬─────────┘

&#x20;         │                         │

&#x20;         ▼                         ▼

&#x20;┌──────────────────┐      ┌──────────────────┐

&#x20;│ Gemma 4 E2B      │      │ EfficientNet-B0  │

&#x20;│ Instruct         │      │                  │

&#x20;└────────┬─────────┘      └────────┬─────────┘

&#x20;         │                         │

&#x20;         ▼                         ▼

&#x20;┌──────────────────┐      ┌──────────────────┐

&#x20;│ Agriculture LoRA │      │ Disease          │

&#x20;│ Fine-tuning      │      │ Classification   │

&#x20;└────────┬─────────┘      └────────┬─────────┘

&#x20;         │                         │

&#x20;         ▼                         ▼

&#x20;   Agriculture                Disease +

&#x20;   Answer                     Treatment +

&#x20;                              Prevention

