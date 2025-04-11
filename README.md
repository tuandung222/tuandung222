# <div align="center">Hi 👋, I'm Dung Vo</div>
## <div align="center">AI Research and Development Engineer & Computer Science Grad Student</div>

Welcome to my GitHub profile! I'm Dung Vo Pham Tuan, currently pursuing my Master's degree in Computer Science with a specialization in Applied Data Science at Ho Chi Minh University of Technology (HCMUT).

I have a strong background in mathematics and a passion for Artificial Intelligence, with experience in Data Science, Natural Language Processing, and Computer Vision. My goal is to become a Professional Machine Learning Scientist, continuously improving my expertise and contributing to cutting-edge AI research.

<img align="right" src="https://github-readme-stats.vercel.app/api/top-langs?username=tuandung222&show_icons=true&locale=en&layout=compact" alt="tuandung222" />

- 🔭 I'm currently working on **Text-based Person Re-identification and Knowledge Distillation for LLMs**

- 🌱 I'm currently learning **MLOps, Vector Databases, and Large Language Model Optimization**

- 👯 I'm looking to collaborate on **AI Research Projects in Computer Vision and NLP**

- 📫 How to reach me: **vophamtuandung05hv@gmail.com**

- 📄 Check out my [portfolio website](https://tuandung222.github.io/Portfolio) for more details

## 🌐 Connect with me

[![LinkedIn](https://img.shields.io/badge/-Dung%20Vo-blue?style=for-the-badge&logo=Linkedin&logoColor=white)](https://www.linkedin.com/in/dung-vo)
[![Gmail](https://img.shields.io/badge/vophamtuandung05hv-red?style=for-the-badge&logo=Gmail&logoColor=white)](mailto:vophamtuandung05hv@gmail.com)
[![GitHub](https://img.shields.io/badge/tuandung222-black?style=for-the-badge&logo=GitHub&logoColor=white)](https://github.com/tuandung222)
[![Portfolio](https://img.shields.io/badge/Portfolio-black?style=for-the-badge&logo=github&logoColor=white)](https://tuandung222.github.io/Portfolio)

## 💼 Professional Experience

### AI Research and Development Engineer
**Dien Toan Group** | Jul 2024 - Apr 2025 | Tan Binh District, HCM City
- Pretrained a multilingual vision-language backbone (Vietnamese/English/Chinese) for Text-based Person Re-identification on a large-scale dataset (36 million image-text pairs) using 4 NVIDIA A100 GPUs
- Pretrained/Fine-tuned models for Text-based Person Re-identification using custom PyTorch implementation without relying on high-level training frameworks
- Collaborated with Deployment Engineers to optimize models in ONNX/Tensor-RT format and deploy them to production using NVIDIA Triton Inference Server
- Designed and documented the data annotation workflow, evaluating tools like CVAT and Label Studio to streamline the dataset preparation process
- Extended the original English pretraining datasets by adding Chinese and Vietnamese captioning annotations, demonstrating that multilingual pretraining improves zero-shot retrieval performance by over 1.2% Rank-1
- Leveraged advanced LLMs and MLLMs with optimized inference frameworks (vLLM, lmdeploy, SGLang) to accelerate data augmentation, enhancing dataset diversity and quality for improved model performance

### AI Researcher
**Dien Toan Group** | Oct 2023 - Jun 2024 | Tan Binh District, HCM City
- Proposed shifting the company's focus from fixed-attribute person re-identification to Vietnamese Text-based Person Re-identification, a more practical solution for the local context
- Established this feature as the company's flagship AI product, gaining attention from government agencies and outperforming competitors
- Constructed the first Vietnamese pretraining and benchmark datasets for this task, significantly enhancing fine-tuning efficiency and model generalization
- Developed a Vietnamese Vision-Language backbone based on the ALBEF architecture, integrating SOTA Vietnamese language models such as PhoBERT (VinAI Research) and ViDeBERTa (FSOFT AI)
- Improved the image encoder stream using architectures from the HAP and SOLIDER frameworks, leveraging Vision Transformer/Swin Transformer models pre-trained on human-centric surveillance datasets

### AI Research Intern
**Dien Toan Group** | Jun 2023 - Jul 2023 | Tan Binh District, HCM City
- Conducted comprehensive survey and analysis of research papers on Transformer-based architectures for Object Detection and Multiple Object Tracking
- Implemented and comprehended the underlying mechanisms of Trackformers (Facebook AI Research, CVPR 2023) for tracking pedestrians and vehicles at the campus of Ho Chi Minh University of Technology
- Preprocessed realistic surveillance video data using OpenCV and FFmpeg for frame extraction, noise reduction, and format standardization
- Evaluated multiple efficient data annotation tools for creating high-quality tracking training datasets
- Optimized Trackformers by modifying architecture/loss to extend from single-class (human-only) to multi-class tracking and mitigate class imbalance, where pedestrian instances significantly outnumbered vehicles in the dataset
- Demonstrated real-time system performance in multiple real-world environments, including a technical presentation at Ho Chi Minh City University of Technology (HCMUT)

### Research Assistant
**Data Science Lab, CSE Faculty, HCMUT** | Aug 2023 - Feb 2025 | District 10, HCM City
- Conducted academic research on Text-based Person Re-identification under the supervision of the Lab Head, who is also one of the leaders of Dien Toan Group
- Served as the sole researcher responsible for the entire project, developing a state-of-the-art model that achieved a 2.8% Rank-1 accuracy improvement on benchmark datasets over recent SOTA models
- The resulting models became the highlighted AI products of the lab and serves as a benchmark for future research
- Due to the sensitivity of surveillance camera data and commercial constraints, the work is being developed as an internal proprietary product rather than submitted for publication
- Developed and publicly disseminated technical documentation detailing key innovations in training pipelines, ablation studies, and multi-modal result visualizations to benefit the broader technical community

## 🎓 Education

### Ho Chi Minh University of Technology
**Master's Degree in Computer Science** | Jan 2024 - Present
- Specializing in Applied Data Science
- Current GPA: 8.48/10 (24/60 credits)

### Ho Chi Minh University of Technology
**Bachelor's Degree in Computer Science** | Aug 2020 - Nov 2024
- Honors Degree with dual specializations in Image Processing & Computer Vision and Applied Artificial Intelligence
- GPA: 8.69/10 (3.8/4) - Thesis Score: 9.7/10 (AI Research)

### Quang Trung High School for the Gifted, Binh Phuoc
**High School Diploma** | Aug 2017 - Jul 2020
- Specialized in Mathematics with a GPA of 9.4/10
- Direct Admission to University due to Third Prize, Vietnam Mathematical Olympiad 2020

## 🚀 Projects

### Knowledge Distillation for Coding Multi-Choice Coding Question Answering
**Individual Project** | Apr 2025
- Implemented an open-source knowledge distillation framework (GitHub repo) to transfer structured reasoning from GPT-4o to a mini-LLM (Qwen2.5 Coder 1.5B Instruct) for Coding Multi-Choice Coding Question Answering
- Generated a YAML-based reasoning dataset from a subset curated samples from CodeMMLU using GPT-4o, leveraging OpenAI SDK for the data synthesis pipeline
- Designed a structured reasoning framework (understanding question → analysis choices → reasoning → conclusion → answer) that mirrors how CS students and researchers systematically approach problems
- Fine-tuned the model using parameter-efficient techniques (LoRA, Lion optimizer) with advanced optimization strategies (gradient checkpointing, mixed precision training,...) for memory efficiency
- Created a 4-bit quantized interactive live demo on Hugging Face Spaces (live demo) for evaluation and demonstration
- Developed comprehensive training analytics with WandB integration for experiment tracking, including prompt monitoring, token distribution analysis, and quality metrics
- Structured the repository with modular components and comprehensive documentation including setup guides, architecture diagrams, and clearly explanations for reproducibility

### Semantic Search with Large Language Model and Vector Database
**Individual Project** | Mar 2025
- Designed and developed a full-stack RAG system using FastAPI, Weaviate, and OpenAI SDK, with a self-hosted vector database for data control and privacy (GitHub repo)
- Implemented a containerized microservices architecture with Docker Compose local development environment
- Extended deployment to Google Kubernetes Engine (GKE) using Terraform for infrastructure provisioning and Kubernetes manifests for orchestration
- Implemented CI/CD pipelines with GitHub Actions for testing individual components, API testing, building Docker images, and pushing them to Docker Registry
- Developed a text processing pipeline for efficient document chunking and optimized vector search performance
- Created a user-friendly web interface for document upload, search, and question answering using Streamlit

### Recognizing Human Activities from Images
**Individual Project** | Oct 2023
- Worked with the Human Action Recognition benchmark from a Kaggle contest
- Proposed a modern approach by fine-tuning the vision-language model CLIP for an open-vocabulary detection task, replacing traditional fixed-category classification
- Reimplemented the training/inference pipeline, Trainer class with similar functionalities like Transformers library, just by PyTorch without using external training frameworks
- Integrated experiment tracking, data versioning, and model registry to streamline training experiments using MLflow
- Implemented a CI/CD pipeline with GitHub Actions to automate Docker image builds and deployments
- Deployed the model API using FastAPI and managed scalable infrastructure on Google Kubernetes Engine (GKE) using Terraform
- Created a user-friendly web interface for real-time human activity recognition using Streamlit

### Building AI Agents for Puzzle Games
**Team Project - Team Lead** | Dec 2022
- Led a team of four, responsible for coordinating overall project development and managing the project timeline
- Specialized in designing algorithmic solutions and creating mathematical models for the puzzle-solving agents
- Developed AI agents to solve Water Sort, Bloxorz, and Vietnamese Reversi games
- Engineered intelligent agents using A*, Minimax, Genetic Algorithms, and Deep Q-Learning (reinforcement learning) to optimize game strategy and performance
- Developed user interfaces and visualizations for game environments using PyGame, OpenGL and Cocos3d, while implementing the AI agents' logic and training with PyTorch for Deep Q-Learning

## 🏆 Honors and Awards

- Honors Degree Graduate with Dual AI Specializations and GPA 8.69/10 (Nov 2024)
- Student of Five Merits at Vietnam National University level and Ho Chi Minh City level (Nov 2024)
- Third Prize, Faculty Thesis Poster Competition For Talent Students (Top 3 Thesis) (May 2024)
- University Incentive Scholarship for Outstanding Students (Sep 2023)
- Odon Vallet Scholarship For Outstanding Vietnamese Students (Sep 2020)
- Third Prize, Vietnam Mathematical Olympiad (VMO) (Jan 2020)
- Consolation Prize, Vietnam Mathematical Olympiad (VMO) (Jan 2019)
- Gold Medal with Top 5, April 30th Mathematics Olympiad for Gifted Students in Southern Vietnam (Mar 2019)
- Gold Medal with Top 1, April 30th Mathematics Olympiad for Gifted Students in Southern Vietnam (Mar 2018)

## 🛠️ Skills

### Core Knowledge
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-5C3EE8?style=for-the-badge&logo=OpenCV&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-yellow?style=for-the-badge&logo=huggingface&logoColor=white)
![MLOps](https://img.shields.io/badge/MLOps-0078D4?style=for-the-badge&logo=azure-devops&logoColor=white)
![Statistics](https://img.shields.io/badge/Statistics-276DC3?style=for-the-badge&logo=r&logoColor=white)

### ML/DL Frameworks
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337357?style=for-the-badge&logo=xgboost&logoColor=white)
![Unsloth](https://img.shields.io/badge/Unsloth-4A154B?style=for-the-badge&logo=beast&logoColor=white)

### NLP Tools
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-yellow?style=for-the-badge&logo=huggingface&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154360?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-3178C6?style=for-the-badge&logo=chainlink&logoColor=white)
![Gensim](https://img.shields.io/badge/Gensim-FF9A00?style=for-the-badge&logo=python&logoColor=white)

### Computer Vision Tools
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![OpenGL](https://img.shields.io/badge/OpenGL-5586A4?style=for-the-badge&logo=opengl&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-4B8BBE?style=for-the-badge&logo=yolov5&logoColor=white)

### Large Model Serving Tools
![vLLM](https://img.shields.io/badge/vLLM-00ADD8?style=for-the-badge&logo=go&logoColor=white)
![lmdeploy](https://img.shields.io/badge/lmdeploy-00FFFF?style=for-the-badge&logo=buffer&logoColor=black)
![llama.cpp](https://img.shields.io/badge/llama.cpp-FF9E0F?style=for-the-badge&logo=llvm&logoColor=white)
![SGLang](https://img.shields.io/badge/SGLang-4EAA25?style=for-the-badge&logo=llvm&logoColor=white)
![Tensor-RT LLM](https://img.shields.io/badge/Tensor--RT%20LLM-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

### Engineering & DevOps
![Python](https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/kubernetes-%23326CE5.svg?style=for-the-badge&logo=kubernetes&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Terraform](https://img.shields.io/badge/terraform-%235835CC.svg?style=for-the-badge&logo=terraform&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=white)

### Languages
![Python](https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white)
![Java](https://img.shields.io/badge/java-%23ED8B00.svg?style=for-the-badge&logo=java&logoColor=white)
![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E)

### Soft Skills
- Critical Thinking
- Communication Skills
- Problem-Solving
- Time Management
- Leadership
- Quick Learning
- Languages: English (Professional working proficiency, TOEIC 760/990)

## 📜 Licenses and Certifications

### DevOps & Cloud
- DevOps Professional Certificate — PagerDuty & LinkedIn (Mar 2025)
- Building Cloud Computing Solutions at Scale Specialization — Duke University & Coursera (Aug 2024)

### AI/ML Operations
- Large Language Model Operations (LLMLOps) Specialization — Duke University & Coursera (Jun 2024)
- Machine Learning Operations (MLOps) Specialization — Duke University & Coursera (Jun 2024)
- Machine Learning Engineering for Production (MLOps) Specialization — DeepLearning.AI (Feb 2024)
- Vector Databases Professional Certificate — Weaviate (Jul 2024)

### Computer Vision
- Building Real-Time Video AI Applications — NVIDIA (Aug 2024)
- Generative AI with Diffusion Models — NVIDIA (Aug 2024)

### AI/ML Core
- Generative AI for Data Scientists Specialization — IBM (May 2024)
- Machine Learning Professional Certificate — IBM (May 2024)
- Advances In Natural Language Processing Specialization — VietAI & New Turing Institute (Mar 2024)
- Large Language Models Professional Certificate — Databricks (Oct 2023)
- Generative Adversarial Networks (GANs) Specialization — DeepLearning.AI (Jul 2023)
- AI Engineering Professional Certificate — IBM (Jul 2023)
- Natural Language Processing Specialization — DeepLearning.AI (Jul 2023)
- Deep Learning Specialization — DeepLearning.AI (Jun 2023)
- TensorFlow Developer Professional Certificate — DeepLearning.AI (Jun 2023)

## 📊 GitHub Stats

<picture>
<source 
  srcset="https://github-readme-stats.vercel.app/api?username=tuandung222&show_icons=true&theme=dark"
  media="(prefers-color-scheme: dark)"
/>
<source
  srcset="https://github-readme-stats.vercel.app/api?username=tuandung222&show_icons=true&theme=light"
  media="(prefers-color-scheme: light)"
/>
<img src="https://github-readme-stats.vercel.app/api?username=tuandung222&show_icons=true" />
</picture>

## 🔭 Featured Repositories

<a href="https://github.com/tuandung222/knowledge-distillation-coding-qa"> <img src="https://github-readme-stats.vercel.app/api/pin/?username=tuandung222&repo=knowledge-distillation-coding-qa" width=400> </a> 
<a href="https://github.com/tuandung222/semantic-search-rag"> <img src="https://github-readme-stats.vercel.app/api/pin/?username=tuandung222&repo=semantic-search-rag" width=400> </a> 
<a href="https://github.com/tuandung222/human-activity-recognition"> <img src="https://github-readme-stats.vercel.app/api/pin/?username=tuandung222&repo=human-activity-recognition" width=400> </a> 
<a href="https://github.com/tuandung222/ai-puzzle-agents"> <img src="https://github-readme-stats.vercel.app/api/pin/?username=tuandung222&repo=ai-puzzle-agents" width=400> </a> 

---

<p align="center">Thanks for visiting my profile! Feel free to reach out for collaborations.</p>
<p align="center">
  <img src="https://komarev.com/ghpvc/?username=tuandung222&style=for-the-badge"/>
  <img src="https://shields.io/github/stars/tuandung222?style=for-the-badge"/>
  <img src="https://img.shields.io/github/followers/tuandung222?style=for-the-badge"/>
</p>
