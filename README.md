<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0D1117,100:4CAF50&height=200&section=header&text=Aviral%20Nigam&fontSize=60&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Building%20Systems%20That%20Verify%20What%20They%20Claim&descAlignY=55&descSize=18" width="100%"/>

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=26&duration=2800&pause=1000&color=4CAF50&center=true&vCenter=true&width=700&lines=Data+Science+%26+Engineering+%40+MIT+Manipal;LLM+Factuality+%2B+Retrieval+%2B+Computer+Vision;Co-Author%2C+Last+Translation+Benchmark" alt="Typing SVG" />

<br/>

<a href="https://aviral0ghub.github.io/"><img src="https://img.shields.io/badge/Portfolio-0D1117?style=for-the-badge&logo=googlechrome&logoColor=4CAF50" /></a>
<a href="https://www.linkedin.com/in/aviral-nigam-a0b469271"><img src="https://img.shields.io/badge/LinkedIn-0D1117?style=for-the-badge&logo=linkedin&logoColor=0A66C2" /></a>
<a href="mailto:nigamaviral21@gmail.com"><img src="https://img.shields.io/badge/Gmail-0D1117?style=for-the-badge&logo=gmail&logoColor=EA4335" /></a>

<br/><br/>

<img src="https://komarev.com/ghpvc/?username=Aviral0gHub&label=Profile%20Views&color=4CAF50&style=for-the-badge" />
<img src="https://img.shields.io/badge/Focus-Trustworthy%20ML-0D1117?style=for-the-badge&labelColor=4CAF50&color=0D1117" />
<img src="https://img.shields.io/badge/Status-Open%20to%20Research%20Collabs-0D1117?style=for-the-badge&labelColor=4CAF50&color=0D1117" />

</div>

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4CAF50,100:0D1117&height=3&width=100%" />

## 📋 Quick Overview

<table align="center" width="100%">
  <tr>
    <td width="50%" valign="top">
      <ul>
        <li>🎓 <b>Education:</b> B.Tech in Data Science &amp; Engineering, MIT Manipal <i>(expected 2028)</i></li>
        <li>📄 <b>Research:</b> Co-Author, <i>Last Translation Benchmark</i> <i>(expected release Sept 2026)</i></li>
        <li>🔬 <b>Past Role:</b> Research Intern @ Robotics Research Center (RRC), IIIT-H, advised by Dr. Sourav Garg <i>(May–Aug 2026)</i></li>
        <li>🧭 <b>General Focus:</b> LLM factuality &amp; evaluation, retrieval systems, and computer vision</li>
        <li>🎹 <b>Beyond Tech:</b> Playing keyboard (Bollywood classics)</li>
      </ul>
    </td>
    <td width="50%" valign="top">
      <ul>
        <li>🧠 <b>LLM Factuality &amp; Agentic Reasoning:</b> Claim-level verification pipelines, retrieval-augmented evidence checking, and NLI-based entailment scoring.</li>
        <li>👁️ <b>Computer Vision &amp; Representation Learning:</b> Fine-grained visual instance retrieval, feature matching, and metric learning architectures.</li>
        <li>⚙️ <b>Machine Learning &amp; Automated Pipelines:</b> End-to-end regression/classification pipelines, vector search (FAISS) indexing, and model evaluation.</li>
      </ul>
    </td>
  </tr>
</table>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4CAF50,100:0D1117&height=3&width=100%" />

## 🔬 Research & Featured Projects

<details open>
<summary>🌐 &nbsp;<b>Last Translation Benchmark (LTB)</b> — <i>Multimodal Factuality Benchmark, Co-Author</i></summary>
<br/>

**[🔗 Project Site](https://last-translation-benchmark.vilda.net/)**

- Co-authoring an upcoming multilingual, multimodal factuality benchmark with researchers from **ETH Zurich (Vilém Zouhar), JHU, CUNI, UvA, and KIT**. Expected release: September 2026.
- Contributed adversarial multimodal inputs specifically designed to break state-of-the-art vision-language models.

</details>

<details open>
<summary>✅ &nbsp;<b>FactTrace</b> — <i>Claim-Level Factuality Verification Pipeline</i></summary>
<br/>

**[🔗 View Repository](https://github.com/Aviral0gHub/facttrace)** &nbsp;·&nbsp; `Python` `PyTorch` `sentence-transformers` `HuggingFace Transformers` `FAISS`

- Built a pipeline that decomposes agentic LLM reasoning traces into atomic claims, retrieves evidence via FAISS-based dense retrieval, and classifies each claim as supported, contradicted, or unverifiable using NLI-based entailment scoring.
- Improved claim-level accuracy from **78.5% → 87.7%** by replacing regex-based heuristics with dense retrieval and RoBERTa-large-MNLI entailment scoring, evaluated on a 65-claim curated benchmark.
- Failure analysis showed unsupported claims concentrate disproportionately in later reasoning steps, surfacing where agentic reasoning chains drift from grounded evidence.

</details>

<details open>
<summary>📊 &nbsp;<b>CalibCheck</b> — <i>LLM Confidence Calibration Analysis</i></summary>
<br/>

**[🔗 View Repository](https://github.com/Aviral0gHub/calibcheck)** &nbsp;·&nbsp; `Python` `HuggingFace Transformers` `Matplotlib`

- Built a calibration analysis pipeline measuring whether an LLM's confidence tracks its actual accuracy, comparing verbalized self-reported confidence against raw next-token probability using Expected Calibration Error, Brier score, and reliability diagrams across 6 topic domains.
- Found Qwen2.5-0.5B-Instruct overconfident by **14.3 points** on average (ECE = 0.14) on a 60-question benchmark, with domain-specific miscalibration reaching **57.5 points** on sports questions (30% actual accuracy vs. 87.5% stated confidence).

</details>

<details open>
<summary>🤖 &nbsp;<b>Robotics Research Center (RRC), IIIT Hyderabad</b> — <i>Research Internship (May–Aug 2026)</i></summary>
<br/>

- Researched fine-grained visual instance retrieval, building deep neural pipelines across large-scale datasets.
- Implemented high-throughput vector search (FAISS) for efficient image indexing and feature retrieval.
- Engineered automated embedding extraction & normalization pipelines for metric learning architectures.

</details>

<details open>
<summary>🌿 &nbsp;<b>Learned Vegetation Index (LVI)</b> — <i>Deep Regression Pipeline</i></summary>
<br/>

**[🔗 View Repository](https://github.com/Aviral0gHub/Learned-Vegetation-Index)** &nbsp;·&nbsp; `PyTorch` `XGBoost`

- Engineered a deep regression architecture mapping multispectral visual features to ground-truth LIDAR physical measurements across 78K+ geo-aligned samples.
- Achieved **6.62m RMSE** (R² = 0.592) using PyTorch and XGBoost, outperforming standard handcrafted baseline features by 23%.

</details>

<details>
<summary>🛰️ &nbsp;<b>Multi-Class Land Cover Classification</b> — <i>Computer Vision &amp; Transfer Learning</i></summary>
<br/>

**[🔗 View Repository](https://github.com/Aviral0gHub/eurosat-landcover-classification)** &nbsp;·&nbsp; `PyTorch` `ResNet-18` `OpenCV`

- Developed an end-to-end visual classification pipeline leveraging transfer learning with ResNet-18 backbones across a 27,000-image dataset.
- Designed custom computer vision augmentations using OpenCV, achieving **96.98%** classification accuracy (0.97 Macro F1).

</details>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4CAF50,100:0D1117&height=3&width=100%" />

## 🧰 Technical Toolkit

<div align="center">

**NLP & LLM Evaluation**
<br/>
<img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black" />
<img src="https://img.shields.io/badge/Sentence--Transformers-6A5ACD?style=flat-square" />
<img src="https://img.shields.io/badge/NLI%20%2F%20Entailment-6A5ACD?style=flat-square" />
<img src="https://img.shields.io/badge/Claim%20Decomposition-6A5ACD?style=flat-square" />
<img src="https://img.shields.io/badge/Calibration%20Analysis-6A5ACD?style=flat-square" />

<br/><br/>

**Deep Learning & Computer Vision**
<br/>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white" />
<img src="https://img.shields.io/badge/FAISS-00599C?style=flat-square" />
<img src="https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white" />

<br/><br/>

**Machine Learning**
<br/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
<img src="https://img.shields.io/badge/XGBoost-111111?style=flat-square&logo=xgboost&logoColor=white" />
<img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
<img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" />

<br/><br/>

**Languages**
<br/>
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/C%2B%2B-00599C?style=flat-square&logo=c%2B%2B&logoColor=white" />
<img src="https://img.shields.io/badge/Java-ED8B00?style=flat-square&logo=openjdk&logoColor=white" />
<img src="https://img.shields.io/badge/SQL-003B57?style=flat-square&logo=postgresql&logoColor=white" />
<img src="https://img.shields.io/badge/Bash-4EAA25?style=flat-square&logo=gnu-bash&logoColor=white" />

<br/><br/>

**Systems & Tools**
<br/>
<img src="https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=linux&logoColor=black" />
<img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white" />
<img src="https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white" />
<img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white" />
<img src="https://img.shields.io/badge/VS%20Code-007ACC?style=flat-square&logo=visualstudiocode&logoColor=white" />
<img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white" />

</div>

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4CAF50,100:0D1117&height=3&width=100%" />

## 📈 Activity & Performance

<p align="center">
  <img src="https://github-readme-stats.vercel.app/api?username=Aviral0gHub&show_icons=true&hide_border=true&bg_color=00000000&title_color=4CAF50&icon_color=4CAF50&text_color=c9d1d9&count_private=true" width="49%" />
  <img src="https://github-readme-streak-stats.herokuapp.com/?user=Aviral0gHub&hide_border=true&background=00000000&ring=4CAF50&fire=4CAF50&currStreakLabel=4CAF50&sideLabels=c9d1d9&currStreakNum=c9d1d9&sideNums=c9d1d9&dates=6e7681" width="49%" />
</p>

<p align="center">
  <img src="https://github-readme-stats.vercel.app/api/top-langs/?username=Aviral0gHub&layout=compact&hide_border=true&bg_color=00000000&title_color=4CAF50&text_color=c9d1d9&langs_count=8" width="45%" />
</p>

<p align="center">
  <img src="https://github-profile-trophy.vercel.app/?username=Aviral0gHub&theme=algolia&no-frame=true&no-bg=true&row=1&column=6&margin-w=8" />
</p>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0D1117,100:4CAF50&height=120&section=footer&width=100%" />

<div align="center">
  <sub>Thanks for stopping by — always happy to talk trustworthy ML, retrieval, or agentic reasoning. 🚀</sub>
</div>
