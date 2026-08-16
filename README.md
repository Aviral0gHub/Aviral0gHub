<div align="center">

  <!-- Animated Header Title -->
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=34&duration=2800&pause=1000&color=4CAF50&center=true&vCenter=true&width=800&lines=Hi,+I'm+Aviral+Nigam;LLM+Factuality+%2B+Retrieval+%2B+CV;Co-Author,+Last+Translation+Benchmark" alt="Typing Header" />

  <p align="center">
    <b>Data Science & Engineering Undergraduate @ MIT Manipal</b>
  </p>

  <!-- Interactive Pill Badges -->
  <p align="center">
    <a href="https://aviral0ghub.github.io/"><img src="https://img.shields.io/badge/Portfolio-101010?style=for-the-badge&logo=googlechrome&logoColor=4CAF50" alt="Website"/></a>&nbsp;
    <a href="https://www.linkedin.com/in/aviral-nigam-a0b469271"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/></a>&nbsp;
    <a href="mailto:nigamaviral21@gmail.com"><img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Gmail"/></a>&nbsp;
    <a href="https://github.com/Aviral0gHub/Aviral0gHub/raw/main/Aviral_Nigam_Resume.pdf"><img src="https://img.shields.io/badge/Download_CV-4CAF50?style=for-the-badge&logo=readdotcv&logoColor=white" alt="Download CV"/></a>
  </p>

  <!-- Waving Header Art -->
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4CAF50&height=120&section=header&text=Building%20Systems%20That%20Verify%20What%20They%20Claim&fontSize=22&fontColor=ffffff&animation=fadeIn" width="100%" />

</div>

<br/>

<!-- ==================== EXECUTIVE PROFILE ==================== -->
<table align="center" width="100%">
  <tr>
    <td width="55%" valign="top">
      <h3>Quick Overview</h3>
      <ul>
        <li><b>Education:</b> B.Tech in Data Science & Engineering, MIT Manipal (expected 2028)</li>
        <li><b>Research:</b> Co-Author, <i>Last Translation Benchmark</i> (expected release Sept 2026)</li>
        <li><b>Past Role:</b> Research Intern @ Robotics Research Center (RRC), IIIT-H, advised by Dr. Sourav Garg (May–Aug 2026)</li>
        <li><b>General Focus:</b> LLM factuality & evaluation, retrieval systems, and computer vision</li>
        <li><b>Beyond Tech:</b> Playing keyboard (Bollywood classics)</li>
      </ul>
    </td>
    <td width="45%" valign="top">
      <h3>Core Focus Areas</h3>
      <ul>
        <li><b>LLM Factuality & Agentic Reasoning:</b> Claim-level verification pipelines, retrieval-augmented evidence checking, and NLI-based entailment scoring.</li>
        <li><b>Computer Vision & Representation Learning:</b> Fine-grained visual instance retrieval, feature matching, and metric learning architectures.</li>
        <li><b>Machine Learning & Automated Pipelines:</b> End-to-end regression/classification pipelines, vector search (FAISS) indexing, and model evaluation.</li>
      </ul>
    </td>
  </tr>
</table>

<br/>

<!-- ==================== INTERACTIVE PROJECTS & RESEARCH ==================== -->
<h2 align="center">Research & Featured Projects</h2>

<details open>
  <summary><b>Last Translation Benchmark (LTB)</b> <i>— Multimodal Factuality Benchmark, Co-Author</i></summary>
  <br/>
  <p>
    <a href="https://last-translation-benchmark.vilda.net/"><b>Project Site</b></a>
  </p>
  <ul>
    <li>Co-authoring an upcoming multilingual, multimodal factuality benchmark with researchers from <b>ETH Zurich (Vilém Zouhar), JHU, CUNI, UvA, and KIT</b>. Expected release: September 2026.</li>
    <li>Contributed adversarial multimodal inputs specifically designed to break state-of-the-art vision-language models.</li>
  </ul>
</details>

<details open>
  <summary><b>FactTrace</b> <i>— Claim-Level Factuality Verification Pipeline</i></summary>
  <br/>
  <p>
    <a href="https://github.com/Aviral0gHub/facttrace"><b>View Project Repository on GitHub</b></a>
  </p>
  <ul>
    <li>Built a pipeline that decomposes agentic LLM reasoning traces into atomic claims, retrieves evidence via FAISS-based dense retrieval, and classifies each claim as supported, contradicted, or unverifiable using NLI-based entailment scoring.</li>
    <li>Improved claim-level accuracy from 78.5% (lexical baseline) to 87.7% by replacing regex-based heuristics with dense retrieval and RoBERTa-large-MNLI entailment scoring, evaluated on a 65-claim curated benchmark.</li>
    <li>Failure analysis showed unsupported claims concentrate disproportionately in later reasoning steps, surfacing where agentic reasoning chains drift from grounded evidence.</li>
  </ul>
</details>

<details open>
  <summary><b>CalibCheck</b> <i>— LLM Confidence Calibration Analysis</i></summary>
  <br/>
  <p>
    <a href="https://github.com/Aviral0gHub/calibcheck"><b>View Project Repository on GitHub</b></a>
  </p>
  <ul>
    <li>Built a calibration analysis pipeline measuring whether an LLM's confidence tracks its actual accuracy, comparing verbalized self-reported confidence against raw next-token probability using Expected Calibration Error, Brier score, and reliability diagrams across 6 topic domains.</li>
    <li>Found Qwen2.5-0.5B-Instruct overconfident by 14.3 points on average (ECE = 0.14) on a 60-question benchmark, with domain-specific miscalibration reaching 57.5 points on sports questions (30% actual accuracy vs. 87.5% stated confidence).</li>
  </ul>
</details>

<details open>
  <summary><b>Robotics Research Center (RRC), IIIT Hyderabad</b> <i>— Research Internship (May–Aug 2026)</i></summary>
  <br/>
  <ul>
    <li>Researched fine-grained visual instance retrieval, building deep neural pipelines across large-scale datasets.</li>
    <li>Implemented high-throughput vector search (FAISS) for efficient image indexing and feature retrieval.</li>
    <li>Engineered automated embedding extraction & normalization pipelines for metric learning architectures.</li>
  </ul>
</details>

<details open>
  <summary><b>Learned Vegetation Index (LVI)</b> <i>— Deep Regression Pipeline</i></summary>
  <br/>
  <p>
    <a href="https://github.com/Aviral0gHub/Learned-Vegetation-Index"><b>View Project Repository on GitHub</b></a>
  </p>
  <ul>
    <li>Engineered a deep regression architecture mapping multispectral visual features to ground-truth LIDAR physical measurements across 78K+ geo-aligned samples.</li>
    <li>Achieved 6.62m RMSE ($R^2 = 0.592$) using PyTorch and XGBoost, outperforming standard handcrafted baseline features by 23%.</li>
  </ul>
</details>

<details>
  <summary><b>Multi-Class Land Cover Classification</b> <i>— Computer Vision & Transfer Learning</i></summary>
  <br/>
  <p>
    <a href="https://github.com/Aviral0gHub/eurosat-landcover-classification"><b>View Project Repository on GitHub</b></a>
  </p>
  <ul>
    <li>Developed an end-to-end visual classification pipeline leveraging transfer learning with ResNet-18 backbones across a 27,000-image dataset.</li>
    <li>Designed custom computer vision augmentations using OpenCV, achieving 96.98% classification accuracy ($0.97$ Macro F1).</li>
  </ul>
</details>

<br/>

<!-- ==================== TECH STACK MATRIX ==================== -->
<h2 align="center">Technical Capability Matrix</h2>

<table align="center" width="100%">
  <tr>
    <td width="25%"><b>NLP & LLM Evaluation</b></td>
    <td>
      <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
      <img src="https://img.shields.io/badge/Sentence--Transformers-6A5ACD?style=for-the-badge" />
      <img src="https://img.shields.io/badge/NLI_%2F_Entailment-6A5ACD?style=for-the-badge" />
    </td>
  </tr>
  <tr>
    <td width="25%"><b>DL & Computer Vision</b></td>
    <td>
      <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
      <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
      <img src="https://img.shields.io/badge/FAISS-00599C?style=for-the-badge" />
      <img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
    </td>
  </tr>
  <tr>
    <td width="25%"><b>Machine Learning</b></td>
    <td>
      <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
      <img src="https://img.shields.io/badge/XGBoost-111111?style=for-the-badge&logo=xgboost&logoColor=white" />
      <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
      <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
    </td>
  </tr>
  <tr>
    <td width="25%"><b>Languages</b></td>
    <td>
      <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
      <img src="https://img.shields.io/badge/C%2B%2B-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white" />
      <img src="https://img.shields.io/badge/Java-ED8B00?style=for-the-badge&logo=openjdk&logoColor=white" />
      <img src="https://img.shields.io/badge/SQL-003B57?style=for-the-badge&logo=postgresql&logoColor=white" />
      <img src="https://img.shields.io/badge/Bash-4EAA25?style=for-the-badge&logo=gnu-bash&logoColor=white" />
    </td>
  </tr>
  <tr>
    <td width="25%"><b>Systems & Tools</b></td>
    <td>
      <img src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black" />
      <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
      <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white" />
      <img src="https://img.shields.io/badge/VS_Code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white" />
    </td>
  </tr>
</table>

<br/>

<!-- ==================== LIVE METRICS & ACTIVITY ==================== -->
<h2 align="center">Activity & Performance</h2>

<p align="center">
  <img src="https://github-readme-stats.vercel.app/api?username=Aviral0gHub&show_icons=true&theme=algolia&hide_border=true&count_private=true" width="48%" />
  <img src="https://github-readme-streak-stats.herokuapp.com/?user=Aviral0gHub&theme=algolia&hide_border=true" width="48%" />
</p>

<!-- ==================== ANIMATED FOOTER ==================== -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4CAF50&height=100&section=footer" width="100%" />
</p>
