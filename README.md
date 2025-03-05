<h1 align='center'>
Shakespearean Sparks: The Dance of Hallucination and Creativity in LLMs' Decoding Layers
</h1>

<p align='center'>
  <a href="https://arxiv.org/abs/2503.02851">
    <img src="https://img.shields.io/badge/arXiv-2503.02851-b31b1b.svg" alt="ArXiv">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  </a>
</p>

</p>

This repository contains the source codes for reproducing the results of the paper: [**Shakespearean Sparks: The Dance of Hallucination and Creativity in LLMs' Decoding Layers**]().

**Author List**: *Zicong He, *Boxuan Zhang, Lu Cheng.

(* Equal Contribution)

## Overview
This project explores the intricate relationship between **hallucination** and **creativity** in Large Language Models (LLMs) at different decoding layers. While hallucination is often considered a flaw, our study provides a **quantitative perspective** that reveals its potential contribution to **creative outputs**.

To systematically investigate this relationship, we propose **HCL (Hallucination-Creativity Layerwise framework)**, which:
- **Quantifies** hallucination and creativity across different model layers.
- **Identifies** the tradeoff between creativity and factual accuracy.
- **Determines** the optimal decoding layer that balances both aspects.

Our findings suggest that earlier layers in LLMs **tend to produce more creative outputs**, while deeper layers prioritize factual accuracy. Leveraging this insight, we introduce a **layer-wise early-exit strategy** to enhance **computational efficiency** without sacrificing quality.

## Getting Started

### Clone repo:
```bash
git clone git@github.com:ZicongHe2002/Shakespearean-Sparks-The-Dance-of-Hallucination-and-Creativity-in-LLMs-Decoding-Layers.git
cd code

```

### Setup environment:
```bash
conda create --name hcl_spark python=3.10
conda activate hcl_spark
pip install -r requirements.txt
```

### Run
```bash
./generate0.sh
```
## Methodology
Our research is built upon a **three-stage evaluation process**:
1. **Layer-wise Response Sampling**: Using an early-exit strategy to extract responses at different layers.
2. **Evaluation Metrics**: Creativity is measured as the **semantic diversity** of correct responses, while hallucination is assessed by **error rates**.
3. **HCB Calculation**: We introduce the **Hallucination-Creativity Balanced (HCB) score**, which helps identify the optimal decoding layer for improved model performance.

## Key Findings
- **Creativity comes with hallucination**: Models with higher creativity scores also exhibit a greater tendency for hallucination.
- **Stronger models generate more creative, yet more hallucinatory, responses**: Larger LLMs tend to balance this tradeoff better at **intermediate layers**.
- **Final layer decoding isn’t always optimal**: Selecting outputs from **earlier layers** can yield a better balance between **diversity and factuality**.
- **Optimal layers are model-dependent but consistent across tasks**: Our results generalize across different LLM architectures and datasets.

## Implementation
**Access models**: In order to observe speedup, you need to access LLMs that have been trained using the LayerSkip recipe. We provide 4 checkpoints on [HuggingFace](https://huggingface.co/collections/facebook/layerskip-666b25c50c8ae90e1965727a) of different Llama models continually pretrained using the LayerSkip recipe:

We conduct experiments using **open-weight LLMs**, including:
- **LLaMA 2-7B**
- **LLaMA 2-13B**
- **LLaMA 3.2-1B**
- **LLaMA 3-8B**

Our dataset sources include **TriviaQA** and **Natural Questions (NQ)**, ensuring a diverse benchmark for creativity and hallucination evaluation.

## Acknowledgements
We sincerely thank the following authors, and HCL is based on their excellent open-source projects or impressive ideas.

**Layerskip**:   https://github.com/facebookresearch/LayerSkip?tab=readme-ov-file

## Repository Structure
