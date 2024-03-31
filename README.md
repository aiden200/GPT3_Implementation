# Generatively Pretrained Transformer Project

This project is dedicated to building a Generatively Pretrained Transformer (GPT) inspired by the seminal paper "Attention is All You Need" and the advancements seen in OpenAI's GPT-2 and GPT-3 models. Our journey explores the intricate connections to ChatGPT, a model that has significantly impacted natural language processing and understanding. In a twist of meta-learning, we've also incorporated GitHub Copilot, a tool based on GPT, to assist in developing parts of this GPT model.

## Background

The project is grounded in the fundamentals of autoregressive language modeling, leveraging the power of tensors and the PyTorch neural network (nn) framework. For those new to these concepts, I highly recommend watching the "makemore" video series by Andrej Karpathy, which provides an excellent introduction to language models and PyTorch.

## Repository Structure

- **data/**: Contains datasets used for training the model.
- **model/**: The core transformer model implementation.
- **src/**: Source code for the project, including utility functions and model components.
- **utils/**: Utility scripts for data preprocessing, model evaluation, and other tasks.
- **.gitignore**: Specifies intentionally untracked files to ignore.
- **LICENSE**: The project is open-sourced under the MIT license.
- **README.md**: This document.
- **evaluate.py**: Script for evaluating the trained model's performance.
- **tokenize.ipynb**: Jupyter notebook detailing the tokenization process.
- **train.py**: Script for training the GPT model.

## Getting Started

To get started with this project, clone the repository and install the required dependencies (listed in a requirements.txt file you should create based on your project's dependencies).

```bash
git clone https://github.com/aiden200/smallLLM.git
cd smallLLM
pip install -r requirements.txt
```

## Training the Model

To train the model, run:
```bash
python train.py
```

## Credits
This project was inspired by the work of Andrej Karpathy, particularly his "makemore" series on YouTube. His teachings on autoregressive language modeling and PyTorch are the best i've seen.

Additionally, this project owes its inception to the foundational works of [Attention is All You Need](https://arxiv.org/abs/1706.03762) and the development of GPT-2 and GPT-3 by OpenAI.