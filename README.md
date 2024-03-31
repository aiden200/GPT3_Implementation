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

This project focuses on training a decoder only Transformer model to predict characters in a sequence, demonstrating a simple form of language understanding and generation. Below are the hyperparameters used in our training setup:

- **Max Length (`max_length`)**: The maximum sequence length is set to 256. This defines how many previous characters the model looks at to predict the next character.
- **Batch Size (`batch_size`)**: We use a batch size of 64, determining how many sequences are processed together in a single training step.
- **Learning Rate (`lr`)**: The learning rate for the optimizer is set at 3e-4. This value controls the speed at which the model learns.
- **Device**: Training is performed on a CUDA-enabled GPU if available, otherwise on CPU. This is automatically determined by the script.
- **Epochs (`epochs`)**: The model will train for 5000 epochs. Each epoch is a complete pass over the entire training dataset.
- **Evaluation Interval (`eval_interval`)**: Evaluation on a validation set occurs every 500 epochs to monitor the model's performance.
- **Embedding Size (`embed_size`)**: The size of the embedding vectors is 384.
- **Attention Heads (`attention_heads`)**: Our model uses 6 attention heads in the multi-head attention mechanism.
- **Dropout (`dropout`)**: A dropout rate of 0.2 is used to prevent overfitting by randomly setting a fraction of the input units to 0 during training.


To train the model, run:
```bash
python train.py
```

## Training Data
We use a text dataset for training, focusing on character-by-character prediction. Our dataset comprises dialogues from a play, where each character's speech acts as a sequence for the model to learn from. The model learns to predict each character based on the sequence of characters that precedes it, effectively learning the structure and pattern of the language in the dataset. There are around 10k characters.

## Credits
This project was inspired by the work of Andrej Karpathy, particularly his ["makemore" series](https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy) on YouTube. His teachings on autoregressive language modeling and PyTorch are the best i've seen.

Additionally, this project owes its inception to the foundational works of [Attention is All You Need](https://arxiv.org/abs/1706.03762) and the development of GPT-2 and GPT-3 by OpenAI.