# 🔍 Vision-Language Model Evaluation via OpenRouter.ai

## Overview

This repository presents an evaluation and comparison of state-of-the-art vision-language models using the [OpenRouter.ai](https://openrouter.ai) unified API. The goal of this project is to benchmark various models on visual reasoning and multimodal understanding tasks and identify the model that performs best in terms of response quality, reasoning depth, and contextual accuracy.

## 🧠 Models Evaluated

We conducted experiments with the following models:

| Model Name                            | Parameters | Modality       | Source           |
|--------------------------------------|------------|----------------|------------------|
| `mistral-small-3.1-24b-instruct`     | 24B        | Vision-Language| Mistral AI       |
| `opengvlab/internvl3-14b`            | 14B        | Vision-Language| OpenGVLab        |
| `moonshotai/kimi-vl-a3b-thinking`    | 3B         | Vision-Language| Moonshot AI      |

### Notable Findings

- ✅ **Best Performing Model**: `opengvlab/internvl3-14b`  
  This model demonstrated superior visual reasoning, nuanced comprehension of complex image-text relationships, and consistent performance across varied prompts.
- 📉 `kimi-vl-a3b-thinking`: While efficient and fast, the output lacked the semantic depth required for more involved vision tasks.
- 🧾 `mistral-small-3.1-24b-instruct`: Performed well on textual tasks but is not optimized for multimodal prompts.

## 📦 Installation

This repo assumes you are using Python and have access to OpenRouter API keys.

```bash
git clone https://github.com/your-username/vision-model-evaluation.git
cd vision-model-evaluation
pip install -r requirements.txt
🔐 OpenRouter API Integration
To use OpenRouter.ai models, set your API key as an environment variable:

bash
Copy
Edit
export OPENROUTER_API_KEY="your-api-key-here"
In your Python code, make calls to the API like this:

python
Copy
Edit
import openai

openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = os.getenv("OPENROUTER_API_KEY")

response = openai.ChatCompletion.create(
    model="opengvlab/internvl3-14b",
    messages=[{"role": "user", "content": "Describe the scene in this image."}],
    files=[open("image.png", "rb")]
)
print(response["choices"][0]["message"]["content"])
📊 Evaluation Criteria
Each model was tested on a common benchmark involving:

Scene description tasks

Visual question answering (VQA)

Image reasoning and captioning

Multimodal dialogue

The results were scored based on:

Relevance and accuracy of output

Logical reasoning

Factual correctness

Consistency and completeness



📌 Conclusion
Through a structured evaluation, we identified InternVL3-14B as the most capable model for complex visual-language reasoning tasks. It outperforms others in handling rich visual inputs and producing coherent, high-quality responses.
