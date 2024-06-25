# JC'S Fine-tuned Language Models Repository

This repository hosts a collection of fine-tuned language models based on various base architectures. Our goal is to provide specialized models for different domains and applications.

## 0. :electric_plug: Requirement
To use the models in this repository, you'll need the following:

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.21.0+
- Other models are include in Notebook for different pre-trained models

## 1. :floppy_disk: Fine-Tuned Models

### 1.1. :rock: Mistral-7B-Geoscience (LLM)

- **Base Model**: Mistral-7B (LLM)
- **Training Data**: Nature Geoscience paper text (~30K samples)
- **Hugging Face Model**: [Link to the model on Hugging Face]
- **Description**: This model is fine-tuned on content from Nature Geoscience papers, enhancing its performance in geoscience-related tasks.
- **Performance**:
  *Question*: How to measure Hg concentration in a rock sample?
  *Answer*:
  `GPT-4`: "To measure Hg concentration in a rock sample, one typically uses instrumental methods like atomic absorption spectrometry (AAS) or inductively coupled plasma mass spectrometry (ICP-MS)....."
  
  `Pre-trained Mistral-7B`: "The mercury content of the samples is determined by atomic absorption spectrophotometry. The method involves dissolving 1 gm of the powdered sample in concentrated nitric acid and then adding an excess amount of potassium iodide solution (KI). This forms a complex with mercury, which can be measured at 253.7 nm wavelength using a flame atomizer....."
  
  `Mistral-7B-Geoscience`: "The analytical method used for the determination of Hg concentrations is based on cold vapor atomic fluorescence spectrometry (CVAFS). The CVAFS technique has been widely applied and validated as an accurate and precise method for measuring trace amounts of mercury in environmental samples."

### 1.2 :factory_worker: Future models~~~~~~~~~

## 2. :running: Fine-tuning Process

- **Load Data & Base Model**
- **Tokenize**
- **Padding**
- **Set up Lora in Model**
- **Train**
- **Evaluate**

## 3. Usage
Here's a basic python code of how to use the the fine-tuned LLM model:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_model_id = "[PRE-TRAINED MODELS]"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
)

eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)

ft_model = PeftModel.from_pretrained(base_model, "[FINE-TUNED MODEL CHECKPOINT NAME]")
```

For specific usage instructions for each model, please refer to their individual documentation.

## 4. :construction: Future Work
We plan to expand this repository with more fine-tuned models across various domains and applications. Some areas we're exploring include:

| Domain/Application                   | Description                                                                 |
|--------------------------------------|-----------------------------------------------------------------------------|
| :hospital: Healthcare                            | Predictive models for disease diagnosis and treatment outcomes               |
| :coin: Finance                               | Risk assessment, fraud detection, stock market prediction                   |
| 	:vertical_traffic_light:Transportation                       | Autonomous vehicles, traffic flow optimization                              |
| :shopping: Retail                                | Customer behavior analysis, demand forecasting                              |
| :book: Education                             | Adaptive learning systems, student performance prediction                   |
| :electron: Energy                                | Renewable energy optimization, smart grid management                        |
| :factory: Manufacturing                        | Predictive maintenance, quality control                                    |
| :bread: Agriculture                           | Crop yield prediction, pest management                                     |


## 5. Stay tuned for updates!
Contributing
We welcome contributions! If you're interested in contributing, please:

## 6. :fork_and_knife: Fork the repository
Create a new branch for your feature
Submit a pull request with a clear description of your changes
For major changes, please open an issue first to discuss what you would like to change. 

## 7. :iphone: Contact
For questions, issues, or collaborations, please contact me at :email: jcchen0331@gmail.com.
You can go to [my website](https://jcjunxing.github.io/) to learn more about myself. You kindness is so much apperciated. :kissing_heart:
