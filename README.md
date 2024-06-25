# JC'S Fine-tuned Language Models Repository

This repository hosts a collection of fine-tuned language models based on various base architectures. Our goal is to provide specialized models for different domains and applications.

## 0. :electric_plug: Requirement
To use the models in this repository, you'll need the following:

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.21.0+
- Other models are include in Notebook for different pre-trained models

## 1. :floppy_disk: Fine-Tuned Models

### 1.1. [Mistral-7B-Geoscience](#Mistral-7B-Geoscience-v0.1)
### 1.2. [Mistral-7B-Logging-Perforation](#Mistral-7B-Loggin-Perforation-v0.1)
### 1.3. :factory_worker: Future models~~~~~~~~~



## 3. Usage
Here's a basic python code of how to use the the fine-tuned LLM model:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_model_id = "[PRE-TRAINED MODELS NAME]"
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

----
----

## 8. Model Detail
### Mistral-7B-Geoscience-v0.1
- **Base Model**: Mistral-7B (LLM)
- **Training Data**: :rock: Nature Geoscience paper text (~30K samples)
- **Hugging Face Model**: [Link to the model on Hugging Face]
- **Description**: This model is fine-tuned on content from Nature Geoscience papers, enhancing its performance in geoscience-related tasks.
- **running: Fine-tuning Process**:

Load Data & Base Model :arrow_right: Tokenize :arrow_right: Padding :arrow_right: Set up Lora in Model :arrow_right: Train :arrow_right: Evaluate

- **Performance**:
  
  *Question*:

  How to measure Hg concentration in a rock sample?
  
  *Answer*:
  
  `GPT-4`: "To measure Hg concentration in a rock sample, one typically uses instrumental methods like atomic absorption spectrometry (AAS) or inductively coupled plasma mass spectrometry (ICP-MS)....."
  
  `Pre-trained Mistral-7B`: "The mercury content of the samples is determined by atomic absorption spectrophotometry. The method involves dissolving 1 gm of the powdered sample in concentrated nitric acid and then adding an excess amount of potassium iodide solution (KI). This forms a complex with mercury, which can be measured at 253.7 nm wavelength using a flame atomizer....."
  
  `Mistral-7B-Geoscience-v0.1`: "The analytical method used for the determination of Hg concentrations is based on cold vapor atomic fluorescence spectrometry (CVAFS). The CVAFS technique has been widely applied and validated as an accurate and precise method for measuring trace amounts of mercury in environmental samples."

----

### Mistral-7B-Logging-Perforation-v0.1
Base Model: Mistral-7B (LLM)
Training Data: :rock: SPE Journal abstract which focusing on Logging and Perforation  (~3K samples)
Hugging Face Model: [Link to the model on Hugging Face]
Description: This model is fine-tuned on content from the content in Petrolumn Engnieer journal, enhancing its performance in logging and perforation tasks related.
Fine-tuning Process:
Load Data & Base Model :arrow_right: Tokenize :arrow_right: Padding :arrow_right: Set up Lora in Model :arrow_right: Train :arrow_right: Evaluate

Performance:

Question:

Name 3 new technologies in Logging and Perforation

Answer:

`GPT-4`: "
Here are three succinct new technologies in logging and perforation:

1. Advanced Logging Sensors: Incorporating spectral logging and high-definition imaging for detailed analysis.

2. Real-time Data Transmission: Digital tools transmit data instantly for immediate analysis.

3.Engineered Perforating Systems: Utilizing shaped charge and precision systems for optimal wellbore perforation."

`Pre-trained Mistral-7B`: "
1. Electronic logging devices (ELDs) are electronic systems that record a driver's hours of service (HOS) information, including the date, time, location, engine hours, vehicle miles driven, and other data. ELDs help ensure compliance with HOS regulations by automatically recording driving time and preventing drivers from falsifying their logs.
2. Perforating guns are specialized tools used to create perforations in wellbores during drilling operations. These guns use high-pressure water or air jets to create small holes in the rock formation, allowing for the injection of fluids such as cement or drilling mud into the formation.
3. Smart perforators are advanced perforating tools that can be remotely controlled and monitored using telemetry technology. They allow operators to adjust the firing sequence and depth of penetration in real-time, improving accuracy and reducing the risk of damage to the wellbore.
"

`Mistral-7B-Geoscience-v0.1`: " The three new technologies are:
1. Distributed fiber-optic sensing (DFOS) technology for real-time monitoring of wellbore conditions during hydraulic fracturing operations. This technology uses optical fibers embedded in the wellbore to measure temperature, pressure, strain, and other parameters that can provide insights into the behavior of the rock formation and the effectiveness of the hydraulic fracturing process.
2. High-pressure perforating technology for creating larger and more uniform perforations in hard formations. This technology involves using high-pressure nitrogen or carbon dioxide gas to create larger perforations with less damage to the surrounding rock.
3. Smart completion systems for optimizing hydraulic fracturing treatments. These systems use sensors and actuators to monitor and control the flow of fluids and proppants into the wellbore during hydraulic fracturing. They allow for real-time adjustments to be made to the treatment based on downhole measurements, resulting in more efficient and effective fracturing treatments.
"

----
