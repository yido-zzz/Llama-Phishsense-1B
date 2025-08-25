---
base_model:
- meta-llama/Llama-Guard-3-1B
datasets:
- ealvaradob/phishing-dataset
language:
- en
license: llama3.2
metrics:
- accuracy
- precision
- recall
library_name: transformers
---
# Revolutionize Phishing Protections with the Shrewd's Llama-Phishsense-1B!
<!-- ![image/png](https://cdn-uploads.huggingface.co/production/uploads/67097b2367976b94cabc116c/v8kIbeAx9WIuOQs4lf8XT.png) -->
![image/png](https://cdn-uploads.huggingface.co/production/uploads/67097b2367976b94cabc116c/UHcCZN-2PMvkq7DvbJt9z.png)

Phishing attacks are constantly evolving, targeting businesses and individuals alike. What if you could deploy a **highly efficient/effective**, **AI-powered defense system** that proactively identifies these threats and safeguards your inbox? 
* Enter the **Shrewd's AcuteShrewdSecurity/Llama-Phishsense-1B**— your new secret SOTA (finetuned Llama-Guard-3-1B) defense to combat phishing. It's trained to sense phishing. 

_PS: it's small enough to be used anywhere, and is a model trained to have the phishing detection sense. [See Launch Post here](https://medium.com/@b1oo/introducing-llama-phishsense-1b-your-ai-powered-phishing-defense-7349765d144e) and paper [here](https://arxiv.org/abs/2503.10944)_.

# Why Phishing is a Growing Threat
Phishing is no longer just a concern for individuals; it’s an enterprise-level threat. **MANY of cyberattacks begin with phishing emails** aimed at compromising valuable data. Malicious actors craft increasingly deceptive messages, making it difficult for even the most vigilant people to distinguish between real and fraudulent emails. 

The results? **Billions in financial losses**, compromised personal and professional accounts, and reputational damage.

# The Solution: AI-Powered Phishing Detection
Traditional security systems struggle to keep pace with modern phishing tactics. That’s where AI comes in. The `Llama-Phishsense-1B` is designed to:
- Automatically detect **phishing patterns** in real-time.
- Protect your organization from **costly breaches**.
- **Empower people** to confidently navigate their inbox, knowing they are safeguarded.

# Join the Movement for Better Cybersecurity
Our initiative is more than just another AI tool—it’s a step toward **global cyber resilience**. By leveraging the latest advances in **Low-Rank Adaptation (LoRA)**, the `AcuteShrewdSecurity/Llama-Phishsense-1B` model is designed to identify phishing attempts with **minimal resources**, making it fast and efficient without sacrificing accuracy.

<!-- The best part? **This model is free and accessible to everyone**—corporate or individual. Whether you’re protecting sensitive company data or your personal accounts, this model can be your first line of defense.
 -->
# Why You Should Use This Model

### 1. **Protect Against Corporate Enterprise Phishing**
In a corporate setting, phishing emails can look legitimate and may easily bypass traditional filters. Attackers specifically tailor their messages to target people, especially those in finance, HR, or IT. The `AcuteShrewdSecurity/Llama-Phishsense-1B` can be integrated into your **corporate email system** to act as an additional layer of protection:
- **Mitigate risks** of people-targeted phishing attacks.
- Prevent unauthorized access to sensitive information.
- **Reduce downtime** associated with recovering from successful phishing exploits.

### 2. **Individual Use Case**
For individuals, managing personal information is more crucial than ever. Phishing emails that appear to be from legitimate services, such as online banking or social networks, can easily slip through basic email filters. This model:
- **Identifies phishing attempts** before you even open the email.
- Provides a **clear 'TRUE' or 'FALSE' prediction** on whether an email is safe.
- **Gives peace of mind** knowing your private data is secure.

### 3. **Offer Phishing Protection as a Service**
For security professionals and IT providers, integrating `Llama-Phishsense-1B` into your security offerings can give clients an added layer of **reliable, AI-driven protection**:
- Add this model to your existing cybersecurity stack.
- **Increase client satisfaction** by offering a proven phishing detection system.
- Help clients **avoid costly breaches** and maintain operational efficiency.

# Model Description

The `Llama-Phishsense-1B` is a fine-tuned version of `meta-llama/Llama-Guard-3-1B`, enhanced to handle phishing detection specifically within corporate email environments. Through advanced **LoRA-based fine-tuning**, it classifies emails as either "TRUE" (phishing) or "FALSE" (non-phishing), offering lightweight yet powerful protection against the ever-growing threat of email scams.

## Key Features:

- **Base Model**: ```meta-llama/Llama-Guard-3-1B and meta-llama/Llama-3.2-1B```
- **LoRA Fine-tuning**: Efficient adaptation using Low-Rank Adaptation for quick, resource-friendly deployment.
- **Task**: Binary email classification—phishing (TRUE) or non-phishing (FALSE).
- **Dataset**: A custom-tailored phishing email dataset, featuring real-world phishing and benign emails.
- **Model Size**: 1 Billion parameters, ensuring robust performance without overburdening resources.
- **Architecture**: Causal Language Model with LoRA-adapted layers for speed and efficiency.

## Why Choose This Model?

Phishing is responsible for the majority of security breaches today. The `Llama-Phishsense-1B` model is your answer to this problem:
- **Highly Accurate**: The model has achieved outstanding results in real-world evaluations, with an **F1-score of 0.99** on balanced datasets.
- **Fast and Efficient**: Leveraging LoRA fine-tuning, it operates faster while requiring fewer computational resources, meaning you get top-notch protection without slowing down your systems.
- **Accessible to Everyone**: Whether you're a IT team or a solo email user, this tool is designed for easy integration and use.

# Training and Fine-tuning:

### LoRA Configuration:
- **Rank**: `r=16`
- **Alpha**: `lora_alpha=32`
- **Dropout**: `lora_dropout=0.1`
- Adapted on the **q_proj** and **v_proj** transformer layers for efficient fine-tuning.

### Training Data:
The model was fine-tuned on a **balanced dataset** of phishing and non-phishing emails (30k each), selected from `ealvaradob/phishing-dataset` to ensure real-world applicability.

### Optimizer:
- **AdamW Optimizer**: Weight decay of `0.01` with a learning rate of `1e-3`.

### Training Configuration:
- **Mixed-precision (FP16)**: Enables faster training without sacrificing accuracy.
- **Gradient accumulation steps**: 10.
- **Batch size**: 10 per device.
- **Number of epochs**: 10.

## Performance (Before and After finetuning):
Our model has demonstrated its effectiveness across multiple datasets (evals from ```zefang-liu/phishing-email-dataset```, and custom created):

| Metric    | Base Model (meta-llama/Llama-Guard-3-1B) | Finetuned Model (AcuteShrewdSecurity/Llama-Phishsense-1B) | Performance Gain (Finetuned vs Base) |
|-----------|------------------------------------------|-----------------------------------------------------|--------------------------------------|
| **Accuracy**   | 0.52                                     | 0.97                                                | 0.45                                 |
| **Precision** | 0.52                                     | 0.96                                                | 0.44                                 |
| **Recall**    | 0.53                                     | 0.98                                                | 0.45                                 |

![image/png](https://cdn-uploads.huggingface.co/production/uploads/67097b2367976b94cabc116c/7GK_s2eLpbscklwIP1xlx.png)

On the validation dataset (which includes **custom expert-designed phishing cases**), the model still performs admirably:
| Metric          | Base Model (meta-llama/Llama-Guard-3-1B)        | Finetuned Model (AcuteShrewdSecurity/Llama-Phishsense-1B) | Performance Gain (Finetuned vs Base) |
|-----------------|------------------------------------------------|-----------------------------------------------------|---------------------------------|
| **Accuracy**     | 0.31                                           | 0.98                                                | 0.67                            |
| **Precision**    | 0.99                                           | 1.00                                                | 0.01                            |
| **Recall**       | 0.31                                           | 0.98                                                | 0.67                            |

Comparasion with some relevant models is seen below. 
![image/png](https://cdn-uploads.huggingface.co/production/uploads/67097b2367976b94cabc116c/m7zpbWT8SVfu2s6XWjdxk.png)

Paper can be found [here](https://arxiv.org/abs/2503.10944). Please reach out to b1oo@shrewdsecurity.com with feedback :).

# How to Use the Model:
Using the `Llama-Phishsense-1B` is as simple as running a few lines of Python code. You’ll need to load both the base model and the LoRA adapter, and you're ready to classify emails in seconds!

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Function to load the model and tokenizer
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("AcuteShrewdSecurity/Llama-Phishsense-1B")
    base_model = AutoModelForCausalLM.from_pretrained("AcuteShrewdSecurity/Llama-Phishsense-1B")
    model_with_lora = PeftModel.from_pretrained(base_model, "AcuteShrewdSecurity/Llama-Phishsense-1B")
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model_with_lora = model_with_lora.to('cuda')
    
    return model_with_lora, tokenizer

# Function to make a single prediction
def predict_email(model, tokenizer, email_text):
    prompt = f"Classify the following text as phishing or not. Respond with 'TRUE' or 'FALSE':\n\n{email_text}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move inputs to GPU if available
    if torch.cuda.is_available():
        inputs = {key: value.to('cuda') for key, value in inputs.items()}

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=5, temperature=0.01, do_sample=False)
    
    response = tokenizer.decode(output[0], skip_special_tokens=True).split("Answer:")[1].strip()
    return response

# Load model and tokenizer
model, tokenizer = load_model()

# Example email text
email_text = "Urgent: Your account has been flagged for suspicious activity. Please log in immediately."
prediction = predict_email(model, tokenizer, email_text)
print(f"Model Prediction for the email: {prediction}")