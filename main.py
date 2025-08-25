import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import email
from email import policy
from bs4 import BeautifulSoup
import pandas as pd


# Function to load the model and tokenizer
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("D:/university/Phishsense-1B/Llama-Phishsense-1B")
    base_model = AutoModelForCausalLM.from_pretrained("D:/university/Phishsense-1B/Llama-Phishsense-1B")
    model_with_lora = PeftModel.from_pretrained(base_model, "D:/university/Phishsense-1B/Llama-Phishsense-1B")

    # Move model to GPU if available
    if torch.cuda.is_available():
        model_with_lora = model_with_lora.to('cuda')

    return model_with_lora, tokenizer


# Function to extract text from .eml file
def get_email_text(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        msg = email.message_from_file(f, policy=policy.default)

    text_content = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            if "attachment" not in content_disposition:
                if content_type == "text/plain":
                    try:
                        text_content += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    except:
                        pass
                elif content_type == "text/html":
                    try:
                        html_content = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        soup = BeautifulSoup(html_content, "html.parser")
                        text_content += soup.get_text()
                    except:
                        pass
    else:
        content_type = msg.get_content_type()
        if content_type == "text/plain":
            try:
                text_content = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            except:
                pass
        elif content_type == "text/html":
            try:
                html_content = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                soup = BeautifulSoup(html_content, "html.parser")
                text_content = soup.get_text()
            except:
                pass

    # Clean up the text and limit length to avoid excessive memory usage
    cleaned_text = " ".join(text_content.split())
    return cleaned_text[:2048]


# Function to make a single prediction
def predict_email(model, tokenizer, email_text):
    prompt = f"Classify the following text as phishing or not. Respond with 'TRUE' or 'FALSE':\n\n{email_text}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move inputs to GPU if available
    if torch.cuda.is_available():
        inputs = {key: value.to('cuda') for key, value in inputs.items()}

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    response = tokenizer.decode(output[0], skip_special_tokens=True).split("Answer:")[1].strip()

    if response.startswith("TRUE"):
        return "TRUE"
    if response.startswith("FALSE"):
        return "FALSE"

    return response


def main():
    # Load model and tokenizer
    model, tokenizer = load_model()

    # Load labels
    try:
        labels_df = pd.read_csv('D:/university/Phishsense-1B/Llama-Phishsense-1B/train_labels.csv')
        # Handle potential BOM in column names
        labels_df.columns = labels_df.columns.str.replace('^\ufeff', '', regex=True)
        labels_dict = pd.Series(labels_df.label.values, index=labels_df.name).to_dict()
    except FileNotFoundError:
        print("Error: train_labels.csv not found. Please make sure the file is in the correct directory.")
        return

    # Directory containing the .eml files
    train_dir = 'D:/university/Phishsense-1B/Llama-Phishsense-1B/split_eml/false'

    correct_predictions = 0
    total_predictions = 0

    # Process each .eml file in the directory
    filenames = [f for f in os.listdir(train_dir) if f.endswith('.eml')]
    for filename in filenames:
        file_path = os.path.join(train_dir, filename)
        file_id = os.path.splitext(filename)[0]

        if file_id not in labels_dict:
            # Try matching without potential extra extensions in the CSV name
            base_file_id = file_id.split('_')[0]
            if base_file_id in labels_dict:
                 file_id = base_file_id
            else:
                print(f"Skipping {filename}, no label found.")
                continue

        print(f"Processing {filename}...")

        # Extract text from the email
        email_text = get_email_text(file_path)

        if email_text:
            # Get prediction from the model
            prediction_str = predict_email(model, tokenizer, email_text)

            # Get true label
            true_label = labels_dict[file_id]

            # Convert prediction string to boolean/int
            prediction = 1 if prediction_str.upper() == 'TRUE' else 0

            print(f"File: {filename} - Prediction: {prediction_str} ({prediction}), True Label: {true_label}")

            if prediction == true_label:
                correct_predictions += 1
            total_predictions += 1
        else:
            print(f"File: {filename} - Could not extract text.\n")

    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"\nModel Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
    else:
        print("\nNo files were processed or found in the directory.")


if __name__ == '__main__':
    main()
