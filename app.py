from typing import Dict, Any
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def initialize_model() -> Dict[str, Any]:
    """
    Initialize the GPT-2 model and tokenizer
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    return {'tokenizer': tokenizer, 'model': model}

def load_user_preferences() -> Dict[str, Any]:
    """
    Load user preferences from data folder
    """
    try:
        with open('data/preferences.txt', 'r') as f:
            preferences = f.read()
            return {'preferences': preferences}
    except FileNotFoundError:
        print("No preferences file found. Using default settings.")
        return {'preferences': None}

def generate_creative_content(model_data: Dict[str, Any], preferences: Dict[str, Any]) -> str:
    """
    Generate creative content based on user preferences
    """
    tokenizer = model_data['tokenizer']
    model = model_data['model']
    
    if preferences['preferences']:
        prompt = preferences['preferences']
    else:
        prompt = "Write a creative story about a brave knight and a mysterious forest."

    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model(**inputs)
    generated_text = tokenizer.decode(outputs.logits[0], skip_padding=True)
    return generated_text

def provide_feedback(generated_text: str) -> str:
    """
    Provide basic feedback on the generated text
    """
    feedback = "Generated text length: " + str(len(generated_text)) + " characters."
    return feedback

def main():
    print("Initializing AI model...")
    model_data = initialize_model()
    print("Model initialized successfully."
)

    print("Loading user preferences...")
    preferences = load_user_preferences()
    print("User preferences loaded."
)

    print("Generating creative content...")
    creative_content = generate_creative_content(model_data, preferences)
    print("Creative content generated. Here's the output:\n" + creative_content)

    feedback = provide_feedback(creative_content)
    print("Feedback: " + feedback)

if __name__ == '__main__':
    main()