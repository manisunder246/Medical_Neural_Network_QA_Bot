from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__, static_folder='.', static_url_path='')

# Load the fine-tuned model and tokenizer
model_path = "./fine_tuned_roberta"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
except Exception as e:
    logging.error(f"Error loading model or tokenizer: {e}")
    raise e

@app.route('/bot', methods=['POST'])
def bot():
    try:
        data = request.get_json()
        context = data['context']
        question = data['question']

        inputs = tokenizer(question, context, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)

        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1

        # Ensure the indices are within the bounds of the context length
        answer_start = min(answer_start.item(), len(inputs['input_ids'][0]) - 1)
        answer_end = min(answer_end.item(), len(inputs['input_ids'][0]))

        answer_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end])
        answer = tokenizer.convert_tokens_to_string(answer_tokens)

        # Ensure the answer is legible and remove any special tokens
        if tokenizer.pad_token is not None:
            answer = answer.replace(tokenizer.pad_token, '')
        if tokenizer.sep_token is not None:
            answer = answer.replace(tokenizer.sep_token, '')
        if tokenizer.cls_token is not None:
            answer = answer.replace(tokenizer.cls_token, '')
        
        answer = answer.strip()

        # Remove the extension logic to avoid truncation issues
        additional_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_end:])
        additional_text = tokenizer.convert_tokens_to_string(additional_tokens)
        answer += ' ' + additional_text.strip()

        return jsonify({'answer': answer})
    except Exception as e:
        logging.error(f"Error in bot endpoint: {e}")
        return jsonify({'error': str(e)}), 100

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/background.jpg')
def background():
    return send_from_directory('.', 'background.jpg')

if __name__ == '__main__':
    app.run(port=5001)
