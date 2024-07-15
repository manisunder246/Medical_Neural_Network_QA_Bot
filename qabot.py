from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model_path = "./fine_tuned_roberta"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

@app.route('/bot', methods=['POST'])
def bot():
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
    answer = answer.replace(tokenizer.pad_token, '').replace(tokenizer.sep_token, '').replace(tokenizer.cls_token, '').strip()

    # Expand the answer to complete the sentence if it seems cut off
    if not answer.endswith('.'):
        additional_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_end:answer_end + 10])
        additional_text = tokenizer.convert_tokens_to_string(additional_tokens).split('.')[0] + '.'
        answer += ' ' + additional_text

    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(port=5001)  # Change the port to 5001 or any other available port
