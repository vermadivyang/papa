from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Load GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Generate a response function
def generate_response(prompt):
    if not prompt.strip():
        return "Please provide a valid question or statement."

    inputs = tokenizer.encode(prompt, return_tensors="pt")
    if inputs.shape[1] == 0:
        return "Sorry, I couldn't understand that. Can you ask in a different way?"

    attention_mask = torch.ones(inputs.shape, device=inputs.device)

    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Flask route to serve the homepage
@app.route("/")
def home():
    return render_template("index.html")

# Flask route to handle chatbot API requests
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("user_input", "")
    response = generate_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

