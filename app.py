from happytransformer import HappyGeneration
from transformers_model import GPTNeoForCausalLM, GPT2Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify


# Load the saved model using Keras
model = load_model('gen_model.hdf5')

# Load the GPT-Neo tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

# Load the GPT-Neo model
transformers_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

# Load the Transformers model weights into the Keras model
model.layers[1].set_weights(transformers_model.get_weights())


# Initialize the Flask app
app = Flask(__name__)

# Define a route for the home screen 
@app.route('/')
def index():
    return """<a href='https://ip4less.com' style='text-decoration:none;'>
    <h1 style='color:red; font-family: Verdana; padding:0 20px; display: inline-block;'>IP4Less</h1>
    </a>"""
    
# Define a route for the POST request   
@app.route("/generate", methods=['POST'])
def process():
    # Get the input text from the POST request
    input_text = request.json['input_text']
    
    # Tokenize the input text and convert it to a tensor
    input_ids = tokenizer.encode(input_text, return_tensors='tf')

    # Generate the output paragraph using the model
    max_length = 100
    output_ids = transformers_model.generate(input_ids, max_length=max_length, do_sample=True)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Return the output paragraph as a JSON response
    return jsonify({'output_text': output_text})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
    