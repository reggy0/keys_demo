import torch
from flask import Flask, jsonify, request
from happytransformer import HappyGeneration, GENSettings

# load the saved model weights when the API starts up
model_weights_path = "happy_gen_model_weights.pt"
happy_gen = HappyGeneration("GPT-NEO", "EleutherAI/gpt-neo-1.3B")
happy_gen.model.load_state_dict(torch.load(model_weights_path))

# Initialize the Flask app
app = Flask(__name__)

# Create a route for generating text
@app.route('/generate_text', methods=['POST'])
def generate_text():
    # Get input data from request
    data = request.json
    keywords = data['keywords']
    # Check format of keywords and convert to list
    if isinstance(keywords, str):
        # Check if keywords are comma-separated
        if "," in keywords:
            keywords_list = keywords.split(",")
        else:
            # Check if keywords are bulleted list
            keywords_list = re.findall(r"[\*•]\s*(\w+)", keywords)
    elif isinstance(keywords, list):
        keywords_list = keywords
    else:
        return jsonify({'error': 'Invalid format for keywords'})
    
    # Create a prompt using the training cases and the provided keywords
    prompt = create_prompt(training_cases, keywords)

    # Generate text using the HappyGeneration model
    args_beam = GENSettings(num_beams=5, no_repeat_ngram_size=3, early_stopping=True, min_length=1, max_length=100)
    result = happy_gen.generate_text(prompt, args=args_beam)

    # Return the generated text as a JSON response
    return jsonify({'generated_text': result.text})

if __name__ == '__main__':
    app.run()
