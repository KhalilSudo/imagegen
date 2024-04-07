from flask import Flask, request, jsonify
from gradio_client import Client
import io  # Added for handling image data in memory
import base64  # Added for Base64 encoding

app = Flask(__name__)

# Replace with the actual API endpoint URL
client = Client("https://diffusionai-imggenerator.hf.space/--replicas/075wb/")

@app.route('/generate_image', methods=['POST'])
def generate_image():
  # Get user input from the request
  prompt = request.json.get('prompt')
  model = request.json.get('model')
  negative_prompt = request.json.get('negative_prompt', "")
  sampling_steps = request.json.get('sampling_steps', 1)
  cfg_scale = request.json.get('cfg_scale', 1)
  sampling_method = request.json.get('sampling_method', "DPM++ 2M Karras")
  seed = request.json.get('seed', -1)
  strength = request.json.get('strength', 0)
  chatgpt = request.json.get('chatgpt', False)
  width = request.json.get('width', 400)
  height = request.json.get('height', 400)

  # Call the Hugging Face Space API using the client
  result = client.predict(
      prompt=prompt,
      parameter_11=model,
      negative_prompt=negative_prompt,
      sampling_steps=sampling_steps,
      cfg_scale=cfg_scale,
      sampling_method=sampling_method,
      seed=seed,
      strength=strength,
      chatgpt=chatgpt,
      ширина=width,
      высота=height,
      api_name="/predict"
  )

  # Handle the API response (return Base64 encoded image data)
  try:
      # Create an in-memory buffer for the image data
      img_data = io.BytesIO()
      img_data.write(result)

      # Encode the image data to Base64 string
      base64_encoded_data = base64.b64encode(img_data.getvalue()).decode('utf-8')

      # Return the Base64 encoded data and success message
      return jsonify({'message': 'Image generated successfully!', 'image_data': base64_encoded_data})
  except Exception as e:
      # Handle potential errors during image generation
      return jsonify({'error': str(e)}), 500  # Internal Server Error

if __name__ == '__main__':
  app.run(debug=True)
