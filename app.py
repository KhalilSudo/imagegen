from flask import Flask, request, jsonify
from gradio_client import Client
import base64

app = Flask(__name__)

client = Client("https://diffusionai-imggenerator.hf.space/--replicas/075wb/")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check for JSON data
        data = request.get_json()
        if not data:
            raise ValueError("Missing request data")

        # Extract prompt (required)
        prompt = data.get("prompt")
        if not prompt:
            raise ValueError("Missing required field: prompt")

        # Extract and handle optional parameters
        model = data.get("model", "PixelArt XL")
        negative_prompt = data.get("negative_prompt")
        sampling_steps = data.get("sampling_steps", 1)
        cfg_scale = data.get("cfg_scale", 1)
        sampling_method = data.get("sampling_method", "DPM++ 2M Karras")
        seed = data.get("seed", -1)
        strength = data.get("strength", 0)
        chatgpt = data.get("chatgpt", True)
        width = data.get("width", 15)
        height = data.get("height", 15)

        # Prediction with Gradio client
        result = client.predict(
            prompt,
            model,
            negative_prompt,
            sampling_steps,
            cfg_scale,
            sampling_method,
            seed,
            strength,
            chatgpt,
            width,
            height,
            api_name="/predict",
        )

        # Encode image to base64 string
        with open(result, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        return jsonify({"image": encoded_image})

    except (ValueError, Exception) as e:
        # Handle any exceptions during processing
        return jsonify({"error": str(e)}), 400  # Return error message and bad request code

if __name__ == "__main__":
    app.run()