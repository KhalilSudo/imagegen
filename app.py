from flask import Flask, request, jsonify
from gradio_client import Client

app = Flask(__name__)

client = Client("https://diffusionai-imggenerator.hf.space/--replicas/075wb/")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    prompt = data["prompt"]
    model = "PixelArt XL"
    negative_prompt = data.get("negative_prompt")
    sampling_steps = data.get("sampling_steps", 1)
    cfg_scale = data.get("cfg_scale", 1)
    sampling_method = data.get("sampling_method", "DPM++ 2M Karras")
    seed = data.get("seed", -1)
    strength = data.get("strength", 0)
    chatgpt = data.get("chatgpt", True)
    width = data.get("width", 15)
    height = data.get("height", 15)

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
        api_name="/predict"
    )

    return jsonify({"filepath": result})

if __name__ == "__main__":
    app.run()