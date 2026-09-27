from flask import Flask, render_template, request, jsonify
import os
import torch

from transformers import (
    Gemma4ForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

from peft import PeftModel

from disease_model import predict_disease
from disease_info import disease_info


# =========================================================
# Flask
# =========================================================

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "static",
    "uploads"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# =========================================================
# Agriculture Gemma Model
# =========================================================

BASE_MODEL = "google/gemma-4-E2B-it"

ADAPTER_PATH = r"C:\agriculture_ai_website\agriculture_gemma4_lora"


print("========================================")
print("Loading Agriculture AI Model...")
print("========================================")


# Processor

print("Loading processor...")

processor = AutoProcessor.from_pretrained(
    BASE_MODEL
)

print("Processor loaded.")


# 4-bit configuration

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


# Load Gemma on GPU

print("Loading Gemma on GPU...")

base_model = Gemma4ForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map={"": 0},
)

print("Gemma loaded.")


# Load LoRA

print("Loading Agriculture LoRA...")

agriculture_model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

agriculture_model.eval()

print("Agriculture LoRA loaded.")

print("========================================")
print("Agriculture AI Model Ready")
print("========================================")


# =========================================================
# Home
# =========================================================

@app.route("/")
def home():

    return render_template("index.html")


# =========================================================
# Agriculture Chat
# =========================================================

@app.route("/ask", methods=["POST"])
def ask():

    try:

        data = request.get_json()

        user_question = data.get("question", "").strip()

        if not user_question:

            return jsonify({
                "answer": "Please enter an agriculture question."
            })


        messages = [
            {
                "role": "user",
                "content": user_question
            }
        ]


        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )


        # Send inputs to GPU

        model_device = next(
            base_model.parameters()
        ).device


        inputs = {
            key: value.to(model_device)
            if hasattr(value, "to")
            else value

            for key, value in inputs.items()
        }


        # Generate answer

        with torch.inference_mode():

            output = agriculture_model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
            )


        # Only decode newly generated tokens

        input_length = inputs["input_ids"].shape[1]

        generated_tokens = output[0][input_length:]


        answer = processor.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()


        return jsonify({
            "answer": answer
        })


    except Exception as e:

        print("Agriculture error:", e)

        return jsonify({
            "answer": "Sorry, an error occurred while processing your question."
        }), 500


# =========================================================
# Plant Disease Detection
# =========================================================

@app.route("/predict_disease", methods=["POST"])
def predict_disease_route():

    try:

        if "image" not in request.files:

            return jsonify({
                "error": "Please upload a plant image."
            }), 400


        image = request.files["image"]


        if image.filename == "":

            return jsonify({
                "error": "Please select an image."
            }), 400


        # Save uploaded image

        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            image.filename
        )

        image.save(image_path)


        # Predict

        predicted_class, confidence = predict_disease(
            image_path
        )


        # Get information

        info = disease_info.get(
            predicted_class,
            {
                "description": "Information not available.",
                "treatment": "Consult an agricultural expert.",
                "prevention": "Follow good crop management practices.",
                "product": ""
            }
        )


        return jsonify({

            "disease": predicted_class,

            "confidence": round(
                confidence,
                2
            ),

            "description": info.get(
                "description",
                ""
            ),

            "treatment": info.get(
                "treatment",
                ""
            ),

            "prevention": info.get(
                "prevention",
                ""
            ),

            "product": info.get(
                "product",
                ""
            )
        })


    except Exception as e:

        print("Disease prediction error:", e)

        return jsonify({
            "error": str(e)
        }), 500


# =========================================================
# Run Flask
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )