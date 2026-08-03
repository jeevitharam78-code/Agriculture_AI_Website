from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Your Colab API URL
API_URL = "https://widow-traverse-handshake.ngrok-free.dev/predict"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():

    user_question = request.json["question"]

    response = requests.post(
        API_URL,
        json={"question": user_question}
    )

    answer = response.json()["answer"]

    return jsonify({
        "answer": answer
    })

if __name__ == "__main__":
    app.run(debug=True)