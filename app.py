import os
import tempfile
from flask import Flask, request, jsonify
from index import AddressAISystem
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

system = AddressAISystem()


@app.route("/process_address", methods=["POST"])
def process_address():
    data = request.get_json()
    address = data.get("address", "")
    result = system.process_address(address)
    return jsonify({"data": result})


@app.route("/bulk_process_addresses", methods=["POST"])
def bulk_process_addresses():
    data = request.get_json()
    addresses = data.get("addresses", [])
    print(addresses)
    results = system.process_addresses_bulk(addresses)
    return jsonify({"data": results})


@app.route("/ocr", methods=["POST"])
def ocr():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".jpeg", dir=tempfile.gettempdir()
    ) as temp_file:
        file_path = temp_file.name
        file.save(file_path)

    try:
        sample_file = genai.upload_file(path=file_path)
        model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
        response = model.generate_content(
            [
                "OCR this image and just give the written words dont give any extra things",
                sample_file,
            ]
        )

        print(response.text)
        return jsonify({"output": response.text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        os.remove(file_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4040)
