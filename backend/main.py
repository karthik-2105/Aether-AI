import os
import re
import subprocess
import threading
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# Load environment variables
load_dotenv()
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
FIREBASE_CREDENTIALS_PATH = "firebase_credentials.json"

# Initialize Firebase Admin SDK
cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Azure OpenAI Client
llm = AzureChatOpenAI(
    azure_deployment="gpt-4",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_key=AZURE_OPENAI_KEY,
    api_version="2023-12-01-preview"
)

# Extract the largest Python code block
def extract_largest_code_block(text):
    matches = re.findall(r"```python(.*?)```", text, re.DOTALL)
    return max(matches, key=len).strip() if matches else ""

# Extract Manim class name from Python code
def extract_class_name(code):
    match = re.search(r'class\s+(\w+)\s*\(.*Scene.*\)', code)
    return match.group(1) if match else None

# Generate Manim code
def generate_manim_code(prompt):
    messages = [{"role": "user", "content": f"Generate Manim code: {prompt}"}]
    response = llm.invoke(messages)
    manim_code = extract_largest_code_block(response.content)
    
    if not manim_code:
        return {"error": "No valid Manim code generated."}

    class_name = extract_class_name(manim_code)
    if not class_name:
        return {"error": "No valid Manim class found."}

    return {"code": manim_code, "class_name": class_name}

# Run Manim script and save video locally
def run_manim_script(manim_code, class_name, doc_id):
    script_filename = f"generated_{doc_id}.py"
    output_filename = f"animation_{doc_id}.mp4"
    script_path = os.path.join("generated_scripts", script_filename)
    output_path = os.path.join("static", output_filename)

    os.makedirs("generated_scripts", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(manim_code)
    
    try:
        subprocess.run([
            "manim", script_path, class_name, "-o", output_path, "--format=mp4"
        ], check=True, capture_output=True, text=True)

        # Store video file path in Firestore
        db.collection("manim_videos").document(doc_id).update({
            "status": "completed",
            "video_path": output_filename  # Store only the filename
        })
        
    except subprocess.CalledProcessError as e:
        db.collection("manim_videos").document(doc_id).update({
            "status": "failed",
            "error": str(e.stderr)
        })

# API Route to generate Manim animation
@app.route("/api/generate_manim", methods=["POST"])
def generate_manim():
    data = request.json
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400
    
    result = generate_manim_code(prompt)
    if "error" in result:
        return jsonify(result), 500
    
    doc_ref = db.collection("manim_videos").add({
        "prompt": prompt,
        "status": "processing",
        "video_path": None
    })
    
    doc_id = doc_ref[1].id  # Get Firestore document ID
    threading.Thread(target=run_manim_script, args=(result["code"], result["class_name"], doc_id)).start()
    
    return jsonify({"message": "Video is being generated", "id": doc_id})

# API Route to fetch video status
@app.route("/api/video_status/<doc_id>", methods=["GET"])
def video_status(doc_id):
    doc = db.collection("manim_videos").document(doc_id).get()
    if doc.exists:
        return jsonify(doc.to_dict())
    return jsonify({"error": "Video not found"}), 404

# API Route to serve generated videos
@app.route("/static/<filename>", methods=["GET"])
def serve_video(filename):
    return send_from_directory("static", filename)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)