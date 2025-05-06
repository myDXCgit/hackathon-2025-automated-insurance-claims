import os
import requests
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import uuid
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Load secrets from Azure Key Vault
KEY_VAULT_NAME = os.environ.get("KEY_VAULT_NAME")
KV_URL = f"https://{KEY_VAULT_NAME}.vault.azure.net/"
credential = DefaultAzureCredential()
secret_client = SecretClient(vault_url=KV_URL, credential=credential)

# Retrieve secrets
VISION_KEY = secret_client.get_secret("COMPUTER-VISION-KEY").value
VISION_ENDPOINT = secret_client.get_secret("COMPUTER-VISION-ENDPOINT").value
FORM_RECOGNIZER_KEY = secret_client.get_secret("FORM-RECOGNIZER-KEY").value
FORM_RECOGNIZER_ENDPOINT = secret_client.get_secret("FORM-RECOGNIZER-ENDPOINT").value
COSMOS_DB_URI = secret_client.get_secret("COSMOS-ENDPOINT").value
COSMOS_DB_KEY = secret_client.get_secret("COSMOS-KEY").value
COSMOS_DB_DATABASE = secret_client.get_secret("COSMOS-DB-NAME").value
COSMOS_DB_CONTAINER = secret_client.get_secret("COSMOS-CONTAINER-NAME").value
LOGIC_APP_WEBHOOK_URL = secret_client.get_secret("LOGIC-APP-WEBHOOK-URL").value
GPT_API_KEY = secret_client.get_secret("AZURE-OPENAI-KEY").value
AZURE_OPENAI_ENDPOINT = secret_client.get_secret("AZURE-OPENAI-ENDPOINT").value

# Clients
form_recognizer = DocumentAnalysisClient(FORM_RECOGNIZER_ENDPOINT, AzureKeyCredential(FORM_RECOGNIZER_KEY))
vision_client = ComputerVisionClient(VISION_ENDPOINT, CognitiveServicesCredentials(VISION_KEY))
cosmos_client = CosmosClient(COSMOS_DB_URI, credential=COSMOS_DB_KEY)
container = cosmos_client.get_database_client(COSMOS_DB_DATABASE).get_container_client(COSMOS_DB_CONTAINER)
openai_client = AzureOpenAI(api_key=GPT_API_KEY, api_version="2024-02-15-preview", azure_endpoint=AZURE_OPENAI_ENDPOINT)

@app.route("/", methods=["GET"])
def serve_form():
    return render_template("index.html")

@app.route("/submit-claim", methods=["POST"])
def submit_claim():
    try:
        name = request.form["name"]
        email = request.form["email"]
        description = request.form["accidentDescription"]
        files = request.files.getlist("carPhotos")

        claim_id = str(uuid.uuid4())
        photo_descriptions = []

        for file in files:
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join("/tmp", filename)
                file.save(filepath)

                with open(filepath, "rb") as image_stream:
                    analysis = vision_client.describe_image_in_stream(image_stream)
                    if analysis.captions:
                        caption = analysis.captions[0].text
                        photo_descriptions.append(caption)

        full_description = f"Customer: {name}\nEmail: {email}\nDescription: {description}\nPhotos: {photo_descriptions}"

        # Store in Cosmos DB
        claim_data = {
            "id": claim_id,
            "name": name,
            "email": email,
            "description": description,
            "photoDescriptions": photo_descriptions
        }
        container.upsert_item(claim_data)

        # Trigger Logic App
        if LOGIC_APP_WEBHOOK_URL:
            requests.post(LOGIC_APP_WEBHOOK_URL, json=claim_data)

        return jsonify({
            "status": "success",
            "claimId": claim_id,
            "message": "Claim submitted successfully."
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
