from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

hf_api_token = os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACEHUB_ACCESS_TOKEN")
if not hf_api_token:
    raise RuntimeError(
        "Set HF_API_TOKEN in your environment or .env file before running this script. "
        "See https://huggingface.co/docs/huggingface_hub/login for details."
    )

hf_model = os.environ.get("HF_MODEL") or os.environ.get("HUGGINGFACEHUB_MODEL_ID")
hf_endpoint_url = os.environ.get("HF_ENDPOINT_URL")
if not hf_model and not hf_endpoint_url:
    raise RuntimeError(
        "Set HF_MODEL or HUGGINGFACEHUB_MODEL_ID, or HF_ENDPOINT_URL in your environment or .env file before running this script. "
        "Example: HF_MODEL=mistralai/Mistral-7B-Instruct"
    )

llm = HuggingFaceEndpoint(
    model=hf_model,
    endpoint_url=hf_endpoint_url,
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
    huggingfacehub_api_token=hf_api_token,
)
model = ChatHuggingFace(llm=llm)
response = model.invoke([("human", "Explain machine learning in simple terms.")])
print(response.content)