from tools.sealion_mcp import SealionConfig, create_sealion_agent
import logging
from tools.pydantic_agent import ExtractUserIllness
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME",None)
ENDPOINT_NAME2 = os.environ.get("ENDPOINT_NAME2",None)

if ENDPOINT_NAME is None:
    raise Exception("LLM Model has not been initiated.")

# Configure the agent
config = SealionConfig(
    endpoint_name=ENDPOINT_NAME,  # Replace with your actual endpoint
    region_name="us-east-1",
    temperature=0.3,  # Lower for better tool following
    max_tokens=1024*5,
    top_p=0.9
)

# Create the agent
print("="*60)
print("Initializing Enhanced Sealion MCP Agent...")
print("="*60)

# Test query in Indonesian
query = (
    "Nama ku Kevin dari Jakarta dan aku sakit masuk angin,aku berlokasi di Vietnam. "
    "Tolong carikan herbal atau obat traditional yang mudah ditemukan di Vietnam."
)
api_key = os.environ.get("SEALION_API","Unkown")
model_name = os.environ.get("SEALION_MODEL","aisingapore/Gemma-SEA-LION-v4-27B-IT")

# Extract Illness Extraction from user's query
illness_extraction = ExtractUserIllness(api_key,model_name).retrieve_response(query,extraction_purpose="health")

# Create first agent to summarize every conversation and generate the final answer
try:
    agent = create_sealion_agent(config)
    print("✓ Agent initialized successfully\n")
except Exception as e:
    print(f"✗ Failed to initialize agent: {e}")
    exit(1)

print("Query (Indonesian):")
print(f"  {illness_extraction['query_translate_based_on_destination']}\n")
print("Processing...\n")
print("-"*60)

query_formatting = (
    f"User illness: {illness_extraction['illness']}"
    f"User origin location : {illness_extraction['origin_location']} and origin country code \
        for google searching: {illness_extraction['origin_code']}"
    f"User destination location : {illness_extraction['destination_location']} and destination country \
            code for google searching: {illness_extraction['destination_code']}"
)

# 1. Get medicine based on their own country
response = agent.invoke(query_formatting, max_iterations=3)

# Generate final response of user's illness based on query translation
# response = agent.invoke(illness_extraction['query_translate_based_on_destination'], max_iterations=6)

# Display results
print("\n" + "="*60)
print("FINAL ANSWER:")
print("="*60)
print(response.get("output", "No response generated"))

if response.get("intermediate_steps"):
    print("\n" + "="*60)
    print("TOOL USAGE:")
    print("="*60)
    for i, step in enumerate(response.get("intermediate_steps", [])):
        if len(step) >= 3:
            print(f"\nStep {i+1}:")
            print(f"  Tool: {step[0]}")
            print(f"  Input: {step[1]}")
            print(f"  Result: {step[2][:200]}..." if len(step[2]) > 200 else f"  Result: {step[2]}")

final_response = ExtractUserIllness(api_key,model_name).retrieve_response(response,extraction_purpose="medicine")
print(final_response)