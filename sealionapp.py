from tools.sealion_mcp import SealionConfig, create_sealion_agent
import logging
from tools.pydantic_agent import ExtractUserIllness
from dotenv import load_dotenv
import os
from typing import Callable, Awaitable, Any
import asyncio
Emit = Callable[[str, Any, str | None], Awaitable[None]]

async def emit_progress_message(step,messages,emit:Emit,data={}):
    return await emit("progress", {
            "step": step,
            "message": messages,
            "data":data
        }, id_=str(step))

async def emit_complete_message(step,messages,emit:Emit,data={}):
    return await emit("completed", {
            "step": step,
            "message": messages,
            "data":data
        }, id_=str(step))

def testing_result():
    return {
          "result": {
              "medicine_details": [
                  {
                      "medicine_name": "ginger tea",
                      "medicine_instruction": "Boil sliced ginger in water (often with honey and/or lime).",
                      "dosage": None
                  },
                  {
                      "medicine_name": "turmeric tea",
                      "medicine_instruction": "Combine turmeric powder or grated fresh turmeric with ginger in hot water.",
                      "dosage": None
                  },
                  {
                      "medicine_name": "lemongrass tea",
                      "medicine_instruction": "Steep lemongrass stalks in hot water.",
                      "dosage": None
                  },
                  {
                      "medicine_name": "galangal tea",
                      "medicine_instruction": "Similar preparation to ginger tea.",
                      "dosage": None
                  },
                  {
                      "medicine_name": "ginger-garlic decoction",
                      "medicine_instruction": "Combine ginger and garlic in a decoction.",
                      "dosage": None
                  },
                  {
                      "medicine_name": "warm broth",
                      "medicine_instruction": "Consume chicken or vegetable broth.",
                      "dosage": None
                  }
              ],
              "non_pharmacologic_methods": [
                  {
                      "method_name": "rest",
                      "instructions": "Get plenty of rest.",
                      "frequency": None,
                      "duration": None,
                      "notes": None
                  },
                  {
                      "method_name": "hydration",
                      "instructions": "Drink plenty of warm fluids.",
                      "frequency": None,
                      "duration": None,
                      "notes": None
                  },
                  {
                      "method_name": "warm clothing",
                      "instructions": "Dress warmly, especially if you feel chilled.",
                      "frequency": None,
                      "duration": None,
                      "notes": None
                  },
                  {
                      "method_name": "steam inhalation",
                      "instructions": "Inhale steam infused with lemongrass or ginger.",
                      "frequency": None,
                      "duration": None,
                      "notes": "Be careful to avoid burns."
                  },
                  {
                      "method_name": "warm compress",
                      "instructions": "Apply warm compresses to the chest or back.",
                      "frequency": None,
                      "duration": None,
                      "notes": None
                  }
              ],
              "analysis": "The text describes traditional remedies for cold and flu symptoms common in Indonesian and Vietnamese cultures, focusing on warming the body and expelling 'cold'. Recommendations include herbal teas (ginger, turmeric, lemongrass, galangal), warm broths, and lifestyle measures like rest, hydration, and warm clothing. The remedies are generally supportive and aim to alleviate symptoms.",
              "severity": "low"
          }
      }

async def main_app(query : str,emit:Emit):
    # Load environment variables from .env file
    load_dotenv()
    
    import boto3
    s=boto3.Session()
    # print("method:", s.get_credentials().method)           # should be 'iam-role'
    # print(boto3.client("sts").get_caller_identity())
    
    if query is None or query.strip() == "":
        await emit_complete_message(999,"No query provided.",emit=emit,data=testing_result())
        return testing_result()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
        
    ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME",None)
    ENDPOINT_NAME2 = os.environ.get("ENDPOINT_NAME2",None)

    if ENDPOINT_NAME is None:
        raise Exception("LLM Model has not been initiated.")
    
    
    await emit_progress_message(0,f"Searching solution for query: {query}",emit=emit,data={})
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

    api_key = os.environ.get("SEALION_API","Unkown")
    model_name = os.environ.get("SEALION_MODEL","aisingapore/Gemma-SEA-LION-v4-27B-IT")
    
    # Step 1: Extract illness from user's query
    #  emit_message(1,,emit=emit)
    await emit_progress_message(1,f"Extracting illness from user query: {query}...",emit=emit,data={})
    
    # Extract Illness Extraction from user's query
    extraction_model = ExtractUserIllness(api_key,model_name)
    illness_extraction = await extraction_model.retrieve_response(query,extraction_purpose="health")
    
    await emit_progress_message(2,"Creating agent to search for medicines...",emit=emit)
    
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
        f"User origin language : {illness_extraction['origin_language']}"
        f"User destination language : {illness_extraction['destination_language']}"
        f"Any important things to note : {illness_extraction['important_things_to_note']}"
    )
    await emit_progress_message(3,"Calling agent to generate response.",emit=emit)
    # 1. Get medicine based on their own country
    try:
        async with asyncio.timeout(2000):  # 35 minutes timeout
            response = await agent.invoke(query_formatting, max_iterations=15,
                                                origin_country=illness_extraction['origin_location'],
                                                destination_country=illness_extraction['destination_location'],
                                                emit = emit)
    except asyncio.TimeoutError:
        await emit_progress_message(3, "Request timed out after 5 minutes", emit=emit)
        raise Exception("Agent execution exceeded time limit (5 minutes)")
    except Exception as e:
        await emit_progress_message(3, f"Error during agent execution: {str(e)}", emit=emit)
        raise Exception(f"Agent execution failed: {str(e)}")
    # Generate final response of user's illness based on query translation
    # response = agent.invoke(illness_extraction['query_translate_based_on_destination'], max_iterations=6)

    # Display results
    print("\n" + "="*60)
    print("FINAL ANSWER:")
    print("="*60)
    print(response.get("output", "No response generated"))

    # if response.get("intermediate_steps"):
    #     print("\n" + "="*60)
    #     print("TOOL USAGE:")
    #     print("="*60)
    #     for i, step in enumerate(response.get("intermediate_steps", [])):
    #         if len(step) >= 3:
    #             print(f"\nStep {i+1}:")
    #             print(f"  Tool: {step[0]}")
    #             print(f"  Input: {step[1]}")
    #             print(f"  Result: {step[2][:200]}..." if len(step[2]) > 200 else f"  Result: {step[2]}")
    await emit_progress_message(4,"Generating final response for user by summarizing every articles found...",emit=emit)
    final_response = await extraction_model.retrieve_response(response,extraction_purpose="medicine",
                                                                    destination_country=illness_extraction['destination_location'],
                                                                    origin_country=illness_extraction['origin_location'])
    await emit_complete_message(999,f"Providing final answer.",emit=emit,data={"result":final_response})
