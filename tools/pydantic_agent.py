from pydantic import BaseModel, Field
from typing import Optional,List
import requests
import json
from openai import OpenAI
import re
from typing import List, Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field, field_validator, constr, conlist, ConfigDict

# 1. Define schema with Pydantic
class HealthQuery(BaseModel):
    illness: str = Field(..., description="Name of the illness or health condition")
    illness_explanation: str = Field(..., description="User's explanation on the illnesses, what he feels and experiences.")
    destination_location: Optional[str] = Field(None, description="Current country or location of the user")
    origin_location: Optional[str] = Field(None, description="Origin country or location of the user")
    query_translate_based_on_destination : Optional[str] = Field(None)
    destination_code : str = Field(None,description="User's destination location country code based on ISO 3166-1 alpha-2")
    origin_code : str = Field(None,description="User's destination location country code based on ISO 3166-1 alpha-2")

# --- Helpers / Enums ---
class Severity(str, Enum):
    low = "low"
    moderate = "moderate"
    high = "high"
    emergency = "emergency"


NonEmptyShort = constr(strip_whitespace=True, min_length=1, max_length=120)
NonEmptyText  = constr(strip_whitespace=True, min_length=1, max_length=2000)


# --- Models ---
class NonPharmacologicMethod(BaseModel):
    """
    A non-drug/self-care recommendation (e.g., warm compress, hydration).
    """
    model_config = ConfigDict(extra="forbid")

    method_name: str = Field(
        ...,
        description="Name of the non-pharmacologic method (e.g., 'warm compress', 'hydration').",
        examples=["warm compress", "hydration"],
    )
    instructions: str = Field(
        ...,
        description="Practical steps for doing it safely/effectively.",
        examples=[
            "Apply a warm, damp compress to the forehead or neck for 10–15 minutes.",
            "Encourage frequent small sips of water or oral rehydration solution.",
        ],
    )
    frequency: Optional[str] = Field(
        None,
        description="How often to do it.",
        examples=["every 3–4 hours as needed", "aim for 50–100 ml/kg/day fluid intake"],
    )
    duration: Optional[str] = Field(
        None,
        description="How long to continue.",
        examples=["for 24–48 hours, reassess if fever persists"],
    )
    notes: Optional[str] = Field(
        None,
        description="Caveats, contraindications, or safety notes.",
        examples=["Avoid if skin is broken or sensitive to heat."],
    )

class MedicineDetails(BaseModel):
    """
    A single recommended medicine or remedy with basic instructions and (optional) dosage.
    """
    model_config = ConfigDict(extra="forbid")

    medicine_name: NonEmptyShort = Field(
        ...,
        description=(
            "The recommended medicine/remedy (can be conventional, traditional, or herbal). "
            "Use generic names when possible (e.g., 'ibuprofen', 'ginger tea')."
        ),
        examples=["ibuprofen", "ginger tea", "honey + lemon"],
    )

    medicine_instruction: NonEmptyText = Field(
        ...,
        description=(
            "Do not generate or take from the example. You must extract what is written in the QUERY."
            "Clear, actionable steps for using the medicine/remedy (preparation, timing, with/without food, "
            "topical/oral, etc.). Keep safety notes concise."
        ),
        examples=[
            "Take after meals with water. Avoid on empty stomach.",
            "Steep sliced ginger in hot water 10 minutes; drink warm.",
        ],
    )

    dosage: Optional[constr(strip_whitespace=True, max_length=300)] = Field(
        None,
        description=(
            "Do not generate or take from the example. You must extract what is written in the QUERY."
            "Frequency/amount/duration when applicable. "
            "Example: '400 mg every 6–8 hours as needed (max 1200 mg/day) for up to 3 days.'"
        ),
        examples=[
            "400 mg every 6–8 hours (max 1200 mg/day) for ≤3 days",
            "1 cup, 2–3×/day for 5–7 days",
        ],
    )

    @field_validator("medicine_name", "medicine_instruction", mode="before")
    def _strip_and_collapse_ws(cls, v: str) -> str:
        if isinstance(v, str):
            return " ".join(v.split())
        return v

class MedicineQuery(BaseModel):
    """
    Structured output for detected medicines/remedies + an explanation and severity assessment.
    """
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "examples": [
            {
                "medicine_details": [
                    {
                        "medicine_name": "ibuprofen",
                        "medicine_instruction": "Take after food with water. Avoid if you have stomach ulcers.",
                        "dosage": "200–400 mg every 6–8 hours as needed (max 1200 mg/day) for ≤3 days"
                    },
                    {
                        "medicine_name": "ginger tea",
                        "medicine_instruction": "Steep fresh ginger in hot water for 10 minutes and drink warm.",
                        "dosage": "1 cup, 2–3×/day for 5 days"
                    }
                ],
                "non_pharmacologic_methods": [
                    {
                        "method_name": "warm compress",
                        "instructions": "Apply a warm, damp cloth to the forehead/neck 10–15 minutes.",
                        "frequency": "repeat every 3–4 hours as needed",
                        "duration": "while febrile; stop if irritation occurs",
                    },
                    {
                        "method_name": "hydration",
                        "instructions": "Encourage frequent small sips of water or oral rehydration solution.",
                        "frequency": "ongoing",
                        "duration": "until symptoms resolve",
                        "notes": "Aim for normal urine output and light-yellow urine.",
                    },
                ],
                "analysis": "Symptoms consistent with viral URI; focus on hydration, rest, NSAIDs for pain/fever.",
                "severity": "low"
            }
        ]
    })

    medicine_details: conlist(MedicineDetails, min_length=1) = Field(
        ...,
        description=(
            "List of one or more recommended medicines/remedies with instructions and optional dosage."
        ),
    )

    non_pharmacologic_methods: conlist(NonPharmacologicMethod, min_length=0) = Field(
        default_factory=list,
        description="List of non-drug/self-care recommendations (e.g., warm compress, hydration).",
    )

    analysis: str = Field(
        ...,
        description=(
            "Concise reasoning: what the likely illness is, key differentials, red flags to watch, "
            "and rationale for recommendations."
        ),
        examples=[
            "Likely tension headache; advise hydration, sleep hygiene, and NSAID trial. No red flags reported."
        ],
    )

    severity: Severity = "low"

class ExtractUserIllness:
    def __init__(self,api_key : str,model_name:str):
        self.API_KEY = api_key
        self.SEALION_MODEL = model_name
        self.client =  OpenAI(
                                api_key=api_key,
                                base_url="https://api.sea-lion.ai/v1"
                            )
    def retrieve_query_translate(self,query,destination_language:str):
        query_translate = self.client.chat.completions.create(
                            model=self.SEALION_MODEL,
                            messages=[
                                {
                                    "role": "user",
                                    "content": f"""
                                                    You are an expert in medicine and multilingual support.

                                                    Task: Translate and adapt the following query into {destination_language}, 
                                                    so that it can be directly used in a search engine to find remedies, treatments, 
                                                    or cures for the illness.

                                                    Query: "{query}"

                                                    Requirements:
                                                    - Return only the translated text (no explanations, no prefixes).
                                                    - Include the medical term for the illness in {destination_language}.
                                                    - Expand the query with common phrases that help in searching for treatments, 
                                                    cures, or remedies.
                                                    - Output must be a single clean sentence in {destination_language}, 
                                                    suitable for internet search.
                                                """
                                }
                            ],
                            extra_body={
                                "chat_template_kwargs": {
                                    "thinking_mode": "off"
                                }
                            },
                    )

        # Get the raw translation string
        translation = query_translate.choices[0].message.content.strip()
        return translation
    def retrieve_response(self,query,extraction_purpose = "health"):
        # 1. Define Pydantic Format
        if extraction_purpose == "medicine":
            PydanticFormat = MedicineQuery
        else:
            PydanticFormat = HealthQuery
        
        prompt = f"""
                    Extract structured health query information from the following text:

                    Text: "{query}"

                    Return JSON strictly in this schema:
                    {PydanticFormat.schema()}
                    """


        completion = self.client.chat.completions.create(
            model=self.SEALION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical information extractor. Always return output as valid JSON that follows the schema: "
                              f"{PydanticFormat.schema()}"
                },
                {
                    "role": "user",
                    "content": f"""
                        {prompt}
                    """
                }
            ],
            extra_body={
                "chat_template_kwargs": {
                    "thinking_mode": "off"
                }
            },
            temperature=0.0
        )
               
        # Parse output
        result_text = completion.choices[0].message.content.strip()

        try:
            clean_text = re.sub(r"^```json|```$", "", result_text.strip(), flags=re.MULTILINE).strip()

            parsed = PydanticFormat.parse_raw(clean_text)
            if extraction_purpose == "health":
                print("in")
                parsed.query_translate_based_on_destination = self.retrieve_query_translate(f"Illness name: {parsed.illness} with {parsed.illness_explanation}",parsed.destination_location)
            
            print("✅ Parsed result:", parsed.dict())
            return parsed.dict()
        except Exception as e:
            print("❌ Validation failed:", e)
            print("Raw output:", result_text)
