import os
import requests
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import json
from jsonschema import validate, ValidationError
import re
import json

# TASK 1
# ======================================================================

load_dotenv()

api_key = os.getenv("LLM_API_KEY")

url = "https://openrouter.ai/api/v1/chat/completions"

MODEL_NAME = "openrouter/auto"


def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        print("Status Code:", response.status_code)
        return None

    return response.json()["choices"][0]["message"]["content"]

result = call_llm(
    system_prompt="You are a helpful assistant.",
    user_prompt="Reply with only the word: hello"
)

print(result)


def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'

    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))

def safe_call_llm(system_prompt, user_prompt):
    """
    Checks for PII before calling the LLM.
    Returns None if PII is detected.
    """

    if has_pii(user_prompt):
        print("Input blocked: PII detected.")
        return None

    return call_llm(system_prompt=system_prompt, user_prompt=user_prompt)


# TASK 2
# ======================================================================

# (B) Tabular Record Batch Scoring:

FEATURE_COLUMNS = [
    "CGPA",
    "Internships",
    "Projects",
    "Workshops/Certifications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "ExtracurricularActivities",
    "PlacementTraining",
    "SSC_Marks",
    "HSC_Marks",
]
 
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "risk_tier": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "flag_for_review": {
            "type": "boolean",
        },
        "primary_signal": {
            "type": "string",
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "recommended_action": {
            "type": "string",
        },
    },
    "required": [
        "risk_tier",
        "flag_for_review",
        "primary_signal",
        "confidence",
        "recommended_action",
    ],
    "additionalProperties": False,
}
 
SYSTEM_PROMPT = """ You are a placement-readiness risk assessor.
Return only one valid JSON object with exactly these fields:
risk_tier, flag_for_review, primary_signal, confidence, recommended_action.
 
Apply this business rubric:
- High risk: multiple weak readiness indicators, especially CGPA below 7.0,
  aptitude below 65, soft-skills rating below 3.5, no placement training, or
  little project/internship experience.
- Medium risk: a mixed profile with some strengths and some material gaps.
- Low risk: consistently strong readiness indicators, especially CGPA at least
  8.0, aptitude at least 80, soft-skills rating at least 4.0, placement
  training, and meaningful project/internship experience.
- flag_for_review must be true for high risk and for ambiguous medium-risk
  profiles; otherwise it must be false.
- confidence must be low, medium, or high based on how clearly the record fits
  the rubric.
- primary_signal must identify the single most influential observed feature.
- recommended_action must be a short, practical intervention.
Do not infer protected characteristics or facts that are absent from the input.
 
Worked example:
Input:
{"CGPA": 6.8, "Internships": 0, "Projects": 1,
"Workshops/Certifications": 0, "AptitudeTestScore": 61,
"SoftSkillsRating": 3.1, "ExtracurricularActivities": "no",
"PlacementTraining": "no", "SSC_Marks": 62, "HSC_Marks": 64}
Output:
{"risk_tier": "high", "flag_for_review": true,
"primary_signal": "AptitudeTestScore is below the readiness threshold",
"confidence": "high",
"recommended_action": "Complete aptitude coaching and a placement-training program"}
 
Assess every later input strictly with the rubric and output JSON only."""
 
USER_PROMPT_TEMPLATE = """Assess this dataset record using the rubric.
Return only the required JSON assessment.
 
Record: {record_json}"""

# TASK 3
# ======================================================================

# Current script directory
current_dir = Path(__file__).parent

# Go to the parent directory, then into part1
csv_path = current_dir.parent / "part1" / "cleaned_data.csv"

# Read the CSV into a DataFrame
cleaned_df = pd.read_csv(csv_path)

# Verify it loaded correctly
print(cleaned_df.head())

FALLBACK_RESPONSE = {
    "risk_tier": None,
    "flag_for_review": None,
    "primary_signal": None,
    "confidence": None,
    "recommended_action": None,
}

records = cleaned_df[FEATURE_COLUMNS].iloc[:3]
results = []

for index, row in records.iterrows():

    # Convert dataset row to JSON
    record = row.to_dict()
    record_json = json.dumps(record, indent=2)

    # Create user prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(
        record_json=record_json
    )

    # Guardrail: Check for PII before calling the LLM
    if has_pii(user_prompt):
        print("Input blocked: PII detected.")
        response = None
    else:
        response = call_llm(SYSTEM_PROMPT, user_prompt)

    # Handle API failure
    if response is None:

        validation_status = "FAIL (LLM API call failed)"
        parsed_response = FALLBACK_RESPONSE.copy()
        raw_response = None

    else:

        raw_response = response.strip()

        # -------------------------
        # Parse JSON
        # -------------------------

        try:
            parsed_response = json.loads(raw_response)

            # -------------------------
            # Validate JSON Schema
            # -------------------------

            try:
                validate(
                    instance=parsed_response,
                    schema=OUTPUT_SCHEMA
                )

                validation_status = "PASS"

            except ValidationError as e:

                print(f"Schema Validation Error (Record {index})")
                print(e)

                parsed_response = FALLBACK_RESPONSE.copy()

                validation_status = f"FAIL (Schema Validation Error: {e.message})"

        except json.JSONDecodeError as e:

            print(f"JSON Decode Error (Record {index})")
            print(e)

            parsed_response = FALLBACK_RESPONSE.copy()

            validation_status = f"FAIL (JSON Decode Error: {e.msg})"

    # -------------------------
    # Print required output
    # -------------------------

    print("\n" + "=" * 80)
    print(f"Record {index + 1}")

    print("\nInput Record:")
    print(record_json)

    print("\nRaw LLM Response:")
    print(raw_response)

    print("\nValidation Outcome:")
    print(validation_status)

    # Store for README/report
    results.append({
        "Input Record": record,
        "LLM Assessment JSON": parsed_response,
        "Validation Status": validation_status,
    })

print("\n" + "=" * 80)
print("Summary")

for i, result in enumerate(results, start=1):
    print(f"Record {i}: {result['Validation Status']}")


# TASK 4
# ======================================================================
record_with_email = {
    "Name": "John Doe",
    "Email": "john.doe@gmail.com",
    "CGPA": 8.4,
    "Internships": 2,
    "Projects": 4,
    "Workshops/Certifications": 3,
    "AptitudeTestScore": 82,
    "SoftSkillsRating": 4.2,
    "ExtracurricularActivities": "yes",
    "PlacementTraining": "yes",
    "SSC_Marks": 88,
    "HSC_Marks": 86
}

user_prompt = USER_PROMPT_TEMPLATE.format(
    record_json=json.dumps(record_with_email, indent=2)
)

print("=" * 80)
print("TEST 1 : INPUT WITH EMAIL")
print("=" * 80)

response = safe_call_llm(
    SYSTEM_PROMPT,
    user_prompt
)

print("Response:", response)

record_without_pii = {
    "CGPA": 8.4,
    "Internships": 2,
    "Projects": 4,
    "Workshops/Certifications": 3,
    "AptitudeTestScore": 82,
    "SoftSkillsRating": 4.2,
    "ExtracurricularActivities": "yes",
    "PlacementTraining": "yes",
    "SSC_Marks": 88,
    "HSC_Marks": 86
}

user_prompt = USER_PROMPT_TEMPLATE.format(
    record_json=json.dumps(record_without_pii, indent=2)
)

print("\n" + "=" * 80)
print("TEST 2 : INPUT WITHOUT PII")
print("=" * 80)

response = safe_call_llm(
    SYSTEM_PROMPT,
    user_prompt
)

print("Response:")
print(response)