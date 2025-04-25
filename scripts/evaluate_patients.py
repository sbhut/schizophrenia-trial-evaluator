from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import json
import tiktoken
import os

load_dotenv()

# Initialize the OpenAI client with OpenRouter
client = OpenAI(
    base_url=os.getenv("ENDPOINT"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Define the model (DeepSeek R1 Zero)
model = "deepseek/deepseek-chat-v3-0324:free"

# Load patient data directly from CSV
csv_file_path = "discharge_patient_data.csv"
patient_df = pd.read_csv(csv_file_path)

# Get all unique subject_id values
unique_subject_ids = patient_df['subject_id'].drop_duplicates()

# Group the DataFrame by 'subject_id'
grouped_patients_df = patient_df.groupby('subject_id')

# Load the JSON file containing the criteria
with open('NCT00249288.json', 'r') as file:
    criteria_data = json.load(file)

# Extract the eligibility criteria
eligibility_criteria = criteria_data['protocolSection']['eligibilityModule']['eligibilityCriteria']

# Split the criteria by "Exclusion Criteria:"
parts = eligibility_criteria.split('Exclusion Criteria:')

# Initialize lists to hold inclusion and exclusion criteria
inclusion_criteria = parts[0].split('\n')
exclusion_criteria = ['Exclusion Criteria:'] + parts[1].split('\n') if len(parts) > 1 else []

# Remove any empty lines
inclusion_criteria = [line for line in inclusion_criteria if line.strip()]
exclusion_criteria = [line for line in exclusion_criteria if line.strip()]


# Function to create prompt using all patient data
def create_schizophrenia_prompt(patient_data):
    # Convert the entire patient group into a string
    patient_info = patient_data.to_csv(index=False)

    # Create the prompt including the full patient data
    return f"""
You are an AI medical assistant evaluating patients for a clinical trial. Your task is to determine whether each patient meets the inclusion criterion based on the provided data.

Inclusion criteria:
{inclusion_criteria[4]}

Patient Data:
{patient_info}

Guidelines for Evaluation:
1. Assess the patient's SANS score on the global assessment subscales.
2. A score of 3 or higher (moderate or greater severity) on at least one subscale (excluding attention) is required.
3. Do not consider positive symptoms when determining eligibility.

Instructions:
1. Evaluate if the patient meets the criterion based on the available data, including discharge notes. Do not default to "Not Enough Information" unless absolutely necessary.
2. Respond with one of the following for each criterion:
   - "Meets Criterion: Yes" if the patient meets the inclusion criterion.
   - "Meets Criterion: No" if the patient does not meet the inclusion criterion.
   - "Meets Criterion: Not Enough Information" if there is insufficient information to decide.
   
Ensure the final decision is consistent with the justification.

Response Format (Strictly Follow This Format):
Patient ID: [ID]
Criteria: [criteria]
Meets Criterion: [Yes, No, or Not Enough Information]
Justification: [Brief explanation based on patient data and medical knowledge]

"""

# Function to count tokens in a prompt using tiktoken
def count_tokens(prompt):
    encoding = tiktoken.encoding_for_model("gpt-4")  # Use "gpt-4" as a proxy for DeepSeek token counting
    return len(encoding.encode(prompt))

# Specify the number of patients to check
num_patients =100

# List to hold results
results = []
valid_subject_ids = []
total_tokens = 0

# Process each subject_id separately until we have 100 valid ones
for subject_id in unique_subject_ids:
    if len(valid_subject_ids) >= num_patients:
        break
    group = grouped_patients_df.get_group(subject_id)
    # Create prompt using the full group of patient data
    prompt = create_schizophrenia_prompt(group)

    # Count tokens for the prompt
    tokens = count_tokens(prompt)
    total_tokens += tokens

    # Check if the prompt length exceeds the maximum allowed tokens
    if tokens > 128000:
        print(f"Skipping subject_id {subject_id} due to prompt length exceeding the token limit.")
        print("=" * 50)
        continue

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            top_p=1.0,
            max_completion_tokens=2048
        )

        # Print the full API response for debugging
        print("API Response:", response)

        if response and response.choices:
            result = response.choices[0].message.content.strip()
            print(result)
        else:
            print(f"No valid response for subject_id {subject_id}. Skipping...")
            result = "No response received"

    except Exception as e:
        print(f"Error for subject_id {subject_id}: {e}")
        result = "API error"

    # Parse the result safely
    lines = result.split('\n')
    result_dict = {}
    for line in lines:
        if ': ' in line:
            key, value = line.split(': ', 1)
            result_dict[key.strip()] = value.strip()

    # Append the result to the list
    results.append({
        'patient_id': subject_id,
        'criteria': result_dict.get('Criteria', ''),
        'meets_criterion': result_dict.get('Meets Criterion', ''),
        'justification': result_dict.get('Justification', '')
    })
    valid_subject_ids.append(subject_id)

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('Fourth_Inclusion_Criteria1.csv', index=False)

print(f"Total tokens used for the input data of {num_patients} patients: {total_tokens}")
