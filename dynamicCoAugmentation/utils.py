import asyncio
import aiohttp
import openai
import google.generativeai as palm
from google.generativeai.types import safety_types
from google.api_core import retry
from config import APIConfig

import os
import json
from typing import Dict, List

# Load configuration
config = APIConfig()

# Configure OpenAI and PaLM APIs
openai.api_key = config.openai_api_key
palm.configure(api_key=config.palm_api_key)

# Retry settings for better error handling
RETRY_SETTINGS = retry.Retry(
    initial=1.0,  # Initial delay before retrying
    maximum=10.0,  # Maximum delay between retries
    multiplier=2,  # Multiplier for backoff
    deadline=60.0  # Total retry time (in seconds)
)

async def async_request_palm(message: str) -> str:
    """
    Asynchronous request to Google PaLM using aiohttp.
    :param message: The prompt message to be sent to the API.
    :return: Response from the PaLM API.
    """
    try:
        model = config.palm_model
        async with aiohttp.ClientSession() as session:
            url = f"https://palm-api-url/{model}/generate-text"
            headers = {
                "Authorization": f"Bearer {config.palm_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "prompt": message,
                "safety_settings": [
                    {
                        "category": safety_types.HarmCategory.HARM_CATEGORY_DEROGATORY,
                        "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": safety_types.HarmCategory.HARM_CATEGORY_VIOLENCE,
                        "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": safety_types.HarmCategory.HARM_CATEGORY_TOXICITY,
                        "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": safety_types.HarmCategory.HARM_CATEGORY_MEDICAL,
                        "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                    },
                ]
            }
            
            async with session.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()  # Raise error for bad status codes
                data = await response.json()
                if "candidates" not in data or len(data["candidates"]) < 1:
                    raise ValueError("No candidates found in response.")
                return data["candidates"][0]["output"]

    except Exception as e:
        print(f"Failed to request PaLM API: {e}")
        return ""

@RETRY_SETTINGS
def sync_request_palm(messages: str) -> str:
    """
    Synchronous request using PaLM SDK with retry logic.
    :param messages: Prompt message to be sent to the API.
    :return: Response from the PaLM API.
    """
    model = config.palm_model
    try:
        completion = palm.generate_text(
            model=model,
            prompt=messages,
            safety_settings=[
                {
                    "category": safety_types.HarmCategory.HARM_CATEGORY_DEROGATORY,
                    "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": safety_types.HarmCategory.HARM_CATEGORY_VIOLENCE,
                    "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": safety_types.HarmCategory.HARM_CATEGORY_TOXICITY,
                    "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": safety_types.HarmCategory.HARM_CATEGORY_MEDICAL,
                    "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                },
            ]
        )
        if len(completion.candidates) < 1:
            print("No candidates received.")
            return ""

        return completion.candidates[0]['output']
    
    except Exception as e:
        print(f"API Request failed: {e}")
        return ""


def read_literature_files(years: List[int], base_path: str) -> Dict[int, List[Dict]]:
    """
    Reads and parses literature files from the specified base directory for given years.
    
    Args:
        years (List[int]): List of years to read literature files for.
        base_path (str): The directory where the literature files are stored.
        
    Returns:
        Dict[int, List[Dict]]: A dictionary mapping each year to a list of literature entries.
    """
    year2literatures = {year: [] for year in years}
    
    for year in years:
        file_path = os.path.join(base_path, f'{year}.pubtator')
        try:
            with open(file_path, 'r') as f:
                literature = {'entity': {}}
                for line in f.readlines():
                    line = line.strip()
                    if line == '' and literature:
                        # Finalize the current literature entry
                        for entity_id in literature['entity']:
                            literature['entity'][entity_id]['entity_name'] = list(literature['entity'][entity_id]['entity_name'])
                        year2literatures[year].append(literature)
                        literature = {'entity': {}}
                        continue
                    if '|t|' in line:
                        literature['title'] = line.split('|t|')[1]
                    elif '|a|' in line:
                        literature['abstract'] = line.split('|a|')[1]
                    else:
                        line_list = line.split('\t')
                        if len(line_list) != 6:
                            entity_name, entity_type, entity_id = line_list[3], line_list[4], None
                        else:
                            entity_name, entity_type, entity_id = line_list[3], line_list[4], line_list[5]
                        if entity_id == '-':
                            continue
                        if entity_id not in literature['entity']:
                            literature['entity'][entity_id] = {'entity_name': set(), 'entity_type': entity_type}
                        literature['entity'][entity_id]['entity_name'].add(entity_name)
        except FileNotFoundError:
            print(f"File not found for year {year}: {file_path}")
            continue
    
    return year2literatures


def get_entity_name(entity_names: List[str]) -> str:
    """
    Returns a formatted string of entity names.
    
    Args:
        entity_names (List[str]): List of entity names.
        
    Returns:
        str: A string representation of the entity names.
    """
    if len(entity_names) == 1:
        return entity_names[0]
    else:
        return '{} ({})'.format(entity_names[0], ', '.join(entity_names[1:]))
    

def build_options(entity_relation: List[str]) -> Tuple[str, dict]:
    """
    Constructs multiple-choice options from a list of entity relations.
    
    Args:
        entity_relation (List[str]): List of relations between entities.
        
    Returns:
        Tuple[str, dict]: A formatted string of options and a dictionary mapping the options to the relations.
    """
    entity_relation_new = entity_relation + ['no-relation', 'others, please specify by generating a short predicate in 5 words']
    option_list = ['A. ', 'B. ', 'C. ', 'D. ', 'E. ']
    ret = ''
    option2relation = {}
    
    for r, o in zip(entity_relation_new, option_list):
        ret += o + r + '\n'
        option2relation[o.strip()] = r
        
    return ret.strip(), option2relation

