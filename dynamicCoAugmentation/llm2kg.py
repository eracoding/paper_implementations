import os
import json
import asyncio
import logging
from typing import Dict, List
from tqdm import tqdm
from config import APIConfig
from models import Literature, RelationExtractionResult
from utils import async_request_palm
from utils import read_literature_files, get_entity_name, build_options

# Set up logging
logging.basicConfig(level=logging.INFO)
config = APIConfig()

ENTITY_MAP = {
    "Species": "anatomies",
    "Chromosome": "cellular components",
    "CellLine": "cellular components",
    "SNP": "biological processes",
    "ProteinMutation": "biological processes",
    "DNAMutation": "biological processes",
    "ProteinAcidChange": "biological processes",
    "DNAAcidChange": "biological processes",
    "Gene": "genes",
    "Chemical": "compounds",
    "Disease": "diseases"
}

ENTITIES_RELATION = {
    ("genes", "genes"): ["covaries", "interacts", "regulates"],
    ("diseases", "diseases"): ["resembles"],
    ("compounds", "compounds"): ["resembles"],
    ("genes", "diseases"): ["downregulates", "associates", "upregulates"],
    ("genes", "compounds"): ["binds", "upregulates", "downregulates"],
    ("compounds", "diseases"): ["treats", "palliates"]
}

VALID_TYPES = ["genes", "compounds", "diseases"]

TEMPLATE_SUMMARY = '''Read the following abstract, generate short summary about {} entity "{}" to illustrate what is {}'s relationship with other medical entity.
Abstract: {}
Summary: '''

TEMPLATE_RELATION_EXTRACTION = '''
Read the following summary, answer the following question.
Summary: {}
Question: predict the relationship between {} entity "{}" and {} entity "{}", first choose from the following options:
{}
Answer: Let's think step by step: '''

TEMPLATE_RELATION_EXTRACTION_ANSWER = '''
Read the following summary, answer the following question.
Summary: {}
Question: predict the relationship between {} entity "{}" and {} entity "{}", first choose from the following options:
{}
Answer: Let's think step by step: {}. So the answer is:'''

async def extract_relations(literature: Dict, demonstration: str) -> List[RelationExtractionResult]:
    extracted_items = []
    title, abstract = literature.get('title'), literature.get('abstract')
    
    for i, (entity1_id, entity1_info) in enumerate(literature['entity'].items()):
        entity1_names, entity1_type = entity1_info['entity_name'], entity1_info['entity_type']
        if entity1_type not in ENTITY_MAP:
            continue
        entity1_type_hetionet = ENTITY_MAP[entity1_type]
        if entity1_type_hetionet not in VALID_TYPES:
            continue
        entity1_name = get_entity_name(entity1_names)
        summary_prompt = TEMPLATE_SUMMARY.format(entity1_type, entity1_name, entity1_name, abstract)
        
        try:
            ret_summary = await async_request_palm(summary_prompt)
        except Exception as e:
            logging.error(f"Failed to get summary: {e}")
            continue
        
        for j, (entity2_id, entity2_info) in enumerate(literature['entity'].items()):
            if i == j:
                continue
            entity2_names, entity2_type = entity2_info['entity_name'], entity2_info['entity_type']
            if entity2_type not in ENTITY_MAP:
                continue
            entity2_type_hetionet = ENTITY_MAP[entity2_type]
            if (entity1_type_hetionet, entity2_type_hetionet) not in ENTITIES_RELATION:
                continue
            
            await asyncio.sleep(2)  # Respect rate limits
            entity2_name = get_entity_name(entity2_names)
            entity_relation = ENTITIES_RELATION[(entity1_type_hetionet, entity2_type_hetionet)]
            options, option2relation = build_options(entity_relation)
            
            relation_prompt = TEMPLATE_RELATION_EXTRACTION.format(ret_summary, entity1_type, entity1_name, entity2_type, entity2_name, options)
            
            try:
                ret_CoT = await async_request_palm(demonstration + relation_prompt)
            except Exception as e:
                logging.error(f"Failed to extract relation: {e}")
                continue
            
            if not ret_CoT:
                continue
            
            answer_prompt = TEMPLATE_RELATION_EXTRACTION_ANSWER.format(ret_summary, entity1_type, entity1_name, entity2_type, entity2_name, options, ret_CoT)
            
            try:
                ret_relation = await async_request_palm(demonstration + answer_prompt)
            except Exception as e:
                logging.error(f"Failed to get final relation answer: {e}")
                continue
            
            # Determine the correct relation
            is_generated = False
            relation = None
            for option, rel in option2relation.items():
                if option in ret_relation or option[0] == ret_relation[0] or rel in ret_relation:
                    if rel == 'others, please specify by generating a short predicate in 5 words':
                        if '.' in ret_relation:
                            relation = ret_relation.split('.')[1]
                        else:
                            relation = ret_relation
                        is_generated = True
                    else:
                        relation = rel
                    break
            
            if not relation:
                is_generated = True
                relation = ret_relation.strip()
                logging.warning(f"Not matching any known options: {ret_relation}")
            
            # Add extracted item
            extracted_items.append({
                'entity1': {
                    'entity_name': entity1_names,
                    'entity_type': entity1_type_hetionet,
                    'entity_id': entity1_id
                },
                'entity2': {
                    'entity_name': entity2_names,
                    'entity_type': entity2_type_hetionet,
                    'entity_id': entity2_id
                },
                'relation': relation,
                'is_generated': is_generated
            })
    
    return extracted_items

async def main():
    year2literatures = await read_literature_files(config.years, config.base_path)
    demonstration = json.load(open('demonstration.json'))
    demonstration = '\n\n'.join(demonstration) + '\n'

    all_extracted = []
    for year, literatures in year2literatures.items():
        tasks = [extract_relations(lit, demonstration) for lit in literatures]
        results = await asyncio.gather(*tasks)
        all_extracted.extend(results)
        
        # Save results per year
        os.makedirs(config.output_path, exist_ok=True)
        output_file = os.path.join(config.output_path, f'{year}_extracted.json')
        with open(output_file, 'w') as outfile:
            json.dump(all_extracted, outfile, indent=2)
        logging.info(f"Data for {year} saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
