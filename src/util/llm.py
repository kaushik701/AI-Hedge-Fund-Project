"""Helper functions for LLM"""

import json
from typing import TypeVar, Type
from pydantic import BaseModel
from util.progress import progress

T = TypeVar('T', bound=BaseModel)

def call_llm(
    prompt,
    model_name,
    model_provider,
    pydantic_model,
    agent_name=None,
    max_retries=3,
    default_factory=None
):
    """Makes an LLM call with retry logic."""
    from llm.models import get_model, get_model_info
    
    # Get model and check if it supports JSON mode
    model_info = get_model_info(model_name)
    llm = get_model(model_name, model_provider)
    
    # Configure structured output if supported
    if model_info and model_info.has_json_mode():
        llm = llm.with_structured_output(pydantic_model, method="json_mode")
    
    # Try to call the LLM with retries
    for attempt in range(max_retries):
        try:
            # Call the LLM
            result = llm.invoke(prompt)
            
            # Handle non-JSON mode models (like DeepSeek)
            if model_info and not model_info.has_json_mode():
                parsed_json = extract_json_from_deepseek_response(result.content)
                if parsed_json:
                    return pydantic_model(**parsed_json)
            else:
                return result
                
        except Exception as e:
            # Update status if agent name is provided
            if agent_name:
                progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries}")
            
            # Last attempt failed, return default response
            if attempt == max_retries - 1:
                print(f"Error in LLM call after {max_retries} attempts: {e}")
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    # This should never be reached
    return create_default_response(pydantic_model)


def create_default_response(model_class):
    """Creates a default response based on the model's fields."""
    default_values = {}
    
    for field_name, field in model_class.model_fields.items():
        # Handle different field types
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), use the first allowed value if possible
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None
    
    return model_class(**default_values)


def extract_json_from_deepseek_response(content):
    """Extracts JSON from a markdown-formatted response."""
    try:
        # Find the JSON code block
        json_start = content.find("```json")
        if json_start == -1:
            return None
            
        # Extract the JSON content
        json_text = content[json_start + 7:]  # Skip past ```json
        json_end = json_text.find("```")
        if json_end == -1:
            return None
            
        # Parse the JSON
        json_text = json_text[:json_end].strip()
        return json.loads(json_text)
        
    except Exception as e:
        print(f"Error extracting JSON from response: {e}")
        return None