from typing_extensions import Annotated, Sequence, TypedDict
import operator
from langchain_core.messages import BaseMessage
import json

def merge_dicts(a, b):
    """Merge two dictionaries."""
    return {**a, **b}

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[dict, merge_dicts]
    metadata: Annotated[dict, merge_dicts]

def show_agent_reasoning(output, agent_name):
    """Display agent reasoning in a readable format."""
    print(f"\n{'=' * 10} {agent_name.center(28)} {'=' * 10}")
    
    # Format output for display
    formatted_output = format_for_display(output)
    print(formatted_output)
    
    print("=" * 48)

def format_for_display(obj):
    """Convert objects to JSON-serializable format."""
    # Handle different types of data
    if isinstance(obj, (dict, list)):
        serializable_obj = convert_to_serializable(obj)
        return json.dumps(serializable_obj, indent=2)
    else:
        try:
            parsed_obj = json.loads(obj)
            return json.dumps(parsed_obj, indent=2)
        except (json.JSONDecodeError, TypeError):
            return obj

def convert_to_serializable(obj):
    """Recursively convert objects to JSON-serializable types."""
    if hasattr(obj, "to_dict"):  # Handle Pandas objects
        return obj.to_dict()
    elif hasattr(obj, "__dict__"):  # Handle custom objects
        return obj.__dict__
    elif isinstance(obj, (int, float, bool, str, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return str(obj)  # Fallback to string