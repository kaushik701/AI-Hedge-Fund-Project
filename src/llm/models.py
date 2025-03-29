import os
from langchain_community.chat_models import ChatDeepseek
from langchain_community.chat_models import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatGroq
from enum import Enum
from pydantic import BaseModel


class ModelProvider(str, Enum):
    """Supported LLM providers"""
    DEEPSEEK = "DeepSeek"
    GEMINI = "Gemini"
    GROQ = "Groq"


class LLMModel(BaseModel):
    """LLM model configuration"""
    display_name: str
    model_name: str
    provider: ModelProvider
    
    def to_choice_tuple(self):
        """Convert to UI choice format"""
        return (self.display_name, self.model_name, self.provider.value)
    
    def has_json_mode(self):
        """Check if JSON mode is supported"""
        return not self.is_deepseek() and not self.is_gemini()
    
    def is_deepseek(self):
        """Check if model is DeepSeek"""
        return self.model_name.startswith("deepseek")
    
    def is_gemini(self):
        """Check if model is Gemini"""
        return self.model_name.startswith("gemini")


# Define available models
AVAILABLE_MODELS = [
    LLMModel(display_name="[deepseek] deepseek-r1", model_name="deepseek-reasoner", provider=ModelProvider.DEEPSEEK),
    LLMModel(display_name="[deepseek] deepseek-v3", model_name="deepseek-chat", provider=ModelProvider.DEEPSEEK),
    LLMModel(display_name="[gemini] gemini-2.0-flash", model_name="gemini-2.0-flash", provider=ModelProvider.GEMINI),
    LLMModel(display_name="[gemini] gemini-2.0-pro", model_name="gemini-2.0-pro-exp-02-05", provider=ModelProvider.GEMINI),
    LLMModel(display_name="[groq] llama-3.3 70b", model_name="llama-3.3-70b-versatile", provider=ModelProvider.GROQ),
]

# Create choice tuples for UI
LLM_ORDER = [model.to_choice_tuple() for model in AVAILABLE_MODELS]


def get_model_info(model_name):
    """Find model info by name"""
    for model in AVAILABLE_MODELS:
        if model.model_name == model_name:
            return model
    return None


def get_model(model_name, model_provider):
    """Create model instance with appropriate API key"""
    provider_configs = {
        ModelProvider.GROQ: ("GROQ_API_KEY", ChatGroq),
        ModelProvider.DEEPSEEK: ("DEEPSEEK_API_KEY", ChatDeepseek),
        ModelProvider.GEMINI: ("GOOGLE_API_KEY", ChatGoogleGenerativeAI)
    }
    
    if model_provider not in provider_configs:
        return None
    
    env_var, model_class = provider_configs[model_provider]
    api_key = os.getenv(env_var)
    
    if not api_key:
        error_msg = f"API Key Error: Please make sure {env_var} is set in your .env file."
        print(error_msg)
        raise ValueError(f"{model_provider.value} API key not found. {error_msg}")
    
    return model_class(model=model_name, api_key=api_key)