"""
Utils file for the semantic negotiation cognition agent.
"""


from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import re
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

def openai_llm_provider(prompt: str, model: Optional[str] = None) -> str:
    """
    OpenAI LLM provider using the modern OpenAI API.
    
    Supports:
    - Direct OpenAI API
    - LiteLLM proxy (set OPENAI_BASE_URL)
    - Other OpenAI-compatible endpoints
    
    Requires: OPENAI_API_KEY in .env file
    Optional: OPENAI_MODEL in .env file (defaults to gpt-4)
    Optional: OPENAI_BASE_URL for custom endpoints (e.g., LiteLLM)
    """
    # Force reload environment variables
    load_dotenv(override=True)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Set it in .env file or export OPENAI_API_KEY='your-key'"
        )
    
    # Use provided model, or fall back to env var, or use gpt-4 as default
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4")
    
    # Support custom base URL (e.g., for LiteLLM proxy)
    base_url = os.getenv("OPENAI_BASE_URL")
    
    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=8000  # Ensure detailed itineraries and complex responses aren't truncated
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI API: {str(e)}"


def bedrock_llm_provider(prompt: str, model: Optional[str] = None) -> str:
    """
    AWS Bedrock LLM provider for Claude and other models.
    
    Requires:
    - AWS credentials configured (via ~/.aws/credentials or environment variables)
    - AWS_REGION in .env file (optional, defaults to us-east-1)
    - BEDROCK_MODEL in .env file (e.g., anthropic.claude-3-sonnet-20240229-v1:0)
    
    AWS Authentication methods (in order of precedence):
    1. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
    2. AWS credentials file: ~/.aws/credentials
    3. IAM role (if running on AWS)
    """
    if not BOTO3_AVAILABLE:
        return "Error: boto3 not installed. Install with: pip install boto3"
    
    # Get AWS region and model
    region = os.getenv("AWS_REGION", "us-east-1")
    model_id = model or os.getenv("BEDROCK_MODEL", "anthropic.claude-3-sonnet-20240229-v1:0")
    
    try:
        # Create Bedrock Runtime client
        bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=region
        )
        
        # Format request based on model family
        if "anthropic.claude" in model_id:
            # Claude 3 format
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 8000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0
            })
        else:
            # Generic format for other models
            body = json.dumps({
                "prompt": prompt,
                "max_tokens": 8000,
                "temperature": 0
            })
        
        # Call Bedrock
        response = bedrock.invoke_model(
            modelId=model_id,
            body=body
        )
        
        # Parse response
        response_body = json.loads(response["body"].read())
        
        # Extract content based on model family
        if "anthropic.claude" in model_id:
            return response_body["content"][0]["text"]
        else:
            return response_body.get("completion", str(response_body))
            
    except Exception as e:
        return f"Error calling AWS Bedrock: {str(e)}"


def get_llm_provider() -> Callable[[str], str]:
    """
    Get the appropriate LLM provider based on configuration.
    
    Checks LLM_PROVIDER in .env:
    - 'openai' -> OpenAI provider
    - 'bedrock' -> AWS Bedrock provider
    - defaults to 'openai'
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if provider == "bedrock":
        return bedrock_llm_provider
    elif provider == "openai":
        return openai_llm_provider
    else:
        print(f"Warning: Unknown LLM_PROVIDER '{provider}', defaulting to OpenAI")
        return openai_llm_provider