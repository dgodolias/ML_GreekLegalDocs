import os
import json
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv


dotenv_path = find_dotenv(raise_error_if_not_found=False) 
print(f"Found .env file at: {dotenv_path}")

api_key_before_load = os.getenv("OPENROUTER_API_KEY")

if dotenv_path:
    load_dotenv(dotenv_path=dotenv_path, override=True)
    print(f"Loaded .env file from: {dotenv_path} with override=True")
else:

    load_dotenv(override=True)
    print("Could not specifically locate .env, attempting default load_dotenv(override=True)")


api_key = os.getenv("OPENROUTER_API_KEY")


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# Available models with fallbacks
AVAILABLE_MODELS = {
    "primary": "google/gemma-3-4b-it:free",
    "fallback": "meta-llama/llama-3.3-8b-instruct:free"
}

def send_message_to_llm(user_message, system_message="You are a helpful assistant", 
                       model=AVAILABLE_MODELS["primary"], max_tokens=500):
    """
    Sends a message to the language model and returns the response.
    
    Args:
        user_message (str): The message to send to the LLM
        system_message (str): The system message to set the LLM's behavior
        model (str): The model identifier to use with OpenRouter
        max_tokens (int): Maximum tokens to generate in the response
        
    Returns:
        str: The LLM's response text
    """
    try:
        print(f"Sending request to OpenRouter API using model: {model}")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            stream=False
        )
        
        # Check if response and choices exist
        if response and hasattr(response, 'choices') and response.choices:
            if len(response.choices) > 0 and response.choices[0].message:
                result = response.choices[0].message.content.strip()
                print(f"API response received successfully")
                return result
        print(response)
        print("Empty or invalid response structure received from API")
        return ""
        
    except Exception as e:
        print(f"Error calling OpenRouter API: {e}")
        # Try with fallback model if primary model fails
        if model != AVAILABLE_MODELS["fallback"]:
            print(f"Trying fallback model: {AVAILABLE_MODELS['fallback']}")
            return send_message_to_llm(
                user_message=user_message,
                system_message=system_message,
                model=AVAILABLE_MODELS["fallback"],
                max_tokens=max_tokens
            )
        return ""
    
# example usage
if __name__ == "__main__":
    user_message = "What is the capital of France?"
    system_message = "You are a helpful assistant."
    
    response = send_message_to_llm(user_message, system_message)
    print(f"LLM Response: {response}")
