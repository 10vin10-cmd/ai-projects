import litellm
from textwrap import dedent
from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic

load_dotenv()

print(f"Libraries and environment variables loaded successfully")


print(os.getenv("OPENAI_API_KEY")[:9])

MODEL_NAME = ChatAnthropic(model="gpt-4o-mini") 

def get_completion(prompt, model, max_tokens=20):
    print("--- Getting completion from LiteLLM---")

    response = litellm.completion(
        model=model,
        messages=[
            {
                "role":"user",
                "content":"You are a helpful travel assistant"
            },
            {
                "role":"user",
                "content":prompt
            }
        ],
        max_tokens=max_tokens
    )

    return response



user_prompt = "what is the capital of France, and what is it famous for?"


response = get_completion(
            user_prompt,
            model=MODEL_NAME
        )

      
print(response.choices[0].message.content)
        
    