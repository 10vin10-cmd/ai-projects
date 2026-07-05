import os
from config import Config
from model import llm, chat_template, get_ai_response, get_ai_response_modern

system_msg = "You are a helpful assistant. Be concise."
user_msg = "What is the capital of France?"

def run_test():
    print("=" * 50)
    print("🚀 RUNNING LLM CONFIGURATION TEST")
    print("=" * 50)
    
    # 1. Print current settings to verify config parsing
    print(f"[CONFIG] Selected Provider: {Config.LLM_PROVIDER}")
    print(f"[CONFIG] Target Model ID:  {Config.MODEL_ID}")
    
    # 2. Basic sanity check on API Keys
    if Config.LLM_PROVIDER == "openai":
        key = os.getenv("OPENAI_API_KEY")
        print(f"[AUTH]   Checking OpenAI Key: {'✅ Found' if key else '❌ Missing'}")
    elif Config.LLM_PROVIDER == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        print(f"[AUTH]   Checking Anthropic Key: {'✅ Found' if key else '❌ Missing'}")
    
    print("-" * 50)
    print("[EXECUTION] Sending a structured test prompt...")
    
    # 3. Test execution using your structural model.py functions
    try:
        #system_msg = "You are a helpful assistant. Be concise."
        #user_msg = "What is the capital of France?"
        
        response_data = get_ai_response(
            model=llm,
            template=chat_template,
            system_prompt=system_msg,
            user_prompt=user_msg
        )
        
        print("\n🎉 TEST SUCCESSFUL for structured response!")
        #print(f"[RESPONSE]: {response_data.strip()}")
        print(f"Type of response: {type(response_data)}")
        print(f"Summary:    {response_data.get('summary')}")
        print(f"Sentiment:  {response_data.get('sentiment')}/100")
        print(f"Response:   {response_data.get('response')}")
    except Exception as e:
        print("\n❌ TEST 1 FAILED!")
        print(f"[ERROR]: {str(e)}")
        print("\n💡 Troubleshooting Tip:")
        print("Ensure your .env keys are correct and match PRIMARY_LLM_PROVIDER.")
    print("=" * 50)

    try:
        response_data_modern = get_ai_response_modern(
            model=llm,
            system_prompt=system_msg,
            user_prompt=user_msg
        )
        print("\n🎉 MODERN TEST SUCCESSFUL!")
        # response_data is a true Pydantic Object instance now
        print(f"Type of response: {type(response_data_modern)}")
        print(f"Summary:    {response_data_modern.summary}")
        print(f"Sentiment:  {response_data_modern.sentiment}/100")
        print(f"Response:   {response_data_modern.response}")
    except Exception as e:
        print("\n❌ TEST 2 FAILED!")
        print(f"[ERROR]: {str(e)}")
        print("\n💡 Troubleshooting Tip:")
        print("Ensure your .env keys are correct and match PRIMARY_LLM_PROVIDER.")
    print("=" * 50)



if __name__ == "__main__":
    run_test()
