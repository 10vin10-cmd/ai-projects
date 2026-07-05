from flask import Flask, request, jsonify, render_template
from config import Config
from model import llm, chat_template, get_ai_response, get_ai_response_modern
import time

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    # This is where we'll add our AI logic later
    data = request.json
    user_message = data.get('message')
    model = llm

    if not user_message or not model:
        return jsonify({"error": "Missing message or model selection"}), 400
    system_msg = "You are an AI assistant helping with customer inquiries. Provide a helpful and concise response."
    
    start_time = time.time()
    try:
        response_obj = get_ai_response_modern(
            model=llm,
            system_prompt=system_msg,
            user_prompt=user_message
        )
        response_dict = response_obj.model_dump()
        
        # Inject performance metrics dynamically
        response_dict['duration'] = round(time.time() - start_time, 2)
        
        print("\n🎉 MODERN TEST SUCCESSFUL!")
        # response_data is a true Pydantic Object instance now
        print(f"Type of response: {type(response_obj)}")
        print(f"Summary:    {response_obj.summary}")
        print(f"Sentiment:  {response_obj.sentiment}/100")
        print(f"Response:   {response_obj.response}")
        
        return jsonify(response_dict)
    except Exception as e:
        print("\n❌ TEST 2 FAILED!")
        print(f"[ERROR]: {str(e)}") 
        print("\n💡 Troubleshooting Tip:")
        print("Ensure your .env keys are correct and match PRIMARY_LLM_PROVIDER.")
        return jsonify({"error": str(e)}), 500
    print("=" * 50)

if __name__ == '__main__':
    app.run(debug=True)

