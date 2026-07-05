from shared_functions import *
from typing import List, Dict, Any
import json
import sys
import os

# 1. Dynamically append path to resolve your core project config settings
flask_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'genai_flask_app'))
if flask_app_path not in sys.path:
    sys.path.append(flask_app_path)
from config import Config
from model import initialize_any_model

# Global variables
food_items = []

# 2. Initialize your dynamic LiteLLM engine (Replacing ModelInference entirely)
# This binds to f"{Config.LLM_PROVIDER}/{Config.MODEL_ID}" automatically
model = initialize_any_model(
    provider=Config.LLM_PROVIDER,
    model_name=Config.MODEL_ID
)

# Apply generation parameters natively using LangChain's configuration binding properties
model.max_tokens = 400
model.temperature = 0.5

def main():
    """Main function for enhanced RAG chatbot system"""
    try:
        print("🤖 Enhanced RAG-Powered Food Recommendation Chatbot")
        print("   Powered by LiteLLM Proxy Engine & ChromaDB")
        print("=" * 55)
        
        # Load food data
        global food_items
        food_items = load_food_data('./FoodDataSet.json')
        print(f"✅ Loaded {len(food_items)} food items")
        
        # Create collection for RAG system
        collection = create_similarity_search_collection(
            "enhanced_rag_food_chatbot",
            {'description': 'Enhanced RAG chatbot with dynamic LiteLLM integration'}
        )
        populate_similarity_collection(collection, food_items)
        print("✅ Vector database ready")
        
        # Test LLM connection
        print("🔗 Testing LLM connection...")
        try:
            # Fix connection validation string handling matching LiteLLM syntax
            test_response = model.invoke("Hello")
            if test_response and test_response.content:
                print("✅ LLM connection established")
            else:
                print("❌ LLM connection failed: Received empty content block")
                return
        except Exception as conn_error:
            print(f"❌ LLM connection failed: {conn_error}")
            return
        
        # Start enhanced RAG chatbot
        enhanced_rag_food_chatbot(collection)
        
    except Exception as error:
        print(f"❌ Error: {error}")

def prepare_context_for_llm(query: str, search_results: List[Dict]) -> str:
    """Prepare structured context from search results for LLM"""
    if not search_results:
        return "No relevant food items found in the database."
    
    context_parts = []
    context_parts.append("Based on your query, here are the most relevant food options from our database:")
    context_parts.append("")
    
    for i, result in enumerate(search_results[:3], 1):
        food_context = []
        food_context.append(f"Option {i}: {result['food_name']}")
        food_context.append(f"  - Description: {result['food_description']}")
        food_context.append(f"  - Cuisine: {result['cuisine_type']}")
        food_context.append(f"  - Calories: {result['food_calories_per_serving']} per serving")
        
        if result.get('food_ingredients'):
            ingredients = result['food_ingredients']
            if isinstance(ingredients, list):
                food_context.append(f"  - Key ingredients: {', '.join(ingredients[:5])}")
            else:
                food_context.append(f"  - Key ingredients: {ingredients}")
        
        if result.get('food_health_benefits'):
            food_context.append(f"  - Health benefits: {result['food_health_benefits']}")
        
        if result.get('cooking_method'):
            food_context.append(f"  - Cooking method: {result['cooking_method']}")
        
        if result.get('taste_profile'):
            food_context.append(f"  - Taste profile: {result['taste_profile']}")
        
        food_context.append(f"  - Similarity score: {result['similarity_score']*100:.1f}%")
        food_context.append("")
        
        context_parts.extend(food_context)
    
    return "\n".join(context_parts)

def generate_llm_rag_response(query: str, search_results: List[Dict]) -> str:
    """Generate response using configured LiteLLM with retrieved context"""
    try:
        # Prepare context from search results
        context = prepare_context_for_llm(query, search_results)
        
        # Build the prompt structure
        prompt = f'''You are a helpful food recommendation assistant. A user is asking for food recommendations, and I've retrieved relevant options from a food database.

User Query: "{query}"

Retrieved Food Information:
{context}

Please provide a helpful, short response that:
1. Acknowledges the user's request
2. Recommends 2-3 specific food items from the retrieved options
3. Explains why these recommendations match their request
4. Includes relevant details like cuisine type, calories, or health benefits
5. Uses a friendly, conversational tone
6. Keeps the response concise but informative

Response:'''

        # Generate response using your active cloud configuration instance via .invoke()
        generated_response = model.invoke(prompt)
        
        # Extract the raw plain text content safely out of the AIMessage instance
        if generated_response and generated_response.content:
            response_text = generated_response.content.strip()
            
            # If response is too short, provide a fallback
            if len(response_text) < 50:
                return generate_fallback_response(query, search_results)
            
            return response_text
        else:
            return generate_fallback_response(query, search_results)
            
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return generate_fallback_response(query, search_results)

def generate_fallback_response(query: str, search_results: List[Dict]) -> str:
    """Generate fallback response when LLM fails"""
    if not search_results:
        return "I couldn't find any food items matching your request. Try describing what you're in the mood for with different words!"
    
    top_result = search_results[0]
    response_parts = []
    
    response_parts.append(f"Based on your request for '{query}', I'd recommend {top_result['food_name']}.")
    response_parts.append(f"It's a {top_result['cuisine_type']} dish with {top_result['food_calories_per_serving']} calories per serving.")
    
    if len(search_results) > 1:
        second_choice = search_results[1]
        response_parts.append(f"Another great option would be {second_choice['food_name']}.")
    
    return " ".join(response_parts)

def enhanced_rag_food_chatbot(collection):
    """Enhanced RAG-powered conversational food chatbot wrapper"""
    print("\n" + "="*70)
    print("🤖 ENHANCED RAG FOOD RECOMMENDATION CHATBOT")
    print("   Powered by Custom LiteLLM Configuration Backend")
    print("="*70)
    print("💬 Ask me about food recommendations using natural language!")
    print("\nExample queries:")
    print("  • 'I want something spicy and healthy for dinner'")
    print("  • 'What Italian dishes do you recommend under 400 calories?'")
    print("\nCommands:")
    print("  • 'help' - Show detailed help menu")
    print("  • 'compare' - Compare recommendations for two different queries")
    print("  • 'quit' - Exit the chatbot")
    print("-" * 70)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                print("🤖 Bot: Please tell me what kind of food you're looking for!")
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n🤖 Bot: Thank you for using the Enhanced RAG Food Chatbot!")
                print("      Hope you found some delicious recommendations! 👋")
                break
            
            elif user_input.lower() in ['help', 'h']:
                show_enhanced_rag_help()
            
            elif user_input.lower() in ['compare']:
                handle_enhanced_comparison_mode(collection)
            
            else:
                # Add parsing code snippet here to execute the full retrieval query pipeline
                # (Matches how your shared_functions module passes the query)
                results = perform_similarity_search(collection, user_input) #results = search_similarity_collection(collection, user_input)
                bot_response = generate_llm_rag_response(user_input, results)
                print(f"\n🤖 Bot: {bot_response}")
                
        except Exception as loop_error:
            print(f"🤖 Bot Error processing input: {loop_error}")

if __name__ == '__main__':
    main()
