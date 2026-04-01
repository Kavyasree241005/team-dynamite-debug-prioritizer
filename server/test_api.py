import google.generativeai as genai
import sys

def test_api():
    try:
        api_key = "AIzaSyD1HZ9DoBxs2MwOhfs72vgc6U2a07GgUNk"
        genai.configure(api_key=api_key)
        
        # Test basic models endpoint
        print("Available models:")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
        
        # Test specific model
        model_name = "gemini-1.5-flash"
        print(f"\nTesting generation with {model_name}...")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Say hello world.")
        print(f"Success! Response: {response.text}")
        
    except Exception as e:
        print("\n=== FULL TRACEBACK ===")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_api()
