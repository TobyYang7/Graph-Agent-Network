import json
import random
import os
from openai import OpenAI

class Agent:
    def __init__(self, system_prompt="You are a helpful assistant."):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'config.json')
        
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
        
        self.api_key = random.choice(self.config['api_key'])
        self.model_name = self.config['model_name']
        self.base_url = self.config['base_url']
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.default_system_prompt = system_prompt

    def get_response(self, user_input=None, system_prompt=None, **kwargs):
        try:
            if system_prompt is None:
                system_prompt = self.default_system_prompt

            params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                "temperature": 0.7,
                "max_tokens": 150,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }

            params.update(kwargs)
            response = self.client.chat.completions.create(**params)
            
            return {
                "content": response.choices[0].message.content,
                "total_tokens": response.usage.total_tokens
            }
        except Exception as e:
            return {"error": f"Error: {str(e)}"}