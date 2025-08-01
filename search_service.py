from dotenv import load_dotenv
import os
import json
import requests
from typing import List, Dict
import torch

load_dotenv()

class SearchService:
    def __init__(self, model=None, tokenizer=None):
        self.serper_api_key = os.getenv("SERPER_API_KEY")
        self.serper_url = "https://google.serper.dev/search"
        self.model = model
        self.tokenizer = tokenizer
        
        #Keywords for identifying if search is needed or not
        self.search_keywords = [
            "latest", "recent", "current", "today", "now", "2024", "2025", "2026",
            "news", "price", "stock", "weather", "update", "breaking",
            "score", "buzz", "election",
            "event", "presently", "yesterday", "market",
            "trending", "popular", "live", "real-time", "ongoing",
            "happened", "year", "currently"
        ]

    #Function determining search is needed or not
    def should_search(self, query: str, force_search: bool = False) -> bool:
        if force_search:
            return True
        
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in self.search_keywords):
            return True
        
        if not self.model or not self.tokenizer:
            return False
        
        #AI generated prompt with a bit editing
        decision_prompt = f"""You are an AI assistant that determines whether a user query requires current web search or can be answered with existing knowledge.

Analyze this user query and respond with ONLY "YES" or "NO":

- Answer "YES" if the query asks for:
* Current events, recent news, or breaking updates
* Real-time information (stock prices, weather, sports scores)
* Recent developments or latest information about any topic
* Information that changes frequently or is time-sensitive
* Specific facts that might have changed recently
* Current status of ongoing situations

User Query: "{query}"

Decision (YES/NO only):"""
        
        try:   
            inputs = self.tokenizer(decision_prompt, return_tensors="pt")
            
            if hasattr(self.model, 'device'):
                inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.95,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            response = self.tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            response = response.strip().upper()
            
            if "YES" in response:
                return True
            elif "NO" in response:
                return False
            else:
                return False
                
        except Exception as e:
            return False

    #Function to search web and get results
    def search_web(self, query: str, num_results: int = 5) -> List[Dict]:
        if not self.serper_api_key:
            return []
        
        payload = json.dumps({
            "q": query,
            "num": num_results
        })
        
        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.request("POST",
                self.serper_url,
                data=payload,
                headers=headers
            )
            data = response.json()
            
            results = []
            
            if "organic" in data:
                for item in data["organic"]:
                    result = {
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "link": item.get("link", ""),
                        "position": item.get("position", 0)
                    }
                    results.append(result)
            return results
            
        except:
            return []
