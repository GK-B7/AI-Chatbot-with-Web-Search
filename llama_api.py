from fastapi import FastAPI, Body
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
import threading
from search_service import SearchService

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
model_id = "meta-llama/Llama-3.2-1B-Instruct"
device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id, token=HUGGINGFACE_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_id, token=HUGGINGFACE_TOKEN)
model = model.to(device)
search_service = SearchService(model=model, tokenizer=tokenizer)

app = FastAPI(title="Llama 3.2-1B Instruct Streaming API")

#Class to validate input json schema - mimic openai chat completions
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    use_search: Optional[bool] = False

#Function to create streaming response 
def generate_response(prompt: str, search_results: List[dict] = None, is_search_query: bool = False):
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        **inputs,streamer=streamer,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=4000,
        temperature=0.7,top_p=0.95,repetition_penalty=1.1,
        do_sample=True,
    )
    #Generate in different threads - needed for streaming
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    #Return streaming data mimicing openai
    if is_search_query:
        search_indicator = {
            "choices": [{
                "delta": {"search_performed": True},
                "index": 0,
                "finish_reason": None
            }],
            "object": "chat.completion.chunk"
        }
        yield f"{json.dumps(search_indicator)}\n\n"
    
    try:
        for new_text in streamer:
            if new_text:
                chunk = {
                    "choices": [{
                        "delta": {"content": new_text},
                        "index": 0,
                        "finish_reason": None
                    }],
                    "object": "chat.completion.chunk"
                }
                yield f"{json.dumps(chunk)}\n\n"
        
        thread.join()
        
        if search_results:
            citations = "\n\n**Sources:**\n\n"
            for i, result in enumerate(search_results, 1):
                title = result.get('title', 'Unknown Title')
                link = result.get('link', '')
                citations += f"• {title}"
                citations += f" - {link}\n\n"
            
            if citations:
                sources_chunk = {
                    "choices": [{
                        "delta": {"content": citations},
                        "index": 0,
                        "finish_reason": None
                    }],
                    "object": "chat.completion.chunk"
                }
                yield f"{json.dumps(sources_chunk)}\n\n"
        
        final_chunk = {
            "choices": [{
                "delta": {},
                "index": 0,
                "finish_reason": "stop"
            }],
            "object": "chat.completion.chunk"
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        
    except Exception as e:
        print(f"Generation error: {e}")
        error_chunk = {
            "choices": [{
                "delta": {"content": f"\n\n[Generation interrupted: {str(e)}]"},
                "index": 0,
                "finish_reason": "error"
            }],
            "object": "chat.completion.chunk"
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest = Body(...)):
    messages = request.messages
    force_search = request.use_search
    
    user_query = ""
    for msg in reversed(messages):
        if msg.role == "user":
            user_query = msg.content
            break
    
    #Checks if search is needed or not
    needs_search = search_service.should_search(user_query, force_search)
    search_results = []
    search_context = ""
    
    #Extract search context and pass to llm api
    if needs_search:
        search_results = search_service.search_web(user_query, num_results=5)
        if search_results:
            search_context = "Current web search results (please use this information to provide accurate, up-to-date answers):\n\n"
            for i, result in enumerate(search_results, 1):
                search_context += f"{result['title']}\n"
                search_context += f" Content: {result['snippet']}\n"
                search_context += f" Source: {result['link']}\n"
                search_context += "\n"
            search_context += "Please synthesize this information with your knowledge and provide a comprehensive answer.\n\n"
    chat = [{"role": m.role, "content": m.content} for m in messages]
    
    if search_context:
        search_system_message = {
            "role": "system",
            "content": f"You are a helpful assistant. Chat with the user and answer questions if any. Context: {search_context}"
        }
        chat.insert(0, search_system_message)
    
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    #Pass data in streaming to frontend
    return StreamingResponse(
        generate_response(prompt, search_results, needs_search),media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
