import asyncio
import json

import gradio as gr
import requests

VLLM_URL = "http://localhost:8000/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"


async def chat_with_llm(message, history):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    if history:
        for user_msg, bot_reply in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_reply})

    # Append the latest user message
    messages.append({"role": "user", "content": message})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.2,
        "stream": True,
    }
    response = requests.post(VLLM_URL, headers=HEADERS, data=json.dumps(payload), stream=True)

    bot_message = ""
    for line in response.iter_lines():
        if line:
            try:
                data = line.decode("utf-8").replace("data: ", "").strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)["choices"][0]["delta"]["content"]
                bot_message += chunk
                yield bot_message  # Stream partial response
                await asyncio.sleep(0)
            except Exception as e:
                print(f"Error processing response chunk: {e}")


gr.ChatInterface(fn=chat_with_llm, title="vLLM Chatbot").launch(share=True)
