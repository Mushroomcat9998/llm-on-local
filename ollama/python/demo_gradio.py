import asyncio
import json

import gradio as gr
import requests

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL_NAME = "qwen2.5:3b"


async def chatbot(message, history):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": message}],
        "stream": True,
    }
    response = requests.post(OLLAMA_URL, json=payload, stream=True)
    bot_response = ""

    for line in response.iter_lines():
        if line:
            try:
                data = line.decode("utf-8").strip()
                if data.startswith("data:"):
                    data = data[len("data:"):].strip()
                if data == "[DONE]":
                    break
                json_data = json.loads(data)
                token = json_data["choices"][0]["delta"].get("content", "")
                bot_response += token
                updated_history = history + [(message, bot_response)]
                # Yield two outputs: clear the textbox and update the chatbot history
                yield "", updated_history
                await asyncio.sleep(0)
            except Exception as e:
                print("Error processing token:", e)
    # Final yield to ensure complete history is returned
    yield "", history + [(message, bot_response)]


with gr.Blocks() as demo:
    with gr.Tab("Chatbot"):
        gr.Markdown(value="## Ollama Chatbot")
        with gr.Row():
            with gr.Column():
                msg = gr.Textbox(placeholder="Type your message...")
                btn = gr.Button("Send")
            chatbot_box = gr.Chatbot(height=800)

        btn.click(chatbot, inputs=[msg, chatbot_box], outputs=[msg, chatbot_box])

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
