from typing import List
from pydantic import BaseModel

import ollama
from ollama import chat
from ollama import ChatResponse

import ogpu.service


MODEL_NAME = "qwen2.5:3b"

@ogpu.service.init()
def setup():
    ogpu.service.logger.info(f"Pulling {MODEL_NAME} model...")
    ollama.pull(MODEL_NAME)
    ogpu.service.logger.info(f"{MODEL_NAME} pulled.")


class Message(BaseModel):
    role: str = "user"
    content: str = "what is the capital of France?"

class InputData(BaseModel):
    messages: List[Message]


@ogpu.service.expose()
def text2text(input_data: InputData) -> Message:

    ogpu.service.logger.info(f"Generating Text2Text response..")
    response: ChatResponse = chat(model=MODEL_NAME, messages=input_data.messages)
    try:
        output = Message(
            role=response['message']['role'],
            content=response['message']['content']
        )
        ogpu.service.logger.info(f"Task completed.")
    except Exception as e:
        ogpu.service.logger.error(f"An error occurred: {e}")
        raise e

    return output

ogpu.service.start()