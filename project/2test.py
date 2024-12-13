import openai
from openai import OpenAI
import os


# Set the environment variable
os.environ["OPENAI_API_KEY"] = "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"

client = OpenAI()

try:
  #Make your OpenAI API request here
  assistant = client.beta.assistants.create(
    name="CoC-Assistant",
    instructions="An assistant designed to play NPCs using game background knowledge.",
    model="gpt-4",  # Ensure this model is supported in your environment
    tools=[{"type": "file_search"}],
)
except openai.APIError as e:
  #Handle API error here, e.g. retry or log
  print(f"OpenAI API returned an API Error: {e}")
  pass
except openai.APIConnectionError as e:
  #Handle connection error here
  print(f"Failed to connect to OpenAI API: {e}")
  pass
except openai.RateLimitError as e:
  #Handle rate limit error (we recommend using exponential backoff)
  print(f"OpenAI API request exceeded rate limit: {e}")
  pass