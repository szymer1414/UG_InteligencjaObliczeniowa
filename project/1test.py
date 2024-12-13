import os
from openai import OpenAI
import openai
# Set the environment variable
os.environ["OPENAI_API_KEY"] = "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"

client = OpenAI()

# Create the assistant
try:
 assistant = client.beta.assistants.create(
    name="CoC-Assistant",
    instructions="An assistant designed to play NPCs using game background knowledge.",
    model="gpt-3.5-turbo",  # Ensure this model is supported in your environment
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
# Create a vector store for storing game background documents
vector_store = client.beta.vector_stores.create(name="CoC-background")

# Prepare the files for upload
file_paths = ["edgar/background.pdf"]
file_streams = [open(path, "rb") for path in file_paths]  # Open files in binary mode

# Upload and poll for file processing completion
file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
    vector_store_id=vector_store.id,
    files=file_streams
)

# Ensure file streams are closed after upload
for stream in file_streams:
    stream.close()

# Update the assistant to link the vector store
assistant = client.beta.assistants.update(
    assistant_id=assistant.id,
    tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
)
