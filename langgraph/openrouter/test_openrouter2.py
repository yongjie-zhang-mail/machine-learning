"""Simple arithmetic test against OpenRouter API.

Removed previously hard-coded API key. Use an environment variable instead:

  export OPENROUTER_API_KEY=sk-or-v1-xxx
  python test_openrouter2.py
"""

import os
import sys
from openai import OpenAI

api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("[ERROR] Please set OPENROUTER_API_KEY (or OPENAI_API_KEY) in your environment.")
    sys.exit(1)

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key,
)

completion = client.chat.completions.create(
#   extra_headers={
#     "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
#     "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
#   },
  extra_body={},
  model="x-ai/grok-4-fast:free",
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "3+2=?"
        }
      ]
    }
  ]
)
print(completion.choices[0].message.content)