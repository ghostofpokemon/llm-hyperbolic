import os
from llm_hyperbolic import HyperbolicChat

def chat_with_hyperbolic():
    system_content = "You are a gourmet. Be descriptive and helpful."
    user_content = "Tell me about Chinese hotpot"

    hyperbolic_chat = HyperbolicChat(
        model_id="hyperbolicchat/meta-llama/Meta-Llama-3.1-70B-Instruct",
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
        api_base="https://api.hyperbolic.xyz/v1",
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    response = hyperbolic_chat.execute(
        messages,
        temperature=0.7,
        max_tokens=1024,
    )

    print("Response:\n", next(response))

if __name__ == "__main__":
    chat_with_hyperbolic()
