from openai import OpenAI

def get_mistral_client(api_key: str):
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )

def generate_answer(client, context: str, question: str):
    completion = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct-v0.2",
        messages=[
            {"role": "system", "content": "Tu es un assistant médical expert."},
            {"role": "user", "content": f"Contexte : {context}\nQuestion : {question}"}
        ],
        temperature=0.5,
        top_p=1,
        max_tokens=300
    )
    return completion.choices[0].message.content
