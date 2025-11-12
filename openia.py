from openai import OpenAI

# Remplace par ta vraie clé API NVIDIA
api_key = "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"

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# Exemple de génération de texte
completion = client.chat.completions.create(
    model="mistralai/mistral-7b-instruct-v0.2",
    messages=[{"role": "user", "content": "Écris un haïku sur les GPU"}],
    temperature=0.5,
    top_p=1,
    max_tokens=200
)

response_text = completion.choices[0].message.content
print(response_text)
