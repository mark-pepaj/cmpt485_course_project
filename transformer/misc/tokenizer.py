from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
text = "Every day is your"
encoded_input = tokenizer.encode(text, return_tensors='pt')
print(f"Encoded input: {encoded_input}")

