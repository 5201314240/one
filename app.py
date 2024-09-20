from gpt4all import GPT4All

model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")  # downloads / loads a 4.66GB LLM

with model.chat_session():
    response = model.generate("hello !", max_tokens=1024)