import requests
import json
from openai import OpenAI
from gpt4all import GPT4All
import subprocess


# Flask应用的URL
url = 'http://121.43.38.159:5005/ask'

# 要发送的问题数据
question_data = {
    'question': 'streamlit如何实现jupyter'
}

# 发送POST请求
response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json.dumps(question_data))

# 检查响应状态码
if response.status_code == 200:
    # 解析响应数据
    answer = response.json().get('answer')
    print(answer)


model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")  # downloads / loads a 4.66GB LLM

with model.chat_session():
    response = model.generate("hello !", max_tokens=1024)
    print(response)


client = OpenAI(
    # 控制台获取key和secret拼接，假使APIKey是key123456，APISecret是secret123456
    api_key="7d3e03a37aeb2cb3056f0bc557881e9a:ODQ2MTE2N2Q4N2U3NDkwNzA1N2ZjNGU1",
    base_url='https://spark-api-open.xf-yun.com/v1'  # 指向讯飞星火的请求地址
)
completion = client.chat.completions.create(
    model='general',
    messages=[
        {
            'role': 'user',
            'content': "streamlit 怎么使用？"
        }
    ]
)
# response = "您输入的信息未找到相关信息，请您重新输入"
response = completion.choices[0].message.content
print(response)



import requests

# 服务器地址和文件名
server_url = 'http://121.43.38.159:5004/download'
filename = 'main.py'  # 替换为你要下载的文件名

# 发送 GET 请求
response = requests.get(server_url, params={'filename': filename})



# 检查请求是否成功
if response.status_code == 200:
    # 将文件内容写入本地文件
    with open(filename, 'wb') as f:

        f.write(response.content)
    print(f"文件 '{filename}' 下载成功！")
else:
    print(f"下载失败，状态码：{response.status_code}")


subprocess.run(['git', 'clone', 'https://github.com/5201314240/one.git'])
