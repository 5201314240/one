import streamlit as st
import numpy as np
import os
import re
import cv2
from openai import OpenAI
import matplotlib.pyplot as plt
from keras.models import load_model
from gpt4all import GPT4All

from sklearn.preprocessing import MinMaxScaler
import time

scaler = MinMaxScaler()


def upload_image():
    st.markdown('<hr>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Please Upload Your Image:", type=['jpg', 'png', 'tif', 'tiff'])

    st.markdown('<hr>', unsafe_allow_html=True)

    keep_directory = './images/'
    os.makedirs(keep_directory, exist_ok=True)

    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_path = os.path.join(keep_directory, file_name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"图片已成功保存到{file_path}")

        st.image(uploaded_file)


def process_image():
    st.markdown('<hr>', unsafe_allow_html=True)

    def file_selector(folder_path="./images"):
        filenames = os.listdir(folder_path)

        selected_file = st.selectbox("Please select your image", filenames, help='Please select your image')
        if selected_file is not None:
            return os.path.join(folder_path, selected_file)

    filepath = file_selector()
    st.markdown('<hr>', unsafe_allow_html=True)

    if filepath is not None:
        image = cv2.imread(filepath)
        x = st.slider('Change Threshold value', min_value=50, max_value=255)

        col1, col2 = st.columns(2)
        with col1:
            st.image(filepath)
        with col2:
            imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, thresh1 = cv2.threshold(imgray, x, 255, cv2.THRESH_BINARY)
            thresh1 = thresh1.astype(np.float64)
            st.image(thresh1, use_column_width=True, clamp=True)
        st.markdown('<hr>', unsafe_allow_html=True)

        st.sidebar.text("Press the button below to view Bar Chart of the image")
        button_style = """  
        <style>  
        .stButton > button {  
            width: 280px;  /* 设置宽度 */  
            height: 50px;  /* 设置高度 */  
            font-size: 20px; /* 设置字体大小 */  
        }  
        </style>  
        """
        st.sidebar.markdown(button_style, unsafe_allow_html=True)

        if st.sidebar.button("Bar Chart"):
            histr = cv2.calcHist([imgray], [0], None, [256], [0, 256])
            st.bar_chart(histr)

        st.sidebar.text("Press the button below to view Canny Edge Detection Technique")
        if st.sidebar.button('Canny Edge Detection'):
            edges = cv2.Canny(imgray, 50, 300)
            # cv2.imwrite('edges.jpg',edges)
            st.image(edges, use_column_width=True, clamp=True)

        y = st.sidebar.slider("Change Value to increase or decrease contours", min_value=50, max_value=255)
        if st.sidebar.button('Contours'):
            image = cv2.imread(filepath)
            imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, y, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            img = cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
            st.image(thresh, use_column_width=True, clamp=True)
            st.image(img, use_column_width=True, clamp=True)
        st.sidebar.markdown('<hr>', unsafe_allow_html=True)


def model_predict():
    global pred

    def file_selector(folder_path="./images"):
        filenames = os.listdir(folder_path)

        selected_file = st.selectbox("Please select your image", filenames)
        if selected_file is not None:
            return os.path.join(folder_path, selected_file)

    filepath = file_selector()
    selected_model = st.selectbox("Please select your model", ['history1', 'history2'])

    if filepath is not None:
        col1, col2 = st.columns(2)
        # 在第一列中显示第一个图像
        with col1:
            image = cv2.imread(filepath)
            image = cv2.resize(image, (256, 256))
            plt.imshow(image, cmap='viridis')
            plt.axis('off')  # 不显示坐标轴
            st.pyplot(plt)
            target_size = (256, 256)

            # 在第二列中显示第二个图像
        with col2:
            images = unit_image_process(image)
            if selected_model == 'history1':
                model = load_model("models/satellite_standard_unet_100epochs_7May2021.hdf5", compile=False)
                pred = model.predict(images)
                pred = np.argmax(pred, axis=3)
                pred = pred[0, :, :]
                pred = label_to_rgb(pred)
                pred = cv2.resize(pred, target_size)
            elif selected_model == 'history2':
                model = load_model("models/history2.hdf5", compile=False)
                pred = model.predict(images)
                pred = np.argmax(pred, axis=3)
                pred = pred[0, :, :]
                pred = label_to_rgb(pred)
                pred = cv2.resize(pred, target_size)
            plt.imshow(pred, cmap='viridis')
            plt.axis('off')  # 不显示坐标轴
            st.pyplot(plt)

        st.markdown("#### 两种模型的损失函数对比")
        # 使用 Streamlit 创建列布局
        col1, col2 = st.columns(2)

        # 在第一列中显示第一个图像
        with col1:
            plt.imshow(cv2.imread('data/history1_loss.jpg'), cmap='viridis')
            plt.axis('off')  # 不显示坐标轴
            st.pyplot(plt)

            # 在第二列中显示第二个图像
        with col2:
            plt.imshow(cv2.imread('data/history2_loss.jpg'), cmap='viridis')
            plt.axis('off')  # 不显示坐标轴
            st.pyplot(plt)

        st.markdown("#### 两种模型的交并比对比")
        col1, col2 = st.columns(2)

        # 在第一列中显示第一个图像
        with col1:
            plt.imshow(cv2.imread('data/history1_IoU.jpg'), cmap='viridis')
            plt.axis('off')  # 不显示坐标轴
            st.pyplot(plt)

            # 在第二列中显示第二个图像
        with col2:
            plt.imshow(cv2.imread('data/history2_IoU.jpg'), cmap='viridis')
            plt.axis('off')  # 不显示坐标轴
            st.pyplot(plt)


def chatbot():
    # model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")  # downloads / loads a 4.66GB LLM

    def word_split(sentence):
        words = re.split(r"\W+", sentence)
        filtered_words = [word for word in words]
        return filtered_words

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": """
            Hello! I'm your image segmentation assistant,\n
            ready to help you identify and classify every detail in your image!"""}]

    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'], unsafe_allow_html=True)

    if prompt := st.chat_input("Please enter your question?"):
        with st.chat_message("user"):
            st.markdown(prompt, unsafe_allow_html=True)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

    if prompt:
        def generate_stream(data):
            for char in data:
                yield char
                time.sleep(0.1)

        my_dict = {'你好': '你好', '多少': '本次调用模型得到的平均损失值为：0.4284，平均IOU值为：0.5972',
                   '类别': '本模型的图像分割类别共有六种。'}
        response = ""
        word_list = word_split(prompt)
        for word in word_list:
            if word in my_dict:
                response += my_dict[word]
        if len(response) == 0:
            # with model.chat_session():
            #     response = model.generate(prompt, max_tokens=1024)
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
                        'content': prompt
                    }
                ]
            )
            # response = "您输入的信息未找到相关信息，请您重新输入"
            response = completion.choices[0].message.content
        with st.chat_message("assistant"):
            content = st.write_stream(generate_stream(response))
        st.session_state.messages.append({'role': 'assistant', 'content': content})


def label_to_rgb(predicted_image):
    Building = '#3C1098'.lstrip('#')
    Building = np.array(tuple(int(Building[i:i + 2], 16) for i in (0, 2, 4)))  # 60, 16, 152

    Land = '#8429F6'.lstrip('#')
    Land = np.array(tuple(int(Land[i:i + 2], 16) for i in (0, 2, 4)))  # 132, 41, 246

    Road = '#6EC1E4'.lstrip('#')
    Road = np.array(tuple(int(Road[i:i + 2], 16) for i in (0, 2, 4)))  # 110, 193, 228

    Vegetation = 'FEDD3A'.lstrip('#')
    Vegetation = np.array(tuple(int(Vegetation[i:i + 2], 16) for i in (0, 2, 4)))  # 254, 221, 58

    Water = 'E2A929'.lstrip('#')
    Water = np.array(tuple(int(Water[i:i + 2], 16) for i in (0, 2, 4)))  # 226, 169, 41

    Unlabeled = '#9B9B9B'.lstrip('#')
    Unlabeled = np.array(tuple(int(Unlabeled[i:i + 2], 16) for i in (0, 2, 4)))  # 155, 155, 155

    segmented_img = np.empty((predicted_image.shape[0], predicted_image.shape[1], 3))

    segmented_img[(predicted_image == 0)] = Building
    segmented_img[(predicted_image == 1)] = Land
    segmented_img[(predicted_image == 2)] = Road
    segmented_img[(predicted_image == 3)] = Vegetation
    segmented_img[(predicted_image == 4)] = Water
    segmented_img[(predicted_image == 5)] = Unlabeled

    segmented_img = segmented_img.astype(np.uint8)
    return (segmented_img)


def unit_image_process(images):
    patch_size = 256
    image = cv2.resize(images, (patch_size, patch_size))
    single_patch_img = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
    single_patch_img = np.expand_dims(single_patch_img, axis=0)
    return single_patch_img


if __name__ == '__main__':

    st.sidebar.title("Image  classification")
    st.write("# Machine Learning and Big Data 👋")

    st.sidebar.markdown('<hr>', unsafe_allow_html=True)

    st.sidebar.header("Please choose an options")

    page_options = ['Image upload', 'Image Processing', 'Model predictions', 'Chatbots']
    selected_page = st.sidebar.selectbox("", page_options)
    st.sidebar.markdown('<hr>', unsafe_allow_html=True)

    if selected_page == 'Image upload':
        upload_image()
    elif selected_page == 'Image Processing':
        process_image()
    elif selected_page == 'Model predictions':
        model_predict()
    elif selected_page == 'Chatbots':
        chatbot()
