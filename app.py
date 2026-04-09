import streamlit as st
import base64
from pathlib import Path
from PIL import Image
import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam

# 设置页面配置
st.set_page_config(
    page_title="多类水果成熟度检测",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🍎"
)

# 背景图片设置函数
def main_bg(main_bg_filename):
    with open(main_bg_filename, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode()
    st.markdown(
        f'''
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_image});
            background-size: cover; /* 确保背景图片覆盖整个背景区域 */
            background-attachment: scroll; /* 背景图片随页面内容滚动 */
            background-position: center; /* 背景图片居中 */
        }}
        </style>
        ''',
        unsafe_allow_html=True
    )

# 调用背景设置函数
main_bg('0001.jpg')

# 页面标题
st.title("多类水果成熟度检测网站")

# 添加标题下方的右对齐链接，字体颜色设置为深黑色
st.markdown("""
    <div style="text-align: right;">
        <a href="http://sf1970.cnif.cn/article/2023/0253-990X/2023-17-354.shtml" target="_blank" style="color: #000000;">相关研究</a> | 
        <a href="https://www.smartag.net.cn/article/2021/2096-8094/2096-8094-2021-3-4-14.shtml" target="_blank" style="color: #000000;">研究进展</a> | 
        <a href="https://zqb.cyol.com/html/2023-03/24/nw.D110000zgqnb_20230324_5-01.htm" target="_blank" style="color: #000000;">研究意义</a> 
    </div>
    """, unsafe_allow_html=True)

# 添加自定义 CSS 使组件居中并调整间距
st.markdown("""
    <style>
    .centered-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .custom-label {
        font-weight: bold;
        margin-bottom: 5px; /* 调整标签和框之间的距离 */
    }
    .custom-selectbox, .custom-file-uploader, .custom-button {
        margin-bottom: 15px; /* 调整组件与下一个组件的距离 */
    }
    .transparent-background {
        background-color: rgba(255, 255, 255, 0) !important; /* 设置背景色为透明 */
    }
    /* 修改上传文件区域的提示文本 */
    .upload-button span[data-testid="stFileUploader"] {
        display: none;
    }
    .upload-button::before {
        content: "拖放文件到这里或点击选择文件";
        display: block;
        text-align: center;
        padding: 20px;
        border: 2px dashed #ccc;
        border-radius: 4px;
        font-size: 16px;
        color: #555;
        background-color: #f9f9f9;
    }
    </style>
    """, unsafe_allow_html=True)

# 创建第一行布局：任务类别选择、模型选择、置信度滑块、数据源选择
st.markdown('<div class="transparent-background">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    st.markdown('<div class="custom-label">任务类别选择</div>', unsafe_allow_html=True)
    task_type = st.selectbox(
        "", ["检测", "分割"], key="task_type", help="选择任务类别"
    )

with col2:
    st.markdown('<div class="custom-label">选择模型</div>', unsafe_allow_html=True)
    model_type = st.selectbox(
        "",
        config.DETECTION_MODEL_LIST if task_type == "检测" else [],
        key="model_type", help="选择检测模型"
    )

with col3:
    st.markdown('<div class="custom-label">置信度</div>', unsafe_allow_html=True)
    confidence = st.slider(" ", 30, 100, 50, key="confidence") / 100

with col4:
    st.markdown('<div class="custom-label">选择数据源</div>', unsafe_allow_html=True)
    source_type = st.selectbox(
        "", ["图片", "视频", "实时检测"], key="source_type", help="选择数据源"
    )

st.markdown('</div>', unsafe_allow_html=True)

# 创建第二行布局：图片和视频上传，以及摄像头启动按钮
st.markdown('<div class="centered-container">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # 加载模型
    model_path = Path(config.DETECTION_MODEL_DIR, model_type) if model_type else ""
   import os
# 替换原来的那一行
if model_path and os.path.exists(model_path):
    try:
        model = load_model(model_path)
    except Exception:
        st.warning("模型加载失败，已切换至基础模式")
        model = None
else:
    model = None

    if not model:
        st.warning("请先选择模型")

    if source_type == "图片":
        st.markdown('<div class="custom-label">选择一张图片</div>', unsafe_allow_html=True)
        uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png", "bmp", "webp"], key="uploaded_image")
        if uploaded_image and model:
            infer_uploaded_image(confidence, model, uploaded_image)
        elif not uploaded_image:
            st.info("请上传一张图片进行检测")

    elif source_type == "视频":
        st.markdown('<div class="custom-label">选择一个视频</div>', unsafe_allow_html=True)
        uploaded_video = st.file_uploader("", type=["mp4", "mov", "avi"], key="uploaded_video")
        if uploaded_video and model:
            infer_uploaded_video(confidence, model, uploaded_video)
        elif not uploaded_video:
            st.info("请上传一个视频进行检测")

    elif source_type == "实时检测":
        st.markdown('<div class="custom-button">', unsafe_allow_html=True)
        if st.button("启动摄像头", key="start_webcam"):
            if model:
                infer_uploaded_webcam(confidence, model)
            else:
                st.warning("请先选择模型")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
