from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile

# 标签映射
LABEL_MAPPING = {
    "b_fully_ripened": "成熟大番茄",
    "b_half_ripened": "半成熟大番茄",
    "b_green": "青色大番茄",
    "l_fully_ripened": "成熟小番茄",
    "l_half_ripened": "半成熟小番茄",
    "l_green": "青色小番茄",
    "fully_ripened": "完全成熟草莓",
    "half_ripened": "半成熟草莓",
    "unripened": "不成熟草莓",
    "fresh_apple": "新鲜苹果",
    "normal_apple": "一般品质苹果",
    "rotten_apple": "腐烂苹果",
    "fresh_banana": "新鲜香蕉",
    "normal_banana": "一般品质香蕉",
    "rotten_banana": "腐烂香蕉",
    "fresh_orange": "新鲜橘子",
    "normal_orange": "一般品质橘子",
    "rotten_orange": "腐烂橘子"
}

def _display_detected_frames(conf, model, st_frame, image):
    image = cv2.resize(image, (720, int(720 * (9 / 16))))
    res = model.predict(image, conf=conf)
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model


def infer_uploaded_image(conf, model, image_file):
    if image_file is None:
        st.error("没有上传图片，请上传一张图片进行检测")
        return

    uploaded_image = Image.open(image_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image=uploaded_image,
            caption="您上传的图片",
            use_column_width=True
        )

    if st.button("点击进行预测"):
        with st.spinner("请等待..."):
            res = model.predict(uploaded_image, conf=conf)
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]

            with col2:
                st.image(res_plotted,
                         caption="预测结果图片",
                         use_column_width=True)
                try:
                    with st.expander("检测结果查看"):
                        names = res[0].names  # 获取预测结果的类名
                        counts = [0] * len(names)
                        for box in boxes:
                            class_id = int(box.cls[0])
                            counts[class_id] += 1
                        result = '\n'.join([f"{LABEL_MAPPING.get(names[i], names[i])}: {count}" for i, count in enumerate(counts) if count > 0])
                        st.write(result)
                except Exception as ex:
                    st.write("检测结果显示错误！")
                    st.write(ex)


def infer_uploaded_video(conf, model, video_file):

    if video_file is None:
        st.error("没有上传视频，请上传一个视频进行检测")
        return

    st.video(video_file)

    if st.button("点击进行预测"):
        with st.spinner("请等待..."):
            try:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(video_file.read())
                vid_cap = cv2.VideoCapture(tfile.name)
                st_frame = st.empty()
                while vid_cap.isOpened():
                    success, image = vid_cap.read()
                    if success:
                        _display_detected_frames(conf, model, st_frame, image)
                    else:
                        vid_cap.release()
                        break
            except Exception as e:
                st.error(f"视频加载错误: {e}")


def infer_uploaded_webcam(conf, model):

    try:
        flag = st.button(label="停止运行")
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(conf, model, st_frame, image)
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"摄像头加载错误: {str(e)}")
