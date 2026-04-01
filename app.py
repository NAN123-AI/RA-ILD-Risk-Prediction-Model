# app.py
import streamlit as st
import numpy as np

# ===== 页面配置 =====
st.set_page_config(page_title="RA-ILD风险预测模型", layout="wide")

# ===== 背景 + CSS =====
st.markdown("""
<style>
/* 页面背景（网络肺图） */
.stApp {
    background-image: url("https://upload.wikimedia.org/wikipedia/commons/8/88/Lungs_anterior.png");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* 半透明浮动卡片 */
.stContainer {
    background: rgba(255, 255, 255, 0.85);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    margin: 20px;
}

/* 标题、文本 */
h1, h2, h3, h4, h5, h6, .stMarkdown {
    color: #1a1a1a;
    font-weight: bold;
}

/* 滑块文字 */
div.stSlider, div.stRadio {
    color: #1a1a1a;
    font-weight: bold;
}

/* Metric显示 */
div[data-testid="metric-container"] {
    background: rgba(240, 240, 240, 0.9);
    padding: 15px;
    border-radius: 15px;
    width: 220px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ===== 页面标题 =====
st.title("RA-ILD风险预测模型（论文一致版）")
st.markdown("基于多因素Logistic回归模型（AUC=0.959）")

# ===== 输入区 =====
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("年龄", 30, 90, 60)
        smoke = st.radio("吸烟史", ["否", "是"])
        smoke_val = 1 if smoke == "是" else 0
        il22 = st.slider("IL-22 (pg/ml)", 100, 350, 220)

    with col2:
        mcvab = st.slider("MCV-Ab", 0, 1000, 500)
        mchc = st.slider("MCHC", 260, 350, 320)

# ===== 中心化 / 标准化 =====
age_c = (age - 60)/10
il22_c = (il22 - 220)/50
mcvab_c = (mcvab - 500)/100
mchc_c = (mchc - 320)/10

# ===== Logistic回归计算 =====
z = (
    -0.032
    -0.059 * il22_c
    +0.110 * age_c
    +4.288 * smoke_val
    +0.006 * mcvab_c
    -0.124 * mchc_c
)
risk = 1 / (1 + np.exp(-z))

# ===== 输出预测结果 =====
with st.container():
    st.subheader("📊 预测结果")
    st.metric("RA-ILD风险概率", f"{risk:.2%}")

    if risk < 0.2:
        st.success("低风险")
    elif risk < 0.5:
        st.warning("中等风险")
    else:
        st.error("高风险")

    st.progress(float(risk))

# ===== IL-22提示 =====
with st.container():
    st.subheader("🧬 IL-22临床提示")
    if il22 < 243.06:
        st.error("IL-22 < 243 → 高风险提示")
    else:
        st.success("IL-22 ≥ 243 → 相对低风险")

# ===== 模型说明 =====
with st.container():
    st.markdown("""
---
### 📚 模型说明
- 多因素Logistic回归模型
- AUC = 0.959
- 敏感度 = 97.0%
- 特异度 = 87.9%
""")
