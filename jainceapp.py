import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # 新增：用于加载字体
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# --- 页面配置 ---
st.set_page_config(page_title="智能抓取数据分析台", layout="wide")
# ========== 核心修改：云端中文完美解决方案 ==========
# 假设您已经把 SimHei.ttf 文件上传到了项目根目录
font_path = 'SimHei.ttf' 

# 尝试加载字体，如果文件不存在则回退
try:
    font_prop = fm.FontProperties(fname=font_path)
    # 将该字体设置为全局默认 sans-serif 字体
    plt.rcParams['font.family'] = font_prop.get_name()
    # 这一步是为了让 matplotlib 的 font manager 注册该字体
    fm.fontManager.addfont(font_path) 
    print("已加载本地字体")
except Exception as e:
    st.warning(f"⚠️ 未找到字体文件 {font_path}，中文可能显示为方块。请确保将 .ttf 文件上传到 GitHub。")
    # 回退设置
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] 

plt.rcParams['axes.unicode_minus'] = False
# =================================================

st.title("🤖 机器人触觉数据可视化分析系统")
st.title("🤖 机器人触觉数据可视化分析系统")
st.markdown("### 只要上传 CSV 文件，立马告诉您哪些传感器最关键！")

# --- 侧边栏：文件上传 ---
st.sidebar.header("1. 数据上传")
uploaded_file = st.sidebar.file_uploader("请上传您的 CSV 数据集", type=["csv"])

# --- 主逻辑 ---
if uploaded_file is not None:
    # 1. 读取数据
    df = pd.read_csv(uploaded_file)
    
    st.sidebar.success("文件上传成功！")
    st.sidebar.markdown(f"**数据行数**: {df.shape[0]}")
    st.sidebar.markdown(f"**特征数量**: {df.shape[1]}")

    # 分割线
    st.divider()

    # --- 第一部分：数据概览 ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 滑落分布 (Target Distribution)")
        if 'slipped' in df.columns:
            # 画饼图
            fig_pie, ax_pie = plt.subplots()
            df['slipped'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax_pie, startangle=90, colors=['#66b3ff','#ff9999'])
            ax_pie.set_ylabel('')
            st.pyplot(fig_pie)
        else:
            st.error("CSV中未找到 'slipped' 列，无法分析滑落情况。")

    with col2:
        st.subheader("📋 数据预览")
        st.dataframe(df.head(8), height=300)

    # --- 第二部分：智能特征分析 ---
    st.divider()
    st.header("🧠 AI 核心分析：哪些传感器最重要？")
    
    if st.button("开始 AI 分析 (点击运行随机森林)"):
        with st.spinner('正在训练模型并筛选特征...'):
            # 简单的数据预处理
            target = 'slipped'
            ignore_cols = ['object', target]
            # 筛选出数值型特征
            feature_cols = [c for c in df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])]
            
            X = df[feature_cols]
            y = df[target]
            
            # 标准化 (为了可视化的一致性)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 训练模型获取重要性
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(X_scaled, y)
            
            # 提取重要性
            importances = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': rf.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            # 1. 展示特征重要性柱状图
            st.subheader("🏆 特征重要性排名 (Top 10)")
            fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
            sns.barplot(x='Importance', y='Feature', data=importances.head(10), ax=ax_bar, palette="viridis")
            st.pyplot(fig_bar)
            
            # 2. 关键特征深入透视
            st.subheader("🔍 关键特征透视 (Top 3 传感器分析)")
            st.markdown("观察这些传感器在 **未滑落(0)** vs **滑落(1)** 时的数值差异：")
            
            top_3_features = importances['Feature'].head(3).tolist()
            
            # 并排画3个箱线图
            cols = st.columns(3)
            for i, feature in enumerate(top_3_features):
                with cols[i]:
                    fig_box, ax_box = plt.subplots()
                    sns.boxplot(x='slipped', y=feature, data=df, ax=ax_box, palette="Set2")
                    ax_box.set_title(f"{feature}")
                    st.pyplot(fig_box)
            
            st.success("分析完成！建议针对上述 Top 3 传感器优化抓取策略。")

else:
    st.info("👈 请在左侧侧边栏上传 CSV 文件以开始分析。")