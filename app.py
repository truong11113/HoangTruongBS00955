# 📦 Khai báo thư viện
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# 🎯 Cấu hình giao diện
st.set_page_config(page_title="Phân tích & Dự đoán Doanh Thu", layout="wide")
st.title("📊 Phân tích & Dự đoán Doanh Thu từ dữ liệu bán hàng")

# 📁 Upload file CSV
uploaded_file = st.file_uploader("Tải lên file Sales_Data_P7.csv", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # 🔧 Tiền xử lý
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Customer_Age'].fillna(df['Customer_Age'].mean(), inplace=True)
    df['Revenue'] = df['Quantity'] * df['Unit_Price'] * (1 - df['Discount'])
    df['Category'] = df['Category'].str.strip().str.lower()
    df['Region'] = df['Region'].str.strip().str.lower()

    st.subheader("📋 Dữ liệu mẫu")
    st.dataframe(df.head())

    # 📊 Phân tích dữ liệu
    st.subheader("📈 Phân tích dữ liệu")
    st.write("✅ Doanh thu theo vùng:")
    st.dataframe(df.groupby('Region')['Revenue'].sum().sort_values(ascending=False))

    st.write("✅ Doanh thu theo danh mục:")
    st.dataframe(df.groupby('Category')['Revenue'].sum().sort_values(ascending=False))

    st.write("✅ Doanh thu theo tháng:")
    st.dataframe(df.groupby('Month')['Revenue'].sum().sort_values(ascending=False))

    st.write("✅ Thống kê độ tuổi khách hàng:")
    st.dataframe(df['Customer_Age'].describe())

    # 📊 Trực quan hóa
    st.subheader("📊 Trực quan hóa dữ liệu")

    fig1, ax1 = plt.subplots()
    df.groupby('Region')['Revenue'].sum().plot(kind='bar', color='skyblue', ax=ax1)
    ax1.set_title("Doanh thu theo vùng")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    df.groupby('Category')['Revenue'].sum().plot(kind='pie', autopct='%1.1f%%', ax=ax2)
    ax2.set_ylabel('')
    ax2.set_title("Tỷ lệ doanh thu theo danh mục")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    monthly = df.groupby('Month')['Revenue'].sum()
    sns.lineplot(x=monthly.index, y=monthly.values, marker='o', color='orange', ax=ax3)
    ax3.set_title("Doanh thu theo tháng")
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots()
    sns.scatterplot(data=df, x='Discount', y='Revenue', hue='Category', ax=ax4)
    ax4.set_title("Giảm giá vs Doanh thu")
    st.pyplot(fig4)

    fig5, ax5 = plt.subplots()
    sns.boxplot(x=df['Customer_Age'], color='lightgreen', ax=ax5)
    ax5.set_title("Phân bố độ tuổi khách hàng")
    st.pyplot(fig5)

    # 🤖 Huấn luyện mô hình
    st.subheader("🤖 Huấn luyện mô hình Linear Regression")

    X = df[['Quantity', 'Unit_Price', 'Discount', 'Customer_Age']]
    y = df['Revenue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.write(f"📉 MSE: {mse:.2f}")
    st.write(f"📈 RMSE: {rmse:.2f}")
    st.write(f"✅ R-squared (R2): {r2:.2f}")

    # 🔍 Phân tích sai số
    st.subheader("🔍 Phân tích sai số dự đoán")
    residuals = y_test - y_pred

    fig6, ax6 = plt.subplots()
    sns.histplot(residuals, kde=True, color='skyblue', ax=ax6)
    ax6.set_title("Phân phối sai số dự đoán")
    st.pyplot(fig6)

    fig7, ax7 = plt.subplots()
    sns.scatterplot(x=y_pred, y=residuals, color='plum', ax=ax7)
    ax7.axhline(0, color='gray', linestyle='--')
    ax7.set_title("Sai số vs Giá trị dự đoán")
    st.pyplot(fig7)

    # 🔁 Cross Validation
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    st.write(f"📊 R2 trung bình qua 5 lần kiểm tra: {scores.mean():.2f}")
