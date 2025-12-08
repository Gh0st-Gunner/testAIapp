import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Khảo sát Outlier & Hồi quy tuyến tính", layout="wide")

st.title("📌 Khảo sát tác động của điểm ngoại lai lên nghiệm Hồi quy tuyến tính")
st.write("### Góc nhìn: Đại số tuyến tính – nghiệm β = (XᵀX)⁻¹Xᵀy")

# ---------------------------------------------------------
# 1. Load CSV
# ---------------------------------------------------------

uploaded = st.file_uploader("📂 Tải file CSV dữ liệu", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("### 📄 Dữ liệu ban đầu")
    st.dataframe(df)

    # Tách X và y
    y = df.iloc[:, 0].values.reshape(-1, 1)
    X = df.iloc[:, 1:].values

    # Thêm cột 1 cho hệ số intercept
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    # ---------------------------------------------------------
    # 2. Hàm tính nghiệm tuyến tính bằng công thức đại số
    # ---------------------------------------------------------
    def linear_solution(X, y):
        XtX = X.T @ X
        Xty = X.T @ y
        beta = np.linalg.inv(XtX) @ Xty
        return beta, XtX, Xty

    # ---------------------------------------------------------
    # 3. Nghiệm với toàn bộ dữ liệu (có outlier)
    # ---------------------------------------------------------
    st.subheader("📌 Nghiệm hồi quy với *dữ liệu đầy đủ* (có OUTLIER)")
    beta_full, XtX_full, Xty_full = linear_solution(X, y)

    st.write("**Ma trận XᵀX:**")
    st.dataframe(pd.DataFrame(XtX_full))

    st.write("**Vector Xᵀy:**")
    st.dataframe(pd.DataFrame(Xty_full))

    st.write("**Nghiệm β (có outlier):**")
    st.dataframe(pd.DataFrame(beta_full, columns=["β"]))

    # ---------------------------------------------------------
    # 4. Loại dòng cuối cùng – giả lập thao tác thủ công
    # ---------------------------------------------------------
    st.subheader("📌 Nghiệm hồi quy *không có outlier* (loại dòng cuối)")

    df_no = df.iloc[:-1, :]
    y_no = df_no.iloc[:, 0].values.reshape(-1, 1)
    X_no = df_no.iloc[:, 1:].values
    X_no = np.hstack([np.ones((X_no.shape[0], 1)), X_no])

    beta_no, XtX_no, Xty_no = linear_solution(X_no, y_no)

    st.write("**Nghiệm β (không có outlier):**")
    st.dataframe(pd.DataFrame(beta_no, columns=["β"]))

    # ---------------------------------------------------------
    # 5. So sánh trực quan
    # ---------------------------------------------------------
    st.subheader("📊 So sánh tác động của OUTLIER lên nghiệm β")

    comparison = pd.DataFrame({
        "β_full (có outlier)": beta_full.flatten(),
        "β_no_outlier": beta_no.flatten()
    })

    st.dataframe(comparison.style.highlight_max(axis=0, color="red"))

    st.write("""
    ### 🧠 Nhận xét:
    - Outlier làm thay đổi **ma trận XᵀX**, khiến nghiệm β bị kéo lệch mạnh.
    - Hệ số tương ứng với các cột có giá trị outlier lớn sẽ thay đổi nhiều nhất.
    - Đây chính là tác động trực tiếp trong không gian vector: *một điểm cực lớn làm thay đổi hướng của hyperplane tối ưu.*
    """)

else:
    st.info("👉 Hãy tải file CSV vào để bắt đầu phân tích.")
