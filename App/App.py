import streamlit as st
import pandas as pd
import joblib
import io

# Cấu hình trang Web
st.set_page_config(page_title="Hệ Hỗ Trợ Quyết Định TMĐT", layout="wide")

st.title("Hệ Thống Dự Đoán Ý Định Mua Hàng Thời Gian Thực")
st.markdown("Hệ thống hỗ trợ ra quyết định phân phối khuyến mãi dựa trên hành vi Clickstream.")

# --- SIDEBAR: CẤU HÌNH MÔ HÌNH ---
st.sidebar.header("Cấu Hình Thuật Toán")

# 1. Chọn mô hình
model_choice = st.sidebar.selectbox(
    "Chọn mô hình phân lớp:",
    ("LightGBM", "Random Forest", "Cây Quyết Định", "Hồi quy Logistic")
)

# 2. Chọn ngưỡng quyết định
st.sidebar.markdown("---")
st.sidebar.subheader("Ngưỡng Quyết Định (Threshold)")
threshold = st.sidebar.slider(
    "Chọn xác suất tối thiểu để kết luận KH có mua hàng:",
    min_value=0.01, max_value=0.99, value=0.50, step=0.01,
    help="Hạ ngưỡng nếu muốn phủ tệp rộng (Tăng Recall). Tăng ngưỡng nếu muốn tiết kiệm ngân sách (Tăng Precision)."
)

# --- MAP TÊN MÔ HÌNH VỚI FILE PKL ---
model_paths = {
    "LightGBM": "Model/lgb_model.pkl", 
    "Random Forest": "Model/random_fr.pkl",
    "Cây Quyết Định": "Model/decision_tree_optimal_model.pkl",
    "Hồi quy Logistic": "Model/logistic_regression_model.pkl"
}

# --- MAIN AREA: UPLOAD DỮ LIỆU ---
st.subheader("Tải Dữ Liệu Khách Hàng (Input)")
uploaded_file = st.file_uploader("Upload file CSV chứa lịch sử lướt web của khách hàng", type=["csv"])

if uploaded_file is not None:
    # Đọc dữ liệu
    df = pd.read_csv(uploaded_file)
    st.write("**Bản xem trước dữ liệu tải lên:**")
    st.dataframe(df.head())

    # Nút bấm dự đoán
    if st.button("Bắt Đầu Phân Lớp (Predict)"):
        with st.spinner('Hệ thống đang xử lý dữ liệu...'):
            try:
                # 1. Tải mô hình
                model = joblib.load(model_paths[model_choice])
                
                # 2. Tiền xử lý: Bỏ các cột ID và các cột gây đa cộng tuyến
                cols_to_drop = ['user_session', 'user_id', 'Frequency', 'Monetary', 'Recency', 'avg_price', 'max_price']
                X_features = df.drop(columns=cols_to_drop, errors='ignore')
                
                # Nếu là Logistic Regression -> Cần dùng Scaler
                if model_choice == "Hồi quy Logistic":
                    scaler = joblib.load("Model/standard_scaler.pkl")
                    X_features = scaler.transform(X_features)
                
                # 3. Dự đoán xác suất
                probabilities = model.predict_proba(X_features)[:, 1]
                
                # 4. Áp dụng ngưỡng (Threshold)
                predictions = (probabilities >= threshold).astype(int)
                
                # 5. Gắn kết quả trả lại Dataframe gốc để hiển thị
                result_df = df[['user_session', 'user_id']].copy() if 'user_session' in df.columns else df.copy()
                result_df['Xác_Suất_Mua_Hàng'] = probabilities
                result_df['Quyết_Định_Hệ_Thống'] = predictions
                result_df['Nhãn_Văn_Bản'] = result_df['Quyết_Định_Hệ_Thống'].map({1: 'Có mua (Tặng Voucher)', 0: 'Không mua (Bỏ qua)'})
                
                # --- HIỂN THỊ KẾT QUẢ ---
                st.success(f"Hoàn thành phân lớp bằng thuật toán {model_choice}!")
                
                # Hiển thị thống kê
                col1, col2 = st.columns(2)
                buyers = result_df['Quyết_Định_Hệ_Thống'].sum()
                col1.metric("Tổng số khách hàng", len(result_df))
                col2.metric("Số KH được đề xuất Tặng Voucher", f"{buyers} khách")
                
                st.write("**Bảng Kết Quả Chi Tiết:**")
                # Highlight màu cho người mua hàng
                def color_buyer(val):
                    color = '#d4edda' if val == 1 else ''
                    return f'background-color: {color}'
                
                st.dataframe(result_df.style.applymap(color_buyer, subset=['Quyết_Định_Hệ_Thống']))
                
                # --- CHỨC NĂNG XUẤT FILE TẢI VỀ ---
                csv_export = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Tải xuống kết quả phân lớp (CSV)",
                    data=csv_export,
                    file_name=f"KetQua_{model_choice.replace(' ', '')}.csv",
                    mime="text/csv",
                )
                
            except Exception as e:
                st.error(f"Có lỗi xảy ra trong quá trình xử lý: {e}")