import streamlit as st
import pandas as pd
import joblib
import os

# =====================================================================
# BÍ QUYẾT ĐỂ CHẠY TRÊN MỌI MÁY TÍNH (TỰ ĐỘNG DÒ ĐƯỜNG DẪN)
# =====================================================================
# 1. Lấy đường dẫn tuyệt đối của chính cái thư mục đang chứa file code này
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Tự động nối thêm chữ "Model" để trỏ vào thư mục chứa các file .pkl
MODEL_DIR = os.path.join(CURRENT_DIR, "Model")

# Cấu hình trang Web
st.set_page_config(page_title="Hệ Hỗ Trợ Quyết Định TMĐT", layout="wide")

st.title("Hệ Thống Dự Đoán Ý Định Mua Hàng Thời Gian Thực")
st.markdown("Hệ thống hỗ trợ ra quyết định phân phối khuyến mãi dựa trên hành vi Clickstream.")

# --- KIỂM TRA ĐƯỜNG DẪN NGAY TRÊN GIAO DIỆN (DEBUG) ---
# Đoạn này giúp bạn biết chắc chắn máy tính đang tìm file ở đâu
if not os.path.exists(MODEL_DIR):
    st.error(f"LỖI HỆ THỐNG: Không tìm thấy thư mục chứa mô hình!")
    st.warning(f"Máy tính đang cố tìm thư mục tại đường dẫn này: {MODEL_DIR}")
    st.info("Cách sửa: Hãy đảm bảo bạn đã tạo thư mục tên là 'Model' nằm cùng chỗ với file code này.")
    st.stop() # Dừng phần mềm lại để bạn sửa lỗi

# --- SIDEBAR: CẤU HÌNH MÔ HÌNH ---
st.sidebar.header("Cấu Hình Thuật Toán")

model_choice = st.sidebar.selectbox(
    "Chọn mô hình phân lớp:",
    ("LightGBM", "Random Forest", "Cây Quyết Định", "Hồi quy Logistic")
)

st.sidebar.markdown("---")
st.sidebar.subheader("Ngưỡng Quyết Định (Threshold)")
threshold = st.sidebar.slider(
    "Chọn xác suất tối thiểu để kết luận KH có mua hàng:",
    min_value=0.01, max_value=0.99, value=0.50, step=0.01,
    help="Hạ ngưỡng nếu muốn phủ tệp rộng. Tăng ngưỡng nếu muốn tiết kiệm ngân sách."
)

# --- MAP TÊN MÔ HÌNH VỚI FILE PKL (TỰ ĐỘNG LẤY ĐƯỜNG DẪN) ---
# Tên file phải KHỚP 100% với tên file trong thư mục Model của bạn
model_paths = {
    "LightGBM": os.path.join(MODEL_DIR, "lgb_model.pkl"), 
    "Random Forest": os.path.join(MODEL_DIR, "random_fr.pkl"),
    "Cây Quyết Định": os.path.join(MODEL_DIR, "decision_tree_optimal_model.pkl"),
    "Hồi quy Logistic": os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
}

# --- MAIN AREA: UPLOAD DỮ LIỆU ---
st.subheader("Tải Dữ Liệu Khách Hàng (Input)")
uploaded_file = st.file_uploader("Upload file CSV chứa lịch sử lướt web của khách hàng", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("**Bản xem trước dữ liệu tải lên:**")
    st.dataframe(df.head())

    if st.button("Bắt Đầu Phân Lớp (Predict)"):
        with st.spinner('Hệ thống đang xử lý dữ liệu...'):
            try:
                # 1. Kiểm tra xem file mô hình cụ thể có tồn tại không
                selected_model_path = model_paths[model_choice]
                if not os.path.exists(selected_model_path):
                    st.error(f"LỖI: Không tìm thấy file mô hình của {model_choice}!")
                    st.warning(f"Đang tìm tại: {selected_model_path}")
                    st.stop()

                # 2. Tải mô hình
                model = joblib.load(selected_model_path)
                
                # 3. Tiền xử lý
                cols_to_drop = ['user_session', 'user_id', 'Frequency', 'Monetary', 'Recency', 'avg_price', 'max_price']
                X_features = df.drop(columns=cols_to_drop, errors='ignore')
                
                # Nếu là Logistic Regression -> Cần dùng Scaler
                if model_choice == "Hồi quy Logistic":
                    scaler_path = os.path.join(MODEL_DIR, "standard_scaler.pkl")
                    if not os.path.exists(scaler_path):
                        st.error("LỖI: Không tìm thấy file 'standard_scaler.pkl' cho Hồi quy Logistic!")
                        st.stop()
                        
                    scaler = joblib.load(scaler_path)
                    X_features = scaler.transform(X_features)
                
                # 4. Dự đoán xác suất
                probabilities = model.predict_proba(X_features)[:, 1]
                
                # 5. Áp dụng ngưỡng
                predictions = (probabilities >= threshold).astype(int)
                
                # 6. Hiển thị kết quả
                result_df = df[['user_session', 'user_id']].copy() if 'user_session' in df.columns else df.copy()
                result_df['Xác_Suất_Mua_Hàng'] = probabilities
                result_df['Quyết_Định_Hệ_Thống'] = predictions
                result_df['Nhãn_Văn_Bản'] = result_df['Quyết_Định_Hệ_Thống'].map({1: 'Có mua (Tặng Voucher)', 0: 'Không mua (Bỏ qua)'})
                
                st.success(f"Hoàn thành phân lớp bằng thuật toán {model_choice}!")
                
                col1, col2 = st.columns(2)
                buyers = result_df['Quyết_Định_Hệ_Thống'].sum()
                col1.metric("Tổng số khách hàng", len(result_df))
                col2.metric("Số KH được đề xuất Tặng Voucher", f"{buyers} khách")
                
                st.write("**Bảng Kết Quả Chi Tiết:**")
                
                def color_buyer(val):
                    color = '#d4edda' if val == 1 else ''
                    return f'background-color: {color}'
                
                st.dataframe(result_df.style.map(color_buyer, subset=['Quyết_Định_Hệ_Thống']))
                
                csv_export = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Tải xuống kết quả phân lớp (CSV)",
                    data=csv_export,
                    file_name=f"KetQua_{model_choice.replace(' ', '')}.csv",
                    mime="text/csv",
                )
                
            except Exception as e:
                st.error(f"Có lỗi hệ thống xảy ra: {e}")