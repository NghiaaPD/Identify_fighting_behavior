# Sử dụng image python chính thức
FROM python:3.10-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Sao chép file requirements.txt vào container
COPY requirements.txt ./ 

# Cài đặt các thư viện yêu cầu trong requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Cài đặt PyTorch với CUDA 12.4
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Sao chép toàn bộ mã nguồn vào thư mục /app
COPY . .

# Thiết lập cổng cho ứng dụng
EXPOSE 8501

# Chạy ứng dụng Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
