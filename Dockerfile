# Sử dụng image python chính thức
FROM python:3.9-slim

# Cài đặt các dependencies cần thiết cho OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

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

# Chạy ứng dụng Streamlit với địa chỉ IP 192.168.x.x
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=192.168.x.x"]
