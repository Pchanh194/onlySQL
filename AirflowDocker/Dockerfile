# Sử dụng Apache Airflow image
FROM apache/airflow:2.2.3

# Cài đặt các gói phụ thuộc cần thiết (nếu có)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép các tệp và thư mục liên quan đến dags vào image
COPY dags /opt/airflow/dags
