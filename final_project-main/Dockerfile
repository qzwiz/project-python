# 1. Базовий образ із Python
FROM python:3.11-slim

# 2. Встановлення системних залежностей
RUN apt-get update && apt-get install -y \
    build-essential \       
    libssl-dev \            
    libffi-dev \            
    wget \                  
    curl \                  
    && rm -rf /var/lib/apt/lists/*   

# 3. Робоча директорія всередині контейнера
WORKDIR /app

# 4. Копіюємо усі файли проєкту в контейнер
COPY . .

# 5. Оновлення pip та встановлення Python-залежностей
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 6. Відкриваємо порт для Streamlit
EXPOSE 8501

# 7. Команда для запуску Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

