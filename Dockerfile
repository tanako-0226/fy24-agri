FROM python:3.10-bullseye

# RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install streamlit numpy pandas matplotlib