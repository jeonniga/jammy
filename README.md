# jammy

git clone https://github.com/kaifcoder/gemini_multipdf_chat.git

cd gemini_multipdf_chat

conda create -n jemmi python=3.10 -y

conda update -n base -c defaults conda

conda activate jemmi

pip install -r requirements.txt

streamlit run app.py

