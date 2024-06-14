このサービスは、顔認証ログインと、LLMによるチャットサービスです。また、文章の類似度分析により、用意されたタスクの実行も可能です。例えば、自然言語からWebデザインを変更することができたり、文字を入力することができます。（英語のみです） 自然言語によるタスク実行プログラムはagentディレクトリの中にあります。modelと付いているディレクトリの中身はspacyの文章の分析のためのモデルなので、プログラムのファイルはありません。

This service includes face recognition login and chat service by LLM. It also allows you to execute prepared tasks by analyzing the similarity of text. For example, you can change the web design from natural language or input text. (Only available in English) The natural language task execution program is in the agent directory. The directory named model contains no program files, as it is a model for analyzing Spacy sentences.

Step-by-step instructions

1 pip install Flask==2.2.5 torch==2.2.2 transformers==4.39.3 scipy==1.12.0 spacy==3.7.4 llama-cpp-python==0.2.6 python-docx==1.1.0 langchain==0.1.16 sentence-transformers==2.7.0 faiss-cpu==1.8.0 scikit-learn==1.3.2 numpy==1.23.3 tensorflow==2.12.0 opencv-python== 4.9.0.80

2 Download and set your own LLM file(.gguf) in the same directry as app.py

3 Move to app.py directory.

4 Command run “python3 app.py”
