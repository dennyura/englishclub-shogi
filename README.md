## ローカルサーバー立ち上げ
uv run uvicorn shogi_ai.web.app:app --reload

 http://127.0.0.1:8000/


import shutil

# コピー元（Input）とコピー先（Working）のパスを指定
src = "/kaggle/input/datasets/dennyura/englishclub-shogi/best_model_animal.pt"
dst = "/kaggle/working"

shutil.copy("/kaggle/input/datasets/dennyura/englishclub-shogi/best_model_animal.pt", "/kaggle/working/")


## ライセンス

MIT License
