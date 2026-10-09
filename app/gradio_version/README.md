# Python (Gradio) version of the demo

Same three tabs as the web page in `app/`, but the model runs in Python. Hugging Face only hosts
this kind of app on paid hardware, so the published demo is the static page in `app/index.html`.

Run it on your own computer from the project folder:

```
pip install gradio spaces transformers torch plotly
python app/gradio_version/app.py
```

It reads the tone index from `data/processed/` and loads the model from the Hub
(the model must be public, or you must be logged in).
