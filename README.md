# MathBridge modular Streamlit app

## Structure

- `app.py`: small entry point only
- `mathbridge/config.py`: names, model and level rules
- `mathbridge/state.py`: Streamlit session state
- `mathbridge/gemma_service.py`: Gemma calls and response parsing
- `mathbridge/export_service.py`: Markdown/PDF generation
- `mathbridge/storage.py`: durable Supabase storage + SQLite fallback
- `mathbridge/teacher_service.py`: teacher batch evaluation
- `mathbridge/views/`: independent Student, Teacher and History screens

## Required Streamlit secret

```toml
GEMMA_API_KEY = "your-google-ai-studio-key"
```

## Durable saved sessions (recommended free option)

Create a free Supabase project, run `supabase_schema.sql` in its SQL editor, then add:

```toml
SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "your-service-role-key"
```

The service-role key must only be stored in Streamlit secrets. Never commit it to GitHub.

Without those two Supabase secrets, MathBridge automatically uses a local SQLite file. It survives browser refreshes while the app instance is alive, but Streamlit Community Cloud may erase it after a reboot or redeployment.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
