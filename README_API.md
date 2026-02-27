# ATS Project - Running the Full Stack Application

## Prerequisites
- Python 3.8+
- Node.js 18+
- npm or bun

## Backend Setup

1. Navigate to the Backend folder:
```powershell
cd Backend
```

2. Create and activate a virtual environment:
```powershell
python -m venv venv
venv\Scripts\activate
```

3. Install all dependencies:
```powershell
pip install -r requirements.txt
pip install email-validator
pip install aiosqlite
```

4. Run the backend server:
```powershell
python main.py
```
The API will be available at http://localhost:8000

## Frontend Setup

1. Navigate to the Frontend folder:
```powershell
cd Frontend
```

2. Install dependencies:
```powershell
npm install
```

3. Run the frontend development server:
```powershell
npm run dev
```
The website will be available at http://localhost:8080

## To Use the Application

1. Start the backend first (it will run on port 8000)
2. Start the frontend (it will run on port 8080)
3. Open http://localhost:8080 in your browser
4. Register a new account or sign in
5. Upload a resume and job description to analyze

## Database

- Development: SQLite (automatically created as `ats_database.db`)
- Production: PostgreSQL (change DATABASE_URL in .env)
