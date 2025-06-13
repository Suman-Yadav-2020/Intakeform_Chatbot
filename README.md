# Intakeform_Chatbot

✅ Step-by-Step Setup (Windows)
🧩 1. Make sure you’re in your project folder
Open Command Prompt or PowerShell, and run:

bash
Copy
Edit
cd path\to\intake_form_backend
Replace path\to with the actual location where your project is.

🛠️ 2. Create the virtual environment
Run:

bash
Copy
Edit
python -m venv venv
This will create a folder called venv with all the necessary scripts.

✅ 3. Activate the virtual environment
Now run:

bash
Copy
Edit
.\venv\Scripts\activate
Once activated, your terminal should look like:

scss
Copy
Edit
(venv) C:\Users\YourName\intake_form_backend>
📦 4. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
🚀 5. Run the FastAPI app
Use:

bash
Copy
Edit
uvicorn main:app --reload
If uvicorn still doesn’t work, use:

bash
Copy
Edit
python -m uvicorn main:app --reload
🔍 Checkpoint
To verify it's working:

Open your browser

Go to: http://localhost:8000/docs

You should see the FastAPI Swagger UI
