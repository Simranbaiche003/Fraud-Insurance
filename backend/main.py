import shutil
import os
import re
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pdfplumber
from PIL import Image
import pytesseract

app = FastAPI(title="Insurance Fraud Detection API", version="1.0.0")

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dataset loader functions
def load_hospitals():
    """
    Loads the hospital dataset and cleans up the column names.
    """
    try:
        file_path = os.path.join(os.path.dirname(__file__), "datasets", "Hospital_Dataset.xlsx")
        if not os.path.exists(file_path):
            print(f"Warning: Hospital dataset not found at {file_path}")
            # Create mock data
            return pd.DataFrame({
                'HospitalName': ['Apollo Hospital', 'Max Healthcare', 'Fortis Hospital', 'AIIMS', 'Manipal Hospital']
            })
        hospitals_df = pd.read_excel(file_path)
        hospitals_df.columns = hospitals_df.columns.str.strip()
        return hospitals_df
    except Exception as e:
        print(f"Error loading hospitals dataset: {e}")
        # Return mock data
        return pd.DataFrame({
            'HospitalName': ['Apollo Hospital', 'Max Healthcare', 'Fortis Hospital', 'AIIMS', 'Manipal Hospital']
        })

def load_diseases():
    """
    Loads the disease-treatment dataset and cleans up the column names.
    """
    try:
        file_path = os.path.join(os.path.dirname(__file__), "datasets", "disease_treatment_dataset.xlsx")
        if not os.path.exists(file_path):
            print(f"Warning: Disease dataset not found at {file_path}")
            # Create mock data
            return pd.DataFrame({
                'Disease': ['Diabetes', 'Hypertension', 'Heart Disease', 'Cancer', 'Pneumonia'],
                'Treatment': ['Insulin, Metformin', 'ACE Inhibitors, Diuretics', 'Bypass Surgery, Angioplasty', 'Chemotherapy, Radiation', 'Antibiotics, Oxygen Therapy']
            })
        diseases_df = pd.read_excel(file_path)
        diseases_df.columns = diseases_df.columns.str.strip()
        return diseases_df
    except Exception as e:
        print(f"Error loading diseases dataset: {e}")
        # Return mock data
        return pd.DataFrame({
            'Disease': ['Diabetes', 'Hypertension', 'Heart Disease', 'Cancer', 'Pneumonia'],
            'Treatment': ['Insulin, Metformin', 'ACE Inhibitors, Diuretics', 'Bypass Surgery, Angioplasty', 'Chemotherapy, Radiation', 'Antibiotics, Oxygen Therapy']
        })

# Fraud checking function
def check_fraud(claim_data, hospitals_df, diseases_df):
    """
    Checks for potential fraud based on hospital, disease, and treatment data.
    """
    hospital_name = claim_data.get("hospital", "").strip()
    disease = claim_data.get("disease", "").strip()
    treatment = claim_data.get("treatment", "").strip()
    amount = claim_data.get("amount", 0)

    print("\n--- Checking for Fraud ---")
    print(f"Hospital Name Extracted: '{hospital_name}'")
    print(f"Disease Extracted: '{disease}'")
    print(f"Treatment Extracted: '{treatment}'")
    print(f"Amount Extracted: {amount}")

    fraud_status = "clean"
    fraud_reason = ""

    # Check 1: If the hospital is not in the dataset, it's fraudulent
    if hospital_name and hospital_name not in hospitals_df['HospitalName'].values:
        fraud_status = "fraudulent"
        fraud_reason = "Hospital not in dataset"
        print(f"Fraud Detected: {fraud_reason}")
        return fraud_status, fraud_reason

    # Check 2: If the disease is not in the dataset, it's fraudulent
    if disease:
        disease_row = diseases_df[diseases_df['Disease'].str.lower() == disease.lower()]
        if disease_row.empty:
            fraud_status = "fraudulent"
            fraud_reason = "Disease not in dataset"
            print(f"Fraud Detected: {fraud_reason}")
            return fraud_status, fraud_reason

        # Check 3: If the treatment is not valid for the disease, it's fraudulent
        if treatment:
            valid_treatments_str = disease_row.iloc[0]['Treatment']
            valid_treatments = [t.strip().lower() for t in valid_treatments_str.split(',')]
            
            if treatment.lower() not in valid_treatments:
                fraud_status = "fraudulent"
                fraud_reason = "Treatment mismatch for disease"
                print(f"Fraud Detected: {fraud_reason}")
                return fraud_status, fraud_reason

    # Check for suspicious amounts
    if amount > 500000:  # Amount greater than 5 lakhs
        fraud_status = "suspicious"
        fraud_reason = "Unusually high claim amount"
        print(f"Suspicious activity detected: {fraud_reason}")
        return fraud_status, fraud_reason

    print("No fraud detected. Claim is clean.")
    return "clean", "All checks passed."

# PDF and image text extraction functions
def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file using pdfplumber.
    """
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_image(file_path: str) -> str:
    """
    Extracts text from an image using pytesseract OCR.
    """
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""

# Load datasets on startup
try:
    hospitals_df = load_hospitals()
    diseases_df = load_diseases()
    print("Datasets loaded successfully")
except Exception as e:
    print(f"Error loading datasets: {e}")
    hospitals_df = pd.DataFrame({'HospitalName': []})
    diseases_df = pd.DataFrame({'Disease': [], 'Treatment': []})

@app.get("/")
async def root():
    return {"message": "Insurance Fraud Detection API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "hospitals_count": len(hospitals_df), "diseases_count": len(diseases_df)}

@app.post("/api/claims/upload")
async def upload_claim(file: UploadFile = File(...)):
    """
    Handles file upload, extracts text, and performs a fraud check.
    """
    temp_file_path = f"temp_{file.filename}"
    try:
        # Save uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract text based on file type
        if file.filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(temp_file_path)
        else:
            text = extract_text_from_image(temp_file_path)

        if not text:
            raise HTTPException(status_code=400, detail="Failed to extract text from the document.")

        print(f"Extracted text: {text[:500]}...")  # Print first 500 characters for debugging

        # Updated regex patterns to be more flexible
        hospital_match = re.search(r"Hospital(?:\s+Name)?:?\s*(.+)", text, re.IGNORECASE | re.MULTILINE)
        disease_match = re.search(r"Disease[:\-\s]+(.+)", text, re.IGNORECASE | re.MULTILINE)
        treatment_match = re.search(r"Treatment[:\-\s]+(.+)", text, re.IGNORECASE | re.MULTILINE)
        amount_match = re.search(r"(?:Claimed\s+)?Amount:?\s*[₹$]?[\s]*([0-9,\.]+)", text, re.IGNORECASE | re.MULTILINE)
        patient_name_match = re.search(r"(?:Policy\s+Holder|Patient)\s+Name:?\s*(.+)", text, re.IGNORECASE | re.MULTILINE)
        claim_id_match = re.search(r"Claim\s+(?:No|ID|Number):?\s*([\w\d/]+)", text, re.IGNORECASE | re.MULTILINE)
        
        # Initialize variables with empty strings
        hospital = ""
        disease = ""
        treatment = ""
        amount = 0
        patient_name = ""
        claim_id = ""

        # Safely extract data
        if hospital_match:
            hospital = hospital_match.group(1).strip().split('\n')[0]  # Take first line only
        
        if disease_match:
            disease = disease_match.group(1).strip().split('\n')[0]
            
        if treatment_match:
            treatment = treatment_match.group(1).strip().split('\n')[0]
            
        if amount_match:
            amount_str = amount_match.group(1).replace(",", "").strip()
            try:
                amount = int(float(amount_str))
            except ValueError:
                amount = 0

        if patient_name_match:
            patient_name = patient_name_match.group(1).strip().split('\n')[0]
            
        if claim_id_match:
            claim_id = claim_id_match.group(1).strip()

        claim_data = {
            "hospital": hospital,
            "disease": disease,
            "treatment": treatment,
            "amount": amount,
            "patientName": patient_name,
            "claimId": claim_id,
        }
        
        print("\n--- Extracted Data for Fraud Check ---")
        print(claim_data)
        print("------------------------------------")
        
        # Perform fraud check
        fraud_status, fraud_reason = check_fraud(claim_data, hospitals_df, diseases_df)
        
        claim_data["fraudStatus"] = fraud_status
        claim_data["fraudReason"] = fraud_reason
        
        return JSONResponse(
            content={"status": "success", "extractedData": claim_data},
            status_code=200
        )
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                print(f"Error removing temp file: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)