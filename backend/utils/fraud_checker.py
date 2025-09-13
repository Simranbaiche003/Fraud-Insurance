import re

def check_fraud(claim_data, hospitals_df, diseases_df):
    """
    Checks for potential fraud based on hospital, disease, and treatment data.
    Returns: clean, suspicious, or fraudulent
    """
    hospital_name = claim_data.get("hospital", "").strip()
    disease = claim_data.get("disease", "").strip()
    treatment = claim_data.get("treatment", "").strip()
    amount = claim_data.get("amount", 0)
    patient_name = claim_data.get("patientName", "").strip()
    claim_id = claim_data.get("claimId", "").strip()

    print("\n--- Checking for Fraud ---")
    print(f"Hospital Name Extracted: '{hospital_name}'")
    print(f"Disease Extracted: '{disease}'")
    print(f"Treatment Extracted: '{treatment}'")
    print(f"Amount Extracted: {amount}")
    print(f"Patient Name: '{patient_name}'")
    print(f"Claim ID: '{claim_id}'")

    fraud_status = "clean"
    fraud_reason = ""

    # FRAUDULENT CHECKS (Critical failures)
    
    # Check 1: Missing critical information
    if not hospital_name or not disease or not treatment or not patient_name:
        fraud_status = "fraudulent"
        fraud_reason = "Missing critical claim information"
        print(f"Fraud Detected: {fraud_reason}")
        return fraud_status, fraud_reason

    # Check 2: Template/placeholder text detection
    placeholder_patterns = [
        r"enter\s+\w+\s+name", r"enter\s+claimed\s+amount", r"provide\s+any\s+extra",
        r"patient\s+name", r"hospital\s+name", r"claim\s+amount"
    ]
    
    combined_text = f"{hospital_name} {disease} {treatment} {patient_name}".lower()
    for pattern in placeholder_patterns:
        if re.search(pattern, combined_text):
            fraud_status = "fraudulent"
            fraud_reason = "Document contains template/placeholder text"
            print(f"Fraud Detected: {fraud_reason}")
            return fraud_status, fraud_reason

    # Check 3: If the hospital is not in the dataset, it's fraudulent
    if hospital_name not in hospitals_df['HospitalName'].values:
        fraud_status = "fraudulent"
        fraud_reason = "Hospital not in dataset"
        print(f"Fraud Detected: {fraud_reason}")
        return fraud_status, fraud_reason

    # Check 4: If the disease is not in the dataset, it's fraudulent
    disease_row = diseases_df[diseases_df['Disease'].str.lower() == disease.lower()]
    if disease_row.empty:
        fraud_status = "fraudulent"
        fraud_reason = "Disease not in dataset"
        print(f"Fraud Detected: {fraud_reason}")
        return fraud_status, fraud_reason

    # Check 5: If the treatment is not valid for the disease, it's fraudulent
    valid_treatments_str = disease_row.iloc[0]['Treatment']
    valid_treatments = [t.strip().lower() for t in valid_treatments_str.split(',')]
    
    if treatment.lower() not in valid_treatments:
        fraud_status = "fraudulent"
        fraud_reason = "Treatment mismatch for disease"
        print(f"Fraud Detected: {fraud_reason}")
        return fraud_status, fraud_reason

    # SUSPICIOUS CHECKS (Warning flags)
    
    # Check 6: High amount claims (over ₹100,000)
    if amount > 100000:
        fraud_status = "suspicious"
        fraud_reason = "High claim amount requires additional verification"
        print(f"Suspicious Activity: {fraud_reason}")
        return fraud_status, fraud_reason
    
    # Check 7: Emergency/urgent treatments (higher risk)
    emergency_keywords = ['emergency', 'urgent', 'critical', 'immediate', 'trauma']
    if any(keyword in treatment.lower() for keyword in emergency_keywords):
        fraud_status = "suspicious"
        fraud_reason = "Emergency treatment requires additional review"
        print(f"Suspicious Activity: {fraud_reason}")
        return fraud_status, fraud_reason
    
    # Check 8: Expensive procedures for common diseases
    expensive_treatments = ['surgery', 'operation', 'transplant', 'bypass', 'angioplasty']
    common_diseases = ['fever', 'cold', 'headache', 'cough']
    
    if (any(exp_treatment in treatment.lower() for exp_treatment in expensive_treatments) and 
        any(common_disease in disease.lower() for common_disease in common_diseases)):
        fraud_status = "suspicious"
        fraud_reason = "Expensive treatment for common condition requires review"
        print(f"Suspicious Activity: {fraud_reason}")
        return fraud_status, fraud_reason
    
    # Check 9: Very short claim IDs (less than 5 characters)
    if len(claim_id) < 5:
        fraud_status = "suspicious"
        fraud_reason = "Claim ID format appears irregular"
        print(f"Suspicious Activity: {fraud_reason}")
        return fraud_status, fraud_reason
    
    # Check 10: Round number amounts (exactly divisible by 10000)
    if amount > 0 and amount % 10000 == 0 and amount > 50000:
        fraud_status = "suspicious"
        fraud_reason = "Suspicious round-number claim amount"
        print(f"Suspicious Activity: {fraud_reason}")
        return fraud_status, fraud_reason
    
    # Check 11: Very common patient names (potential fake names)
    common_fake_names = ['john doe', 'jane doe', 'test user', 'sample patient', 'demo patient']
    if patient_name.lower() in common_fake_names:
        fraud_status = "suspicious"
        fraud_reason = "Patient name appears to be placeholder or generic"
        print(f"Suspicious Activity: {fraud_reason}")
        return fraud_status, fraud_reason

    # All checks passed - Clean claim
    print("No fraud detected. Claim is clean.")
    return "clean", "All checks passed - claim appears legitimate"