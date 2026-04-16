import os
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from dotenv import load_dotenv

load_dotenv()

# We use BioMistral-7B as the primary model.
# If it's unavailable on free endpoints, we fallback to mistralai/Mistral-7B-Instruct-v0.2
PRIMARY_MODEL = "BioMistral/BioMistral-7B"
FALLBACK_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

hf_token = os.getenv("HF_TOKEN")
# We initialize client. If hf_token is None, it uses the free public tier.
# It is better to have a token to increase rate limits.
client = InferenceClient(model=PRIMARY_MODEL, token=hf_token)
fallback_client = InferenceClient(model=FALLBACK_MODEL, token=hf_token)

def _generate_text(prompt: str, max_new_tokens: int = 512, temperature: float = 0.3) -> str:
    """Helper to generate text with fallback logic."""
    # We use chat_completion which is generally more stable and handles the inference routing better
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = client.chat_completion(
            messages,
            max_tokens=max_new_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"BioMistral inference failed ({e}). Falling back to {FALLBACK_MODEL}...")
        try:
            response = fallback_client.chat_completion(
                messages,
                max_tokens=max_new_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as fallback_e:
            print(f"Fallback inference failed: {fallback_e}")
            return "Error: Unable to generate response due to API limits or model unavailability."


def generate_clinical_narrative(analysis_data: dict, patient_meta: dict = None) -> str:
    """
    Takes ResNet-50 output + Grad-CAM metadata and generates 
    a radiologist-style clinical narrative via BioMistral.
    """
    tumor_loc = analysis_data.get('tumor_location')
    location_desc = "Not detected"
    if tumor_loc and analysis_data['label'] != 'No Tumor':
        hemisphere = "Right hemisphere" if tumor_loc.get('x', 0.5) > 0.5 else "Left hemisphere"
        location_desc = f"{hemisphere} (x:{tumor_loc.get('x'):.2f}, y:{tumor_loc.get('y'):.2f})"

    # Format the probability distribution
    probs = ""
    for cls, prob in analysis_data['probabilities'].items():
        probs += f"  - {cls}: {prob:.1%}\n"

    patient_context = f"PATIENT CONTEXT:\n{patient_meta}\n" if patient_meta else ""

    prompt = f"""<s>[INST] You are a board-certified neuroradiologist reviewing an AI-assisted brain MRI analysis. Generate a structured clinical impression. Keep it professional, concise, and do not use markdown features like bold or italics unnecessarily, plain text formatting is preferred.

ANALYSIS DATA:
- Primary AI Diagnosis: {analysis_data['label']}
- System Confidence: {analysis_data['confidence']:.1%}
- Probability Distribution:
{probs}
- Grad-CAM Peak Activation Localization: {location_desc}

{patient_context}
Generate exactly 4 sections:
1. Radiological Impression (2-3 sentences analyzing the finding)
2. Differential Considerations (Discuss the probabilities)
3. Recommended Follow-up Actions
4. Clinical Caveats (Remind that this is an AI-assisted finding)

Do NOT provide a definitive diagnosis. Frame it as AI-assisted findings requiring professional clinical correlation. [/INST]
"""

    narrative = _generate_text(prompt, max_new_tokens=512, temperature=0.3)
    
    # If the response failed heavily:
    if "Error:" in narrative and "API limits" in narrative:
        return f"System Note: The natural language narrative engine is currently unavailable due to API rate limits or model deployment status. Primary diagnosis remains {analysis_data['label']} with {analysis_data['confidence']:.1%} confidence."
        
    return narrative


def chat_with_oracle(message: str, history: list, analysis_context: dict = None) -> str:
    """
    Powers the Nexus Oracle chat using BioMistral.
    """
    
    context_str = ""
    if analysis_context:
        context_str = f"""
Current MRI Analysis Context:
- Primary Diagnosis: {analysis_context.get('label')}
- Confidence: {analysis_context.get('confidence', 0):.1%}
- Clinical Narrative so far: {analysis_context.get('clinical_narrative', 'N/A')}
"""

    # Format history into the prompt
    # Llama-style instruction format since BioMistral is Mistral-based
    prompt = f"<s>[INST] You are the Nexus Oracle, an advanced AI clinical assistant. Your role is to answer questions from a clinician about a recent MRI brain scan analysis. Be concise, highly clinical, and professional.\n{context_str}\n"
    
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        if role in ["clin", "user", "clinician"]:
            prompt += f"{content} [/INST] "
        else:
            prompt += f"{content} </s><s>[INST] "
            
    # Add the current message
    prompt += f"{message} [/INST]"
    
    response = _generate_text(prompt, max_new_tokens=300, temperature=0.5)
    
    if "Error:" in response and "API limits" in response:
        return "SYSTEM ERROR: Neural link disrupted. (Hugging Face API rate limit reached or model offline. Please provide an HF_TOKEN in the .env file)."
        
    return response

