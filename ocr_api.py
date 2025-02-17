from fastapi import FastAPI, File, UploadFile
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import io

# Load the AI OCR model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

app = FastAPI()

@app.post("/ocr")
async def extract_text(file: UploadFile = File(...)):
    try:
        # Convert image to PIL format
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")

        # Process the image through TrOCR
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return {"text": extracted_text}

    except Exception as e:
        return {"error": str(e)}