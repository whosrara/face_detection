from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from ssim1 import compare_images  # Import the compare_image function

import cv2
import numpy as np

app = FastAPI()

@app.post("/compare")
async def compare(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    try:

        # Read images as binary
        image1_content = await image1.read()
        image2_content = await image2.read()
        
        nparr1 = np.frombuffer(image1_content, np.uint8)
        img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)

        nparr2 = np.frombuffer(image2_content, np.uint8)
        img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

        # Call the compare_image function. You might need to adjust this part
        # depending on how compare_image is implemented in ssim1.py.
        # Assuming compare_image accepts bytes and returns a similarity score or result
        result = compare_images(img1, img2)
        # print(result)
        # Return the result as JSON
        return JSONResponse(content={"result": result})
    except Exception as e:
        print(e)
        return JSONResponse(content={"error": True})


# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
