# This is conceptual server-side Python code.
# You'll need to install Flask, OpenCV, and potentially a deep learning framework.
# pip install Flask opencv-python numpy

from flask import Flask, request, send_file
import cv2
import numpy as np
from io import BytesIO
import random

app = Flask(__name__)

# Placeholder for a hair detection model (YOU WOULD NEED TO IMPLEMENT/LOAD THIS)
# This is the most complex part. For a real app, you'd load a deep learning model.
def detect_hair_mask(image_np):
    # This is a VERY SIMPLIFIED placeholder.
    # In reality, this would involve a complex deep learning model to segment hair.
    # For demonstration, let's just assume the top 30% of the image is "hair-like"
    # or use a very basic color threshold (which won't work well for real hair).

    # A more realistic approach would load a model like:
    # net = cv2.dnn.readNetFromTensorflow('path/to/hair_model.pb', 'path/to/config.pbtxt')
    # Then preprocess the image, run inference, and get a hair segmentation mask.

    # MOCK HAIR MASK: Let's assume a rough top region or a simple color filter
    # For actual hair detection, research "semantic segmentation human hair"
    height, width, _ = image_np.shape
    hair_mask = np.zeros((height, width), dtype=np.uint8)

    # Simple mock: Assume top 30% might contain hair
    hair_mask[0:int(height * 0.3), :] = 255 # Mark top 30% as "hair"
    
    # Or for a slightly more "intelligent" mock, let's say dark pixels in the top half
    # gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    # hair_mask = np.zeros_like(gray_image)
    # hair_mask[gray_image < 50] = 255 # Arbitrary threshold for dark colors

    # You'd replace this with actual hair segmentation logic
    return hair_mask

@app.route('/generate-hair-lines', methods=['POST'])
def generate_hair_lines():
    if 'image' not in request.files:
        return 'No image file provided', 400

    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    # Read the image
    in_memory_file = BytesIO()
    file.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) # Reads as BGR by default

    if img is None:
        return 'Could not decode image', 400

    # 1. Get Hair Mask
    hair_mask = detect_hair_mask(img) # This is where your complex hair detection goes

    # 2. Find hair pixel coordinates
    hair_y, hair_x = np.where(hair_mask > 0) # Get coordinates where hair_mask is active

    if len(hair_x) == 0:
        # If no hair detected by the mock, just return original or handle error
        print("No hair detected with the mock function.")
        is_success, buffer = cv2.imencode(".png", img)
        return send_file(BytesIO(buffer.tobytes()), mimetype='image/png', as_attachment=False)


    # 3. Draw random red lines
    output_img = img.copy() # Work on a copy
    num_lines = 100 # Number of random lines to draw
    line_length_max = 50 # Max length of lines
    line_thickness = 1 # Thickness of lines
    line_color = (0, 0, 255) # Red in BGR format (OpenCV uses BGR)

    for _ in range(num_lines):
        # Pick a random starting point from detected hair pixels
        idx = random.randint(0, len(hair_x) - 1)
        start_x, start_y = hair_x[idx], hair_y[idx]

        # Generate a random angle (0 to 360 degrees)
        angle = random.uniform(0, 2 * np.pi)
        
        # Generate random length
        length = random.randint(10, line_length_max)

        # Calculate end point
        end_x = int(start_x + length * np.cos(angle))
        end_y = int(start_y + length * np.sin(angle))

        # Draw the line
        cv2.line(output_img, (start_x, start_y), (end_x, end_y), line_color, line_thickness)

    # Encode the processed image to send back
    is_success, buffer = cv2.imencode(".png", output_img)
    if not is_success:
        return 'Failed to encode image', 500

    return send_file(
        BytesIO(buffer.tobytes()),
        mimetype='image/png',
        as_attachment=False,
        download_name='hair_lines_output.png'
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000) # Run on port 5000, for example
