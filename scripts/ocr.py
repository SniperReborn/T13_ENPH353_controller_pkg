import cv2
import numpy as np
import tensorflow as tf

class SignReaderCNN:
    def __init__(self, model_path):
        # 1. Load the TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # --- CRITICAL: UPDATE THIS TO MATCH YOUR TRAINING LABELS ---
        # If your model predicts 0=A, 1=B, etc., this string must match that order exactly!
        self.class_mapping = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" 
        
        # Dynamically grab the input shape your model expects (e.g., 32x32, 1 channel)
        self.input_shape = self.input_details[0]['shape']
        self.img_height = self.input_shape[1]
        self.img_width = self.input_shape[2]
        self.channels = self.input_shape[3] if len(self.input_shape) > 3 else 1

    def predict_line(self, binary_image, boxes):
        """Takes the full binary image and a list of bounding boxes, returns a string."""
        predicted_text = ""
        
        for (x, y, w, h, c) in boxes:
            # 1. Crop the letter out of the binary image
            # Add a 2-pixel pad so we don't slice off the very edges of the letters
            pad = 2
            y1, y2 = max(0, y - pad), min(binary_image.shape[0], y + h + pad)
            x1, x2 = max(0, x - pad), min(binary_image.shape[1], x + w + pad)
            
            crop = binary_image[y1:y2, x1:x2]
            if crop.size == 0: 
                continue
                
            # 2. Resize to match what your CNN expects (e.g., 32x32)
            resized = cv2.resize(crop, (self.img_width, self.img_height))
            
            # 3. Preprocess / Normalize
            # Most CNNs expect float32 values between 0.0 and 1.0
            input_data = np.float32(resized) / 255.0
            
            # Expand dimensions to match the model's batch format: (1, Height, Width, Channels)
            input_data = np.reshape(input_data, self.input_shape)
            
            # 4. Run Inference!
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # 5. Get the highest probability class and map it to a letter
            predicted_index = np.argmax(output_data[0])
            predicted_text += self.class_mapping[predicted_index]
            
        return predicted_text