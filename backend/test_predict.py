from predict import predict

image_path = "/Users/yashwantsinghsonwaniya/Documents/Deep Learning/Medical-Image-Diagnosis-System/my_Images/img1.jpg"

result, confidence = predict(image_path)

print("Prediction:", result)

print(f"Confidence: {confidence:.2f}%")