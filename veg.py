# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 01:00:50 2023


"""

import streamlit as st
from PIL import Image
import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the trained classifier
with open("C:/veg/vegetable.pkl", "rb") as model_file:
    clf = pickle.load(model_file)

st.title("Vegetable Image Classification App")
st.write("Upload an image for vegetable classification.")

# File uploader for image selection
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    original_image = Image.open(uploaded_image)
    st.image(original_image, caption="Original Image", use_column_width=True)
    st.write("")

    st.write("Just a second...")

    # Load and preprocess the uploaded image
    color_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    enhanced_image = cv2.equalizeHist(gray_image)
    smoothed_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
    #k-means
    k = 3
    pixel_values = color_image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()].reshape(color_image.shape)

    # Canny edge detection
    edges = cv2.Canny(segmented_image, 100, 200)

    # Define a function to extract HOG features
    def extract_hog_features(image):
        features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys", visualize=True)
        return features

    # Define a function to extract color histogram features
    def extract_color_histogram(image):
        b, g, r = cv2.split(image)
        hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
        hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
        color_hist_features = np.concatenate((hist_b, hist_g, hist_r)).flatten()
        return color_hist_features

    # Define a function to extract LBP features
    def extract_lbp_features(image):
        lbp_features = local_binary_pattern(image, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp_features.ravel(), bins=np.arange(0, 11), range=(0, 10))
        lbp_features = hist.astype("float")
        lbp_features /= (lbp_features.sum() + 1e-8)
        return lbp_features

    # Define a function to extract GLCM features
    def extract_glcm_features(image):
        glcm = graycomatrix(image, [1], [0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, "contrast")[0, 0]
        energy = graycoprops(glcm, "energy")[0, 0]
        entropy = -np.sum(glcm * np.log2(glcm + 1e-8))
        glcm_features = [contrast, energy, entropy]
        return glcm_features

    # Extract features from the image
    hog_features = extract_hog_features(smoothed_image)
    color_hist_features = extract_color_histogram(color_image)
    lbp_features = extract_lbp_features(smoothed_image)
    glcm_features = extract_glcm_features(smoothed_image)
    combined_features = np.concatenate((hog_features, color_hist_features, lbp_features, glcm_features))

    # Predict the class using the loaded classifier
    predicted_class = clf.predict([combined_features])
    # Display processed images row-wise
    col1, col2, col3,col4 = st.columns(4)  # Create 4 columns

    # Display Smoothed Image in the first column
    with col1:
        st.image(smoothed_image, caption="Smoothed Image", use_column_width=True)

    # Display Enhanced Image in the second column
    with col2:
        st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)

    # Display Segmented Image in the third column
    with col3:
        st.image(segmented_image, caption="Segmented Image", use_column_width=True)

    # Display Canny Edged Image in the fourth column
    with col4:
        st.image(edges, caption="Canny Edged Image", use_column_width=True)
    # Display the predicted class
    st.write("Predicted Class:", predicted_class[0])



