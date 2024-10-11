import gradio as gr
import numpy as np
from PIL import Image
from skimage.util import random_noise
import cv2


# Function to add noise to the image
def add_noise(image, noise_type, mean=0, var=0.01, amount=0.05, salt_vs_pepper=0.5):
    # Convert image to float for processing
    image = np.array(image).astype(float) / 255.0  # Normalize the image
    kwargs = {}

    # Set noise parameters based on the selected noise type
    if noise_type in ['gaussian', 'speckle']:
        kwargs['mean'] = mean
        kwargs['var'] = var
    elif noise_type in ['salt', 'pepper', 's&p']:
        kwargs['amount'] = amount
        if noise_type == 's&p':
            kwargs['salt_vs_pepper'] = salt_vs_pepper
    elif noise_type == 'localvar':
        kwargs['local_vars'] = np.full(image.shape, var)

    # Add noise to the image
    noisy_image = random_noise(image, mode=noise_type.replace("s&p", "salt&pepper"), **kwargs, clip=True)
    return Image.fromarray((noisy_image * 255).astype(np.uint8))

# Function to apply denoising to the image
def apply_denoising(image, method, gaussian_kernel, median_kernel, bilateral_diameter, bilateral_sigma_color, bilateral_sigma_space, nlm_h, nlm_template_window_size, nlm_search_window_size):
    # Convert image to array for processing
    image = np.array(image)
    # Apply the selected denoising method
    if method == "Gaussian Blur":
        denoised = cv2.GaussianBlur(image, (gaussian_kernel, gaussian_kernel), 0)
    elif method == "Median Blur":
        denoised = cv2.medianBlur(image, median_kernel)
    elif method == "Bilateral Filter":
        denoised = cv2.bilateralFilter(image, bilateral_diameter, bilateral_sigma_color, bilateral_sigma_space)
    elif method == "Non-Local Means":
        denoised = cv2.fastNlMeansDenoisingColored(image, None, nlm_h, nlm_h, nlm_template_window_size, nlm_search_window_size)
    return Image.fromarray(denoised)

# Function to apply morphological operations
def apply_morphological_operation(image, kernel_size, iterations, operation):
    image = np.array(image)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == "Erosion":
        result = cv2.erode(image, kernel, iterations=iterations)
    elif operation == "Dilation":
        result = cv2.dilate(image, kernel, iterations=iterations)
    elif operation == "Opening":
        result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == "Closing":
        result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return Image.fromarray(result)

# Function to apply edge detection
def apply_edge_detection(image, min_val, max_val, operation, kernel_size):
    image = np.array(image.convert('L'))
    if operation == "Canny":
        edges = cv2.Canny(image, min_val, max_val)
    elif operation == "Sobel-X":
        edges = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    elif operation == "Sobel-Y":
        edges = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    elif operation == "Sobel-XY":
        edges_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
        edges_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
        edges = cv2.addWeighted(edges_x, 0.5, edges_y, 0.5, 0)
    elif operation == "Laplacian":
        edges = cv2.Laplacian(image, cv2.CV_64F, ksize=kernel_size)
    edges = np.clip(edges, 0, 255).astype(np.uint8)
    return Image.fromarray(edges)

# Gradio interface setup
with gr.Blocks() as demo:
    gr.Markdown("# OpenCV Image Processing with Gradio - Add Noise, Remove Noise, Morphological Operations and Edge Detection")

    tab_names = ["Add Noise", "Remove Noise", "Morphological Operations", "Edge Detection"]

    # ---- ADD NOISE TAB ----
    with gr.Tab("Add Noise"):
        with gr.Row():
            img_input = gr.Image(label="Input Image", type="pil")
            img_output = gr.Image(label="Output Image", type="pil")
        noise_type = gr.Radio(["gaussian", "localvar", "poisson", "salt", "pepper", "s&p", "speckle"], label="Type of Noise", value="gaussian")
        mean_slider = gr.Slider(0, 1, value=0, label="Mean (for Gaussian/Speckle)", visible=True)
        var_slider = gr.Slider(0, 0.1, value=0.01, label="Variance", visible=True)
        amount_slider = gr.Slider(0, 1, value=0.05, label="Amount (for Salt/Pepper/S&P)", visible=False)
        salt_vs_pepper_slider = gr.Slider(0, 1, value=0.5, label="Salt vs Pepper (for S&P)", visible=False)

        noise_button = gr.Button("Add Noise")

        def on_noise_type_change(noise_type):
            if noise_type in ['gaussian', 'speckle']:
                return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            elif noise_type in ['salt', 'pepper']:
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            elif noise_type == 's&p':
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
            elif noise_type == 'localvar':
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

        noise_type.change(fn=on_noise_type_change, inputs=noise_type, outputs=[mean_slider, var_slider, amount_slider, salt_vs_pepper_slider])
        noise_button.click(fn=add_noise, inputs=[img_input, noise_type, mean_slider, var_slider, amount_slider, salt_vs_pepper_slider], outputs=img_output)

    # ---- REMOVE NOISE TAB ----
    with gr.Tab("Remove Noise"):
        with gr.Row():
            denoise_img_input = gr.Image(label="Input Noisy Image", type="pil")
            denoise_img_output = gr.Image(label="Output Image", type="pil")
        denoise_method = gr.Radio(["Gaussian Blur", "Median Blur", "Bilateral Filter", "Non-Local Means"], label="Denoising Method", value="Gaussian Blur")
        gaussian_kernel = gr.Slider(1, 31, step=2, value=5, label="Gaussian Kernel Size", visible=True)
        median_kernel = gr.Slider(1, 31, step=2, value=5, label="Median Kernel Size", visible=False)
        bilateral_diameter = gr.Slider(1, 31, step=2, value=9, label="Bilateral Filter Diameter", visible=False)
        bilateral_sigma_color = gr.Slider(1, 150, value=75, label="Bilateral Filter Sigma Color", visible=False)
        bilateral_sigma_space = gr.Slider(1, 150, value=75, label="Bilateral Filter Sigma Space", visible=False)
        nlm_h = gr.Slider(1, 20, value=10, label="Non-Local Means h", visible=False)
        nlm_template_window_size = gr.Slider(1, 21, step=2, value=7, label="Non-Local Means Template Window Size", visible=False)
        nlm_search_window_size = gr.Slider(1, 51, step=2, value=21, label="Non-Local Means Search Window Size", visible=False)
        denoise_button = gr.Button("Remove Noise")

        def on_denoise_method_change(method):
            if method == "Gaussian Blur":
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            elif method == "Median Blur":
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            elif method == "Bilateral Filter":
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            elif method == "Non-Local Means":
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

        denoise_method.change(fn=on_denoise_method_change, inputs=denoise_method, outputs=[gaussian_kernel, median_kernel, bilateral_diameter, bilateral_sigma_color, bilateral_sigma_space, nlm_h, nlm_template_window_size, nlm_search_window_size])
        denoise_button.click(fn=apply_denoising, inputs=[denoise_img_input, denoise_method, gaussian_kernel, median_kernel, bilateral_diameter, bilateral_sigma_color, bilateral_sigma_space, nlm_h, nlm_template_window_size, nlm_search_window_size], outputs=denoise_img_output)

    # ---- MORPHOLOGICAL OPERATIONS TAB ----
    with gr.Tab("Morphological Operations"):
        with gr.Row():
            morph_img_input = gr.Image(label="Input Image", type="pil")
            morph_img_output = gr.Image(label="Output Image", type="pil")
        kernel_slider = gr.Slider(1, 11, value=3, step=2, label="Kernel Size")
        iter_slider = gr.Slider(1, 10, value=1, step=1, label="Iterations")
        morph_operation = gr.Radio(["Erosion", "Dilation", "Opening", "Closing"], label="Morphological Operation", value="Erosion")
        apply_morph_button = gr.Button("Apply Morphological Operation")
        apply_morph_button.click(fn=apply_morphological_operation, inputs=[morph_img_input, kernel_slider, iter_slider, morph_operation], outputs=morph_img_output)

    # ---- EDGE DETECTION TAB ----
    with gr.Tab("Edge Detection"):
        with gr.Row():
            edge_img_input = gr.Image(label="Input Image", type="pil")
            edge_img_output = gr.Image(label="Output Image", type="pil")
        min_val_slider = gr.Slider(50, 150, label="Min Threshold", visible=True)
        max_val_slider = gr.Slider(100, 200, label="Max Threshold", visible=True)
        kernel_size_slider = gr.Slider(1, 11, value=3, step=2, label="Kernel Size", visible=True)
        edge_operation = gr.Radio(["Canny", "Sobel-X", "Sobel-Y", "Sobel-XY", "Laplacian"], label="Edge Operation", value="Canny")
        apply_edge_button = gr.Button("Apply Edge Detection")

        def on_edge_operation_change(operation):
            if operation == "Canny":
                return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

        edge_operation.change(fn=on_edge_operation_change, inputs=edge_operation, outputs=[min_val_slider, max_val_slider, kernel_size_slider])
        apply_edge_button.click(fn=apply_edge_detection, inputs=[edge_img_input, min_val_slider, max_val_slider, edge_operation, kernel_size_slider], outputs=edge_img_output)

    gr.Markdown("### To transfer the image to another tab for further processing, select the source and destination tabs and click the Transfer Image button.")
    
    # ---- DYNAMIC TRANSFER BUTTON ----
    with gr.Row():
        source_tab_dropdown = gr.Dropdown(tab_names, label="Transfer From Tab")
        destination_tab_dropdown = gr.Dropdown(tab_names, label="Transfer To Tab")
        transfer_image_button = gr.Button("Transfer Image")

    def dynamic_image_transfer(add_noise_input, add_noise_output, denoise_input, denoise_output, morph_input, morph_output, edge_input, edge_output, source, destination):
        image_to_send = None
        if source == "Add Noise":
            image_to_send = add_noise_output if add_noise_output else add_noise_input
        elif source == "Remove Noise":
            image_to_send = denoise_output if denoise_output else denoise_input
        elif source == "Morphological Operations":
            image_to_send = morph_output if morph_output else morph_input
        elif source == "Edge Detection":
            image_to_send = edge_output if edge_output else edge_input

        updates = {
            "Add Noise": gr.update(value=image_to_send) if destination == "Add Noise" else gr.update(),
            "Remove Noise": gr.update(value=image_to_send) if destination == "Remove Noise" else gr.update(),
            "Morphological Operations": gr.update(value=image_to_send) if destination == "Morphological Operations" else gr.update(),
            "Edge Detection": gr.update(value=image_to_send) if destination == "Edge Detection" else gr.update(),
        }

        return [updates.get("Add Noise"), updates.get("Remove Noise"), updates.get("Morphological Operations"), updates.get("Edge Detection")]

    transfer_image_button.click(
        fn=dynamic_image_transfer,
        inputs=[img_input, img_output, denoise_img_input, denoise_img_output, morph_img_input, morph_img_output, edge_img_input, edge_img_output, source_tab_dropdown, destination_tab_dropdown],
        outputs=[img_input, denoise_img_input, morph_img_input, edge_img_input]
    )

    gr.Markdown("""
    ### Transfer Image Instructions:
    - **Select Source Tab**: Choose the tab from which you want to transfer the image (e.g., Add Noise, Remove Noise, Morphological Operations, Edge Detection).
    - **Select Destination Tab**: Choose the tab where you want to send the image.
    - **Click Transfer Image**: After selecting the source and destination, click the "Transfer Image" button to move the image to the selected tab.
    - **Note**: If the source tab has no processed image, the input image will be transferred instead.
    """)
# Launch the Gradio interface
demo.launch()
