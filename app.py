import gradio as gr
import cv2
import tempfile
import os
from ultralytics import YOLO


def yolov12_inference(image, video, webcam, model_id, image_size, conf_threshold):
    try:
        # Handle the case where model_id might be a file path or a model name
        if os.path.exists(model_id):
            model = YOLO(model_id)
        else:
            # If it's not a file path, try to load it as a model name
            model = YOLO(model_id)
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        # Fall back to a default model
        try:
            # Try to use YOLOv8 models which are more stable
            for model_name in ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']:
                if os.path.exists(model_name):
                    print(f"Falling back to available model: {model_name}")
                    model = YOLO(model_name)
                    break
            else:
                # If none of the models exist as files, try loading directly
                model = YOLO('yolov8n.pt')
        except Exception as e2:
            print(f"Error loading fallback model: {e2}")
            raise Exception(f"Failed to load any model: {e}, {e2}")
        
    # Process based on input type
    if image is not None and image is not False:
        try:
            results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
            if len(results) > 0:
                annotated_image = results[0].plot()
                return annotated_image[:, :, ::-1], None
            else:
                # Handle case where results might be empty
                print("No results from model prediction")
                return image, None
        except Exception as e:
            print(f"Error in image prediction: {e}")
            return image, None
    elif video is not None and video is not False:
        try:
            video_path = tempfile.mktemp(suffix=".webm")
            with open(video_path, "wb") as f:
                with open(video, "rb") as g:
                    f.write(g.read())

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                return None, video
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            output_video_path = tempfile.mktemp(suffix=".webm")
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'vp80'), fps, (frame_width, frame_height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
                if len(results) > 0:
                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)
                else:
                    # If no results, just write the original frame
                    out.write(frame)

            cap.release()
            out.release()

            return None, output_video_path
        except Exception as e:
            print(f"Error in video processing: {e}")
            return None, video
    elif webcam is not None and webcam is not False:
        # Process webcam input the same way as image input
        try:
            results = model.predict(source=webcam, imgsz=image_size, conf=conf_threshold)
            if len(results) > 0:
                annotated_image = results[0].plot()
                return annotated_image[:, :, ::-1], None
            else:
                print("No results from webcam prediction")
                return webcam, None
        except Exception as e:
            print(f"Error in webcam prediction: {e}")
            return webcam, None
    else:
        # Handle case where no valid input is provided
        print("No valid input provided")
        return None, None


def yolov12_inference_for_examples(image, model_path, image_size, conf_threshold):
    try:
        annotated_image, _ = yolov12_inference(image, None, None, model_path, image_size, conf_threshold)
        return annotated_image
    except Exception as e:
        print(f"Error in inference for examples: {e}")
        # Return the original image if there's an error
        return image


def road_cracks_inference(image, video, webcam, model_path, image_size, conf_threshold):
    # Use the trained road cracks model or fall back to a default model if not found
    try:
        if os.path.exists(model_path):
            model = YOLO(model_path)
        else:
            # Fall back to default YOLOv12 model if the road cracks model isn't found
            print(f"Road cracks model not found at {model_path}, using default YOLOv12 model")
            model = YOLO('yolov12n.pt')
    except Exception as e:
        print(f"Error loading model: {e}")
        try:
            # Fall back to a default model
            model = YOLO('yolov12n.pt')
        except Exception as e2:
            print(f"Error loading fallback model: {e2}")
            raise Exception(f"Failed to load any model: {e}, {e2}")
    
    # Process based on input type
    if image is not None and image is not False:
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
        annotated_image = results[0].plot()
        
        # Get detection information
        boxes = results[0].boxes
        class_names = model.names
        detections = []
        
        for i, box in enumerate(boxes):
            cls_id = int(box.cls.item())
            conf = box.conf.item()
            class_name = class_names[cls_id]
            detections.append(f"{class_name}: {conf:.2f}")
        
        return annotated_image[:, :, ::-1], None, "\n".join(detections) if detections else "No cracks detected"
    elif video is not None and video is not False:
        video_path = tempfile.mktemp(suffix=".webm")
        with open(video_path, "wb") as f:
            with open(video, "rb") as g:
                f.write(g.read())

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video_path = tempfile.mktemp(suffix=".webm")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'vp80'), fps, (frame_width, frame_height))

        all_detections = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            
            # Collect detections from each frame
            boxes = results[0].boxes
            class_names = model.names
            for i, box in enumerate(boxes):
                cls_id = int(box.cls.item())
                conf = box.conf.item()
                class_name = class_names[cls_id]
                all_detections.append(f"{class_name}: {conf:.2f}")

        cap.release()
        out.release()
        
        # Get unique detections
        unique_detections = list(set(all_detections))
        detection_summary = "\n".join(unique_detections) if unique_detections else "No cracks detected"

        return None, output_video_path, detection_summary
    elif webcam is not None and webcam is not False:
        # Process webcam input the same way as image input
        results = model.predict(source=webcam, imgsz=image_size, conf=conf_threshold)
        annotated_image = results[0].plot()
        
        # Get detection information
        boxes = results[0].boxes
        class_names = model.names
        detections = []
        
        for i, box in enumerate(boxes):
            cls_id = int(box.cls.item())
            conf = box.conf.item()
            class_name = class_names[cls_id]
            detections.append(f"{class_name}: {conf:.2f}")
        
        return annotated_image[:, :, ::-1], None, "\n".join(detections) if detections else "No cracks detected"
    else:
        # Handle case where no valid input is provided
        print("No valid input provided")
        return None, None, "No input provided"


def road_cracks_inference_for_examples(image, model_path, image_size, conf_threshold):
    try:
        annotated_image, _, detections = road_cracks_inference(image, None, None, model_path, image_size, conf_threshold)
        return annotated_image, detections
    except Exception as e:
        print(f"Error in road cracks inference for examples: {e}")
        # Return the original image and an error message if there's an error
        return image, "Error processing image"


def general_detection_app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image", visible=True)
                video = gr.Video(label="Video", visible=False)
                webcam = gr.Image(type="pil", label="Webcam", visible=False, sources=["webcam"])
                input_type = gr.Radio(
                    choices=["Image", "Video", "Webcam"],
                    value="Image",
                    label="Input Type",
                )
                model_id = gr.Dropdown(
                    label="Model",
                    choices=[
                        "yolov8n.pt",
                        "yolov8s.pt",
                        "yolov8m.pt",
                        "yolov8l.pt",
                        "yolov8x.pt",
                    ],
                    value="yolov8n.pt",  # Use the smallest model by default for faster loading
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.25,
                )
                yolov12_infer = gr.Button(value="Detect Objects")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)
                output_video = gr.Video(label="Annotated Video", visible=False)

        def update_visibility(input_type):
            image_visible = input_type == "Image"
            video_visible = input_type == "Video"
            webcam_visible = input_type == "Webcam"
            output_image_visible = input_type in ["Image", "Webcam"]
            output_video_visible = input_type == "Video"

            return [
                gr.update(visible=image_visible),
                gr.update(visible=video_visible),
                gr.update(visible=webcam_visible),
                gr.update(visible=output_image_visible),
                gr.update(visible=output_video_visible),
            ]

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, video, webcam, output_image, output_video],
        )

        def run_inference(image, video, webcam, model_id, image_size, conf_threshold, input_type):
            if input_type == "Image":
                return yolov12_inference(image, None, None, model_id, image_size, conf_threshold)
            elif input_type == "Video":
                return yolov12_inference(None, video, None, model_id, image_size, conf_threshold)
            else:  # Webcam
                return yolov12_inference(None, None, webcam, model_id, image_size, conf_threshold)

        yolov12_infer.click(
            fn=run_inference,
            inputs=[image, video, webcam, model_id, image_size, conf_threshold, input_type],
            outputs=[output_image, output_video],
        )

        # Use the downloaded Ultralytics example images
        example_images = []
        
        # Check if example images directory exists and add some example images
        example_images_dir = "example_images"
        if os.path.exists(example_images_dir):
            image_files = os.listdir(example_images_dir)[:3]  # Get first 3 example images
            for img_file in image_files:
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    example_images.append([
                        os.path.join(example_images_dir, img_file),
                        model_id.value,
                        640,
                        0.25,
                    ])
        
        # If no example images found, don't use examples
        if not example_images:
            print("No example images found")
        
        if example_images:
            try:
                gr.Examples(
                    examples=example_images,
                    fn=yolov12_inference_for_examples,
                    inputs=[
                        image,
                        model_id,
                        image_size,
                        conf_threshold,
                    ],
                    outputs=[output_image],
                    cache_examples='lazy',
                )
            except Exception as e:
                print(f"Error setting up examples: {e}")


def road_cracks_detection_app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image", visible=True)
                video = gr.Video(label="Video", visible=False)
                webcam = gr.Image(type="pil", label="Webcam", visible=False, sources=["webcam"])
                input_type = gr.Radio(
                    choices=["Image", "Video", "Webcam"],
                    value="Image",
                    label="Input Type",
                )
                # Path to the trained road cracks model
                model_path = gr.Textbox(
                    label="Road Cracks Model Path",
                    value="road_cracks_results/road_cracks_model/weights/best.pt",
                    info="Path to the trained road cracks model"
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.25,
                )
                detect_button = gr.Button(value="Detect Road Cracks")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)
                output_video = gr.Video(label="Annotated Video", visible=False)
                detection_info = gr.Textbox(label="Detection Information", lines=5)

        def update_visibility(input_type):
            image_visible = input_type == "Image"
            video_visible = input_type == "Video"
            webcam_visible = input_type == "Webcam"
            output_image_visible = input_type in ["Image", "Webcam"]
            output_video_visible = input_type == "Video"

            return [
                gr.update(visible=image_visible),
                gr.update(visible=video_visible),
                gr.update(visible=webcam_visible),
                gr.update(visible=output_image_visible),
                gr.update(visible=output_video_visible),
            ]

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, video, webcam, output_image, output_video],
        )

        def run_inference(image, video, webcam, model_path, image_size, conf_threshold, input_type):
            if input_type == "Image":
                return road_cracks_inference(image, None, None, model_path, image_size, conf_threshold)
            elif input_type == "Video":
                return road_cracks_inference(None, video, None, model_path, image_size, conf_threshold)
            else:  # Webcam
                return road_cracks_inference(None, None, webcam, model_path, image_size, conf_threshold)

        detect_button.click(
            fn=run_inference,
            inputs=[image, video, webcam, model_path, image_size, conf_threshold, input_type],
            outputs=[output_image, output_video, detection_info],
        )

        # Example images for road cracks detection
        test_images_dir = "roadCracksDataset/test/images"
        example_images = []
        
        # Check if test images directory exists and add some example images
        if os.path.exists(test_images_dir):
            image_files = os.listdir(test_images_dir)[:3]  # Get first 3 test images
            for img_file in image_files:
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    example_images.append([
                        os.path.join(test_images_dir, img_file),
                        model_path.value,
                        640,
                        0.25,
                    ])
        
        # If no test images found, don't use examples
        if not example_images:
            print("No example images found for road cracks detection")
        
        if example_images:
            try:
                gr.Examples(
                    examples=example_images,
                    fn=road_cracks_inference_for_examples,
                    inputs=[
                        image,
                        model_path,
                        image_size,
                        conf_threshold,
                    ],
                    outputs=[output_image, detection_info],
                    cache_examples='lazy',
                )
            except Exception as e:
                print(f"Error setting up road cracks examples: {e}")


gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    Road Cracks Detection & General Object Detection
    </h1>
    """)
    gr.HTML(
        """
        <h3 style='text-align: center'>
        <a href='https://arxiv.org/abs/2502.12524' target='_blank'>arXiv</a> | <a href='https://github.com/sunsmarterjie/yolov12' target='_blank'>github</a>
        </h3>
        """)
    
    with gr.Tabs():
        with gr.TabItem("Road Cracks Detection"):
            road_cracks_detection_app()
        with gr.TabItem("General Object Detection"):
            general_detection_app()
            
if __name__ == '__main__':
    gradio_app.launch()
