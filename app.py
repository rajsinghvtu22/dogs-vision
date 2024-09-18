import gradio as gr
import tensorflow as tf 

# Load model
model_save_path = "dog_vision_model_demo.keras"
loaded_model_for_demo = tf.keras.models.load_model(model_save_path)

# Load labels
with open("stanford_dogs_class_names.txt", "r") as f:
  class_names = [line.strip() for line in f.readlines()]

# Create prediction function
def pred_on_custom_image(image, # input image (preprocessed by Gradio's Image input to be numpy.array)
                         model: tf.keras.Model =loaded_model_for_demo,  # Trained TensorFlow model for prediction
                         target_size: int = 224,  # Desired size of the image for input to the model
                         class_names: list = class_names): # List of class names
  """
  Loads an image, preprocesses it, makes a prediction using a provided model,
  and returns a dictionary of prediction probabilities per class name.

  Args:
      image: Input image.
      model: Trained TensorFlow model for prediction.
      target_size (int, optional): Desired size of the image for input to the model. Defaults to 224.
      class_names (list, optional): List of class names for plotting. Defaults to None.

  Returns:
     Dict[str: float]: A dictionary of string class names and their respective prediction probability.
  """

  # Note: gradio.inputs.Image handles opening the image
  # # Prepare and load image
  # custom_image = tf.keras.utils.load_img(
  #   path=image_path,
  #   color_mode="rgb",
  #   target_size=target_size,
  # )

  # Create resizing layer to resize the image
  resize = tf.keras.layers.Resizing(height=target_size,
                                    width=target_size)

  # Turn the image into a tensor and resize it
  custom_image_tensor = resize(tf.keras.utils.img_to_array(image))

  # Add a batch dimension to the target tensor (e.g. (224, 224, 3) -> (1, 224, 224, 3))
  custom_image_tensor = tf.expand_dims(custom_image_tensor, axis=0)

  # Make a prediction with the target model
  pred_probs = model.predict(custom_image_tensor)[0]

  # Predictions get returned as a dictionary of {label: pred_prob}
  pred_probs_dict = {class_names[i]: float(pred_probs[i]) for i in range(len(class_names))}

  return pred_probs_dict

# Create Gradio interface
interface_title = "Dog Vision 🐶👁️"
interface_description = """
Identify different dogs in images with deep learning. Model trained with TensorFlow/Keras.

## Links

* Original dataset: http://vision.stanford.edu/aditya86/ImageNetDogs/ 
* Code to train model: https://dev.mrdbourke.com/zero-to-mastery-ml/end-to-end-dog-vision-v2/
"""
interface = gr.Interface(fn=pred_on_custom_image,
                         inputs=gr.Image(),
                         outputs=gr.Label(num_top_classes=3),
                         examples=["dog-photo-1.jpeg", 
                                    "dog-photo-2.jpeg",
                                    "dog-photo-3.jpeg",
                                    "dog-photo-4.jpeg"],
                         title=interface_title,
                         description=interface_description)
interface.launch(debug=True)
