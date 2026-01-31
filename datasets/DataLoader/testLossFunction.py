import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

def visualize_heatmaps(heatmaps, window_size=(200, 200), screen_size=(1920, 1080), margin=10):
    """
    Visualizes concatenated heatmaps in OpenCV windows, arranging them to fit the screen.

    Args:
        heatmaps (np.ndarray): Concatenated heatmaps of shape (height, width, num_heatmaps).
        window_size (tuple): Size of each window in pixels (width, height).
        screen_size (tuple): Screen resolution in pixels (width, height).
        margin (int): Margin between windows in pixels.
    """
    num_heatmaps = heatmaps.shape[-1]
    rows = int(np.sqrt(num_heatmaps))
    cols = (num_heatmaps + rows - 1) // rows  # Ensure all heatmaps are displayed

    for i in range(num_heatmaps):
        heatmap = heatmaps[..., i]
        
        heatmap = np.squeeze(heatmap)
        # Debug: Print shape and type of each heatmap
        #print(f"Heatmap {i} shape: {heatmap.shape}, dtype: {heatmap.dtype}")
        
        # Ensure the heatmap is a valid 2D array
        if len(heatmap.shape) != 2:
            #print(f"Skipping Heatmap {i} due to invalid shape: {heatmap.shape}")
            continue
        
        # Convert heatmap to a valid format if necessary
        if not isinstance(heatmap, np.ndarray):
            heatmap = np.array(heatmap)
        
        if heatmap.dtype != np.uint8:
            heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap = heatmap.astype(np.uint8)
        
        # Resize the heatmap to the specified window size
        resized_heatmap = cv2.resize(heatmap, window_size)

        # Calculate the position of the window on the screen
        row = i // cols
        col = i % cols
        x_pos = col * (window_size[0] + margin)
        y_pos = row * (window_size[1] + margin)

        # Ensure the windows fit within the screen bounds
        if x_pos + window_size[0] > screen_size[0] or y_pos + window_size[1] > screen_size[1]:
            #print(f"Window {i} would be out of screen bounds, adjusting position.")
            x_pos = min(x_pos, screen_size[0] - window_size[0])
            y_pos = min(y_pos, screen_size[1] - window_size[1])

        # Display the heatmap in an OpenCV window
        window_name = f'Heatmap {i}'
        cv2.imshow(window_name, resized_heatmap)
        cv2.moveWindow(window_name, x_pos, y_pos)

    # Wait until the user presses any key to close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#Use if DataLoader C is configured with VALID_SEGMENTATIONS > 1
class VanillaMSELossSimple(keras.losses.Loss):
    def __init__(self, weight=1.0, scale=1.0, **kwargs):
        super(VanillaMSELossSimple, self).__init__(**kwargs)
        self.weight = weight
        self.scale  = scale

    def call(self, y_true, y_pred):
        # Ensure both y_true and y_pred are cast to float32
        float_type = keras.backend.floatx()
        y_true = tf.cast(y_true, float_type)
        y_pred = tf.cast(y_pred, float_type)

        # Compute the squared difference
        squared_difference = tf.square( (y_true - y_pred) * self.scale)

        # Compute the mean over all elements
        mse_loss = tf.reduce_mean(squared_difference)

        return mse_loss * self.weight


# Define the VanillaMSELoss class
class VanillaMSELoss(keras.losses.Loss):
    def __init__(self, weight=1.0, scale=1.0, num_instances=64, num_classes=182, **kwargs):
        super(VanillaMSELoss, self).__init__(**kwargs)
        self.weight            = weight
        self.scale             = scale
        self.num_instances     = num_instances  # Number of segmentation categories, we try to make the NN life easier by reducing them
        self.num_classes       = num_classes    # Number of segmentation categories, now 182 including background
        self.scaling_factor    = 120.0          # Scaling factor for combined heatmaps
        self.segmentation_gain = 2.0
        self.instance_gain     = 2.0

    def save_one_hot_data(self, one_hot_data, prefix):
        """
        Save one-hot encoded data as PNG files.

        Args:
            one_hot_data (tf.Tensor): One-hot encoded data to be saved.
            prefix (str): Prefix for filenames.
        """
        # Convert tensor to numpy array
        one_hot_data_np = one_hot_data.numpy()

        # Get the shape of the data
        height, width, num_classes = one_hot_data_np.shape[1:4]

        # Iterate through each class and save it as an individual image
        for i in range(num_classes):
            # Extract the ith class channel
            class_channel = one_hot_data_np[0, :, :, i]  # Assuming batch size of 1

            # Convert to a valid format if necessary
            if class_channel.dtype != np.uint8:
                class_channel = cv2.normalize(class_channel, None, 0, 255, cv2.NORM_MINMAX)
                class_channel = class_channel.astype(np.uint8)

            # Save as PNG
            filename = f'{prefix}_class{i}.png'
            cv2.imwrite(filename, class_channel)
            print(f"Saved one-hot encoded data for class {i} to {filename}")

    def save_heatmap_data(self, heatmap_data, prefix):
        """
        Save heatmap data as PNG files.

        Args:
            heatmap_data (tf.Tensor): Heatmap data to be saved.
            prefix (str): Prefix for filenames.
        """
        # Convert tensor to numpy array
        heatmap_data_np = heatmap_data.numpy()

        # Normalize heatmap to range [0, 255]
        heatmap_data_np = cv2.normalize(heatmap_data_np, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_data_np = heatmap_data_np.astype(np.uint8)

        # Save as PNG
        filename = f'{prefix}.png'
        cv2.imwrite(filename, heatmap_data_np)
        print(f"Saved heatmap data to {filename}")

    def call(self, y_true, y_pred):
        # Ensure both y_true and y_pred are cast to float32
        float_type = keras.backend.floatx()
        #Regular loss
        # Calculate MSE for heatmaps 0-34 (except segmentation masks)
        #----------------------------------------------------------------------------
        y_true_first = tf.cast(y_true[..., :34], float_type)
        y_pred_first = tf.cast(y_pred[..., :34], float_type)

        mse_loss = tf.reduce_mean(tf.square((y_true_first - y_pred_first) * self.scale))
        #----------------------------------------------------------------------------

        #Instance 
        #----------------------------------------------------------------------------
        # Extract the combined heatmap (heatmaps 35-52 combined into one)
        y_true_instance = tf.cast(y_true[..., 34], tf.int32)
        y_pred_instance = tf.cast(y_pred[..., 34], tf.int32)

        # Shift the values to make them suitable for one-hot encoding
        y_true_instance_indices = 120 - y_true_instance# <- these need to be flipped because of the recounting in rearrangeInstanceCount in DataLoaderC/processing/resize.h
        y_pred_instance_indices = 120 - y_pred_instance# <- these need to be flipped because of the recounting in rearrangeInstanceCount in DataLoaderC/processing/resize.h

        # One-hot encode the combined heatmaps for all classes (including background)
        y_true_instance_one_hot = tf.one_hot(y_true_instance_indices, depth=self.num_instances, dtype=float_type)
        y_pred_instance_one_hot = tf.one_hot(y_pred_instance_indices, depth=self.num_instances, dtype=float_type)

        # Compute MSE for each class in a vectorized manner
        instance_mse_loss = tf.reduce_mean( tf.square((y_true_instance_one_hot - y_pred_instance_one_hot) * (self.scaling_factor * self.scale)), axis=[0, 1, 2] )

        # Average the loss across all classes
        instance_loss = tf.reduce_mean(instance_mse_loss)
        #----------------------------------------------------------------------------

        # Save heatmap 34 data
        self.save_heatmap_data(tf.squeeze(y_true_instance_indices), 'heatmap_y_true_34')
        self.save_heatmap_data(tf.squeeze(y_pred_instance_indices), 'heatmap_y_pred_34')

        # Save one-hot encoded arrays
        self.save_one_hot_data(y_true_instance_one_hot, 'one_hot_y_true_34')
        self.save_one_hot_data(y_pred_instance_one_hot, 'one_hot_y_pred_34')

        #Segmentation 
        #----------------------------------------------------------------------------
        # Extract the combined heatmap (heatmaps 35-52 combined into one)
        y_true_combined = tf.cast(y_true[..., 35], tf.int32)
        y_pred_combined = tf.cast(y_pred[..., 35], tf.int32)

        # Shift the values to make them suitable for one-hot encoding
        y_true_indices = y_true_combined + 120
        y_pred_indices = y_pred_combined + 120

        # One-hot encode the combined heatmaps for all classes (including background)
        y_true_one_hot = tf.one_hot(y_true_indices, depth=self.num_classes, dtype=float_type)
        y_pred_one_hot = tf.one_hot(y_pred_indices, depth=self.num_classes, dtype=float_type)

        # Compute MSE for each class in a vectorized manner
        class_mse_loss = tf.reduce_mean( tf.square((y_true_one_hot - y_pred_one_hot) * (self.scaling_factor * self.scale)), axis=[0, 1, 2] )

        # Average the loss across all classes
        segmentation_loss = tf.reduce_mean(class_mse_loss)
        #----------------------------------------------------------------------------

        # Save heatmap 35 data
        self.save_heatmap_data(tf.squeeze(y_true_indices), 'heatmap_y_true_35')
        self.save_heatmap_data(tf.squeeze(y_pred_indices), 'heatmap_y_pred_35')

        # Save one-hot encoded arrays
        self.save_one_hot_data(y_true_one_hot, 'one_hot_y_true_35')
        self.save_one_hot_data(y_pred_one_hot, 'one_hot_y_pred_35')

        # Sum the losses
        #----------------------------------------------------------------------------
        total_loss = mse_loss + (segmentation_loss * self.segmentation_gain) + (instance_loss * self.instance_gain)

        return total_loss * self.weight

# Function to load and stack heatmaps
def load_and_stack_heatmaps(base_dir, sampleNumber=0, num_heatmaps=36):
    heatmaps = []
    for i in range(num_heatmaps):
        filename = f'sample{sampleNumber}_C8bit_hm{i}.png'
        file_path = os.path.join(base_dir, filename)
        if os.path.exists(file_path):
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            heatmaps.append(img)
        else:
            print(f"Missing file: {file_path}")
            return None
    stacked_heatmaps = np.stack(heatmaps, axis=-1)  # Stack along the channel dimension
    stacked_heatmaps = stacked_heatmaps.astype(np.int32) - 120  # Subtract 120 from all values
    #stacked_heatmaps = stacked_heatmaps - 120  # Subtract 120 from all values
    return stacked_heatmaps

# Main function to compute the loss
def compute_loss(groundtruth_dir, prediction_dir, sampleNumber=0):
    total_loss = 0
    loss_fn = VanillaMSELoss()
    #loss_fn = VanillaMSELossSimple()

    y_true = load_and_stack_heatmaps(groundtruth_dir,sampleNumber=sampleNumber)
    y_pred = load_and_stack_heatmaps(prediction_dir,sampleNumber=sampleNumber)

    if y_true is not None and y_pred is not None:
        y_true = np.expand_dims(y_true, axis=0)  # Add batch dimension
        y_pred = np.expand_dims(y_pred, axis=0)  # Add batch dimension

        y_true = tf.convert_to_tensor(y_true)#, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred)#, dtype=tf.float32)

        loss = loss_fn(y_true, y_pred)
        total_loss += loss.numpy()

    print(f"Total Loss: {total_loss}")
    visualize_heatmaps(y_pred)


    return total_loss

# Example usage
sampleNumber= 0
groundtruth_dir = 'groundtruth'
prediction_dir  = 'prediction'
os.system("mkdir %s" % groundtruth_dir)
os.system("mkdir %s" % prediction_dir)
os.system("rm *.png *.jpg")

loss = compute_loss(groundtruth_dir, prediction_dir, sampleNumber=sampleNumber)
print(f"Total Loss: {loss}")
