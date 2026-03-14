import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import utils as ut

test_mode = True


ut.clear_console()

# *** load the data from a built in dataset ***
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if test_mode:
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    print(f"Image shape: {x_train[0].shape}")



#converts every pixel to a floating-point number between:0.1 - 1.0
#Neural networks train better with small, normalized values
x_train = x_train / 255.0
x_test = x_test / 255.0


# CNNs expect a channel dimension
# Reshape from (60000, 28, 28) to (60000, 28, 28, 1)
# The "1" is for grayscale. RGB images would be "3"
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

if test_mode:
    ut.clear_console()
    print("After reshaping for CNN input:")
    print(f"New shape: {x_train.shape}")

# *** build the model ***
# relu = rectified linear unit activation function
model = tf.keras.Sequential([
    # --- Convolutional Block 1 ---
    # Conv2D: slides 32 small 3x3 filters across the image
    # Each filter learns to detect a specific feature (edges, curves, etc.)
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    
    # MaxPooling: shrinks the image by taking the max value in each 2x2 region
    # 28x28 → 13x13 (after conv) → 6x6 (after pool)
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # --- Convolutional Block 2 ---
    # More filters to detect more complex patterns
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # --- Convolutional Block 3 ---
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    
    # --- Classification Head ---
    # Flatten: convert 3D feature maps to 1D vector
    tf.keras.layers.Flatten(),
    
    # Dense layers for final classification
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    
    # Output: 10 classes (digits 0-9)
    tf.keras.layers.Dense(10, activation='softmax')
])
if test_mode:
    ut.clear_console()
    print("Model architecture:")    
    model.summary() # print model architecture



# --- compile the model ---
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) 


verb = 0
if test_mode:
    ut.clear_console()
    print("Model compiled.")
    verb = 1

# --- train the model ---
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1, verbose=verb)
# validation_split=0.1 means 10% of training data is used for validation
#verbose=1 means print progress during training


# --- evaluate the model ---
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
#print('\nTest accuracy:', test_acc) 
print(f"\nTest accuracy: {test_acc * 100:.2f}%")