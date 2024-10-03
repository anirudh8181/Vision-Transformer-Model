import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math
from tensorflow import keras
from keras import layers
from keras.layers import Softmax

num_classes=10
input_shape= (32,32,3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

print(f"x_train shape {x_train.shape}- y_train shape: {y_train.shape}")
print(f"x_test shape {x_test.shape}- y_test shape: {y_test.shape}")

x_train = x_train[:500]

y_train = y_train[:500]

x_test = x_test[:500]

y_test = y_test[:500]

#DECLARING PARAMETERS FOR MODEL
learning_rate = 0.001
weight_decay= 0.0001
batch_size = 256
num_epochs = 40
image_size = 72
patch_size = 6
num_patches = (image_size // patch_size)**2 
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim*2,
    projection_dim
    ]
transformer_layers = 8
mlp_head_units = [2048, 1024]

#DATA AUGUMENTATION FOR ROBUSTNESS
data_augmentation = keras.Sequential(
[

    layers.Normalization(),
    layers.Resizing(image_size, image_size),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(factor = 0.02),
    layers.RandomZoom(
        height_factor = 0.2, width_factor = 0.2)
  ],
  name="data_augementation"        
 )
data_augmentation.layers[0].adapt(x_train)

#DEFINE MLP HEAD
def mlp(x,  hidden_units, dropout_rate):
    for units in hidden_units:
       x=layers.Dense(units, activation= tf.nn.gelu)(x)
       x=layers.Dropout (dropout_rate)(x) 
    return x
   
#PATCHES
# Define patches layer
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [-1, patch_dims])  # Remove batch dimension
        return patches

# Create an instance of the patches layer
plt.figure(figsize=(4,4))    
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.imshow(image.astype("uint8"))
plt.axis("off")

resized_image = tf.image.resize(
     tf.convert_to_tensor([image]), size=(image_size, image_size)

)
patches = Patches(patch_size)(resized_image)

print(f"image size: {image_size} x {image_size}")
print(f"image size: {patch_size} x {patch_size}")
print(f"patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")
print(f"Shape of patches: {patches.shape}")

# Calculate the number of subplots
# Calculate the number of subplots
# Calculate the number of subplots
num_patches = patches.shape[0]
num_cols = min(num_patches, 10)  # Limit the number of columns to 10
num_rows = int(np.ceil(num_patches / num_cols))

plt.figure(figsize=(4*num_cols, 4*num_rows))

for i, patch in enumerate(patches):
    ax = plt.subplot(num_rows, num_cols, i + 1) 
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype("uint8")) 
    plt.axis("off")

plt.show()

#ENCODER
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches =  num_patches
        self.projection = layers.Dense(units=projection_dim) 
        self.position_embedding  = layers.Embedding(
            input_dim = num_patches, output_dim = projection_dim
            )

    def call(self, patch):
         positions =  tf.range(start=0, limit = self.num_patches, delta = 1)
         encoded =  self.projection(patch) + self.position_embedding(positions)
         return encoded
    

class ExpandDimsLayer(layers.Layer):
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=1)

    
softmax_layer = Softmax(axis=-1)

def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    
    # Augment data.
    augmented = data_augmentation(inputs)
    
    # Create patches.
    patches = Patches(patch_size)(augmented)
    
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    
    # Classify outputs.
    logits = layers.Dense(num_classes)(representation)

    # Apply softmax activation
    probabilities = softmax_layer(logits)
    
    # Create the Keras Model with both input and output
    model = keras.Model(inputs=inputs, outputs=probabilities)
    
    return model


def run_experiment(model):
    optimizer = tf.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top_5_accuracy"),
        ],
    )

    checkpoint_filepath = "/tmp/checkpoint.weights.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath, 
        monitor="val_accuracy",  # Corrected typo
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback]
    )

    model.load_weights(checkpoint_filepath)

    # Evaluate the model
    loss, accuracy, top5_accuracy = model.evaluate(x_test, y_test)
    print(f'Rest accuracy: {round(accuracy * 100, 2)}%')
    print(f'Test top 5 accuracy: {round(top5_accuracy * 100, 2)}%')


vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)    


class_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

def img_predict(images, model):
    if len(images.shape) == 3:
        out = model.predict(images.reshape(-1, *images.shape))
    else:
        out = model.predict(images)
    prediction = np.argmax(out, axis=1)
    img_prediction = [class_names[i] for i in prediction]
    return img_prediction

index = 16
plt.imshow(x_test[index])
prediction = img_predict(x_test[index], vit_classifier)
print(prediction)
