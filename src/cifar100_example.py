import tensorflow as tf
print(tf.__version__)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

import matplotlib.pyplot as plt

plt.imshow(x_train[0])
print(x_train[0])
print(y_train[0])


print("Image batch shape: ", x_train.shape)
print("Label batch shape: ", y_train.shape)

#x_norm_train = [tf.image.per_image_standardization(frame) for frame in x_train ]

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(100, activation=tf.nn.softmax)])
                                    
                        
model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1)

model.evaluate(x_test, y_test)

