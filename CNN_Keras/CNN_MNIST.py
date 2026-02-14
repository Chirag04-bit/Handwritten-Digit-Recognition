import numpy as np
import argparse
import cv2
from cnn.neural_network import CNN
from keras.utils import to_categorical
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

# -------------------- Parse Arguments --------------------
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save_model", type=int, default=-1, help="Flag to save the model")
ap.add_argument("-l", "--load_model", type=int, default=-1, help="Flag to load pre-trained model")
ap.add_argument("-w", "--save_weights", type=str, help="Path to save/load model weights")
args = vars(ap.parse_args())

# -------------------- Load MNIST Dataset --------------------
print("Loading MNIST Dataset...")
dataset = fetch_openml('mnist_784', version=1)

# Reshape data: 70000 samples, 28x28 images, 1 channel
mnist_data = dataset.data.values.reshape((-1, 28, 28, 1))
mnist_data = mnist_data.astype("float32") / 255.0  # Normalize pixel values

# Convert labels to integers
labels = dataset.target.astype("int")

# Split into training and testing sets
train_img, test_img, train_labels, test_labels = train_test_split(
    mnist_data, labels, test_size=0.1, random_state=42
)

# -------------------- One-Hot Encoding --------------------
total_classes = 10  # digits 0-9
train_labels = to_categorical(train_labels, total_classes)
test_labels = to_categorical(test_labels, total_classes)

# -------------------- Build and Compile Model --------------------
print("\nCompiling model...")
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
clf = CNN.build(
    width=28, height=28, depth=1, total_classes=total_classes,
    Saved_Weights_Path=args["save_weights"] if args["load_model"] > 0 else None
)
clf.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# -------------------- Train Model --------------------
batch_size = 128
num_epoch = 20
verbose = 1

if args["load_model"] < 0:
    print("\nTraining the model...")
    clf.fit(train_img, train_labels, batch_size=batch_size, epochs=num_epoch, verbose=verbose)

    print("\nEvaluating accuracy on test data...")
    loss, accuracy = clf.evaluate(test_img, test_labels, batch_size=batch_size, verbose=1)
    print("Test Accuracy: {:.2f}%".format(accuracy * 100))

# -------------------- Save Weights --------------------
if args["save_model"] > 0 and args["save_weights"] is not None:
    print("\nSaving weights to file...")
    clf.save_weights(args["save_weights"], overwrite=True)

# -------------------- Visualize Some Predictions --------------------
for idx in np.random.choice(np.arange(0, len(test_labels)), size=5, replace=False):
    probs = clf.predict(test_img[np.newaxis, idx])
    prediction = probs.argmax(axis=1)

    # Resize image for display
    image = (test_img[idx] * 255).astype("uint8")
    image = cv2.merge([image[:, :, 0]] * 3)
    image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, str(prediction[0]), (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    print(f"Predicted Label: {prediction[0]}, Actual Label: {np.argmax(test_labels[idx])}")
    # Uncomment below to see images
    cv2.imshow("Digit", image)
    cv2.waitKey(0)

# cv2.destroyAllWindows()
