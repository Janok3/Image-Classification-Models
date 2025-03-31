import os
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import json

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Scikit-Learn
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support, 
                             log_loss,precision_score, recall_score, f1_score)

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#Nengo
import nengo
import nengo_dl


base_dir = "dataset"
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def extract_features_from_images(image_array):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

    features = []
    for img in image_array:
        x = np.expand_dims(img, axis=0)
        feat = feature_extractor.predict(x).flatten()
        features.append(feat)
    return np.array(features)

def calculate_kmeans_loss(X_feats, kmeans):
    # Get the cluster assignments for the features
    cluster_assignments = kmeans.predict(X_feats)
    # Calculate the loss as the sum of squared distances to the cluster centers
    loss = sum(np.linalg.norm(X_feats[i] - kmeans.cluster_centers_[cluster_assignments[i]])**2 for i in range(len(X_feats)))
    return loss

# =========================================================
# MODEL TRAINING FUNCTIONS
# =========================================================
def run_CNN_Epochs(num_epochs):
    print("\n=== CNN (Epoch-based) ===")
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=True
    )
    val_gen = test_datagen.flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=False
    )

    base = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base.trainable = False

    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Disabled for plotting purposes
    # early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(patience=3, factor=0.1, verbose=1)

    start_time = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=num_epochs,
        callbacks=[reduce_lr],
        verbose=1
    )
    runtime = time.time() - start_time

    # Collect metrics
    val_acc_list = history.history['val_accuracy']
    train_acc_list = history.history['accuracy']
    train_loss_list = history.history['loss']
    val_loss_list = history.history['val_loss']

    y_true = val_gen.classes
    y_pred_probs = model.predict(val_gen)
    y_pred = (y_pred_probs >= 0.5).astype(int).ravel()

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    metrics_dict = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "train_acc_list": train_acc_list,
        "val_acc_list": val_acc_list,
        "train_loss_list": train_loss_list,
        "val_loss_list": val_loss_list,
        "runtime": runtime
    }
    return metrics_dict

def run_LogReg_Epochs(num_epochs):
    print("\n=== Logistic Regression (Epoch-based) ===")
    BATCH_SIZE = 32
    IMG_SIZE = (128,128)

    train_datagen = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
        shear_range=0.1, zoom_range=0.1, horizontal_flip=True,
        rescale=1./255
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=False
    )

    train_acc_list = []
    val_acc_list = []
    start_time = time.time()

    from sklearn.linear_model import SGDClassifier
    classifier = SGDClassifier(loss='log', penalty='l2', learning_rate='constant', eta0=0.001)
    all_classes = np.array([0,1], dtype=np.uint8)

    # Gather test data once
    X_val, y_val = [], []
    for _ in range(len(val_gen)):
        Xb, yb = next(val_gen)
        X_val.append(Xb)
        y_val.append(yb)
    X_val = np.concatenate(X_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)
    X_val_feats = extract_features_from_images(X_val)

    train_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        steps_per_epoch = len(train_gen)
        X_train_batch = []
        y_train_batch = []
        for _ in range(steps_per_epoch):
            xb, yb = next(train_gen)
            X_train_batch.append(xb)
            y_train_batch.append(yb)
        X_train_batch = np.concatenate(X_train_batch, axis=0)
        y_train_batch = np.concatenate(y_train_batch, axis=0)
        X_train_feats = extract_features_from_images(X_train_batch)

        classifier.partial_fit(X_train_feats, y_train_batch, classes=all_classes)

        y_train_pred = classifier.predict(X_train_feats)
        train_acc = accuracy_score(y_train_batch, y_train_pred)
        y_val_pred = classifier.predict(X_val_feats)
        val_acc = accuracy_score(y_val, y_val_pred)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")

        # Calculate loss for training
        train_loss = log_loss(y_train_batch, classifier.predict_proba(X_train_feats))
        train_loss_list.append(train_loss)

        # Calculate loss for validation
        val_loss = log_loss(y_val, classifier.predict_proba(X_val_feats))
        val_loss_list.append(val_loss)

    runtime = time.time() - start_time
    y_val_pred = classifier.predict(X_val_feats)
    final_acc = accuracy_score(y_val, y_val_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_val_pred, average='binary')

    metrics_dict = {
        "accuracy": final_acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "train_acc_list": train_acc_list,
        "val_acc_list": val_acc_list,
        "train_loss_list": train_loss_list,
        "val_loss_list": val_loss_list,
        "runtime": runtime
    }
    return metrics_dict

def run_KMeans_Epochs(num_epochs):
    print("\n=== K-Means (Epoch-based) ===")
    BATCH_SIZE = 32
    IMG_SIZE = (128,128)

    train_datagen = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
        shear_range=0.1, zoom_range=0.1,
        horizontal_flip=True, fill_mode='nearest', rescale=1./255
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=False
    )

    X_val_list, y_val_list = [], []
    for _ in range(len(val_gen)):
        Xb, yb = next(val_gen)
        X_val_list.append(Xb)
        y_val_list.append(yb)
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    X_val_feats = extract_features_from_images(X_val)

    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=2, random_state=42, batch_size=64)

    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []
    start_time = time.time()

    for epoch in range(num_epochs):
        steps_per_epoch = len(train_gen)
        X_train_list = []
        y_train_list = []
        for _ in range(steps_per_epoch):
            xb, yb = next(train_gen)
            X_train_list.append(xb)
            y_train_list.append(yb)
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        X_train_feats = extract_features_from_images(X_train)

        BATCH_SIZE_KM = 64
        idx = 0
        while idx < X_train_feats.shape[0]:
            batch_feats = X_train_feats[idx:idx+BATCH_SIZE_KM]
            kmeans.partial_fit(batch_feats)
            idx += BATCH_SIZE_KM

        train_clusters = kmeans.predict(X_train_feats)
        cluster_to_class = {}
        for c in range(2):
            labels_in_cluster = y_train[train_clusters == c]
            if len(labels_in_cluster) == 0:
                cluster_to_class[c] = 0
            else:
                cluster_to_class[c] = Counter(labels_in_cluster).most_common(1)[0][0]
        y_train_pred = np.array([cluster_to_class[c] for c in train_clusters])
        train_acc = accuracy_score(y_train, y_train_pred)

        val_clusters = kmeans.predict(X_val_feats)
        y_val_pred = np.array([cluster_to_class[c] for c in val_clusters])
        val_acc = accuracy_score(y_val, y_val_pred)

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        train_loss = calculate_kmeans_loss(X_train_feats, kmeans)
        train_loss_list.append(train_loss)

        # Calculate loss for validation
        val_loss = calculate_kmeans_loss(X_val_feats, kmeans)
        val_loss_list.append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")

    runtime = time.time() - start_time
    final_acc = val_acc  # from last epoch
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_val_pred, average='binary')

    metrics_dict = {
        "accuracy": final_acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "train_acc_list": train_acc_list,
        "val_acc_list": val_acc_list,
        "train_loss_list": train_loss_list,
        "val_loss_list": val_loss_list,
        "runtime": runtime
    }
    return metrics_dict

def run_GAN_Epochs(num_epochs):
    print("\n=== GAN Discriminator (Epoch-based) ===")
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cpu")
    img_size = 64
    batch_size = 64

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(test_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    class Discriminator(nn.Module):
        def __init__(self, img_channels=3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(img_channels, 64, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 512, 4, 2, 1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Flatten()
            )
            self.classifier = nn.Sequential(
                nn.Linear(512*4*4, 1),
            )
        def forward(self, x):
            feats = self.net(x)
            return self.classifier(feats)

    model = Discriminator().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999))

    train_acc_list = []
    val_acc_list = []
    start_time = time.time()

    def evaluate(loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                preds = (out>0).float()
                correct += (preds.eq(yb.float().unsqueeze(1))).sum().item()
                total += xb.size(0)
        return correct/total if total > 0 else 0

    train_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            targets = yb.float().unsqueeze(1)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = (out>0).float()
            correct_train += (preds.eq(targets)).sum().item()
            total_train += xb.size(0)

        train_acc = correct_train/total_train
        val_acc = evaluate(val_loader)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")

        # Calculate loss for training
        train_loss = running_loss/total_train
        train_loss_list.append(train_loss)

        # Calculate loss for validation
        val_loss = evaluate(val_loader)
        val_loss_list.append(val_loss)

    runtime = time.time() - start_time

    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = (logits>0).float().cpu().numpy().ravel()
            y_pred_all.extend(preds)
            y_true_all.extend(yb.cpu().numpy())
    final_acc = accuracy_score(y_true_all, y_pred_all)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average='binary')

    metrics_dict = {
        "accuracy": final_acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "train_acc_list": train_acc_list,
        "val_acc_list": val_acc_list,
        "train_loss_list": train_loss_list,
        "val_loss_list": val_loss_list,
        "runtime": runtime
    }
    return metrics_dict

def run_SNN_Epochs(num_epochs):
    print("\n=== SNN (Epoch-based) ===")
    
    # Load and preprocess custom images using tf.keras.preprocessing
    batch_size = 32
    img_height = 32  # CIFAR-10 uses 32x32 images, adjust as needed
    img_width = 32

    # Load training data with validation split
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical'
    )

    # Load test data directly without splitting
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical'
    )

    # Get class names from the dataset
    class_names = train_ds.class_names

    # Convert dataset to numpy arrays
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for images, labels in train_ds:
        x_train.append(images.numpy())
        y_train.append(labels.numpy())

    for images, labels in test_ds:
        x_test.append(images.numpy())
        y_test.append(labels.numpy())

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    def get_2d_cnn_model(inpt_shape):
        def _get_cnn_block(layer, num_filters, layer_objs_lst):
            conv = tf.keras.layers.Conv2D(
                num_filters, 3, padding="same", activation="relu",
                kernel_initializer="he_uniform",
                kernel_regularizer=tf.keras.regularizers.l2(0.005))(layer)
            avg_pool = tf.keras.layers.AveragePooling2D()(conv)

            layer_objs_lst.append(conv)
            return avg_pool

        layer_objs_lst = []  # To store the layer objects to probe later in Nengo-DL

        inpt_layer = tf.keras.Input(shape=inpt_shape)
        layer_objs_lst.append(inpt_layer)

        layer = _get_cnn_block(inpt_layer, 32, layer_objs_lst)
        layer = _get_cnn_block(layer, 64, layer_objs_lst)
        layer = _get_cnn_block(layer, 128, layer_objs_lst)

        flat = tf.keras.layers.Flatten()(layer)

        dense = tf.keras.layers.Dense(
            512, activation="relu", kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(0.005))(flat)
        layer_objs_lst.append(dense)

        output_layer = tf.keras.layers.Dense(
            len(class_names), activation="softmax", kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(0.005))(dense)
        layer_objs_lst.append(output_layer)

        model = tf.keras.Model(inputs=inpt_layer, outputs=output_layer)
        return model, layer_objs_lst

    model, layer_objs_lst = get_2d_cnn_model((32, 32, 3))
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Initialize lists for metrics
    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []

    # Train the TF network and collect metrics
    start_time = time.time()
    history = model.fit(train_ds, validation_data=test_ds, epochs=num_epochs)

    # Store training and validation metrics
    train_acc_list = history.history['accuracy']
    val_acc_list = history.history['val_accuracy']
    train_loss_list = history.history['loss']
    val_loss_list = history.history['val_loss']
    runtime = time.time() - start_time

    # Evaluate using test_ds
    results = model.evaluate(test_ds)

    # Get the spiking network.
    sfr = 20
    ndl_model = nengo_dl.Converter(
        model,
        swap_activations={
            tf.keras.activations.relu: nengo.SpikingRectifiedLinear()},
        scale_firing_rates=sfr,
        synapse=0.005,
        inference_only=True)

    def get_nengo_compatible_test_data_generator(test_dataset, batch_size=32, n_steps=30):
        x_test_list = []
        y_test_list = []
        
        # Convert dataset to numpy arrays
        for images, labels in test_dataset:
            x_test_list.append(images.numpy())
            y_test_list.append(labels.numpy())
        
        x_test = np.concatenate(x_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        
        num_images = x_test.shape[0]
        # Calculate the actual batch size for the last batch
        last_batch_size = num_images % batch_size
        if last_batch_size == 0:
            last_batch_size = batch_size
        
        # Flatten the images
        reshaped_x_test = x_test.reshape((num_images, 1, -1))
        # Tile/Repeat them for `n_steps` times
        tiled_x_test = np.tile(reshaped_x_test, (1, n_steps, 1))

        # Generate complete batches
        for i in range(0, num_images - last_batch_size, batch_size):
            yield (tiled_x_test[i:i+batch_size], y_test[i:i+batch_size])
        
        # Handle the last batch separately
        if last_batch_size > 0:
            # Pad the last batch to match the batch size
            last_batch_x = tiled_x_test[-last_batch_size:]
            last_batch_y = y_test[-last_batch_size:]
            pad_size = batch_size - last_batch_size
            
            # Repeat the last samples to fill the batch
            padded_x = np.pad(last_batch_x, ((0, pad_size), (0, 0), (0, 0)), mode='edge')
            padded_y = np.pad(last_batch_y, ((0, pad_size), (0, 0)), mode='edge')
            
            yield (padded_x, padded_y)

    # Get the probes for Input, first Conv, and the Output layers.
    ndl_mdl_inpt = ndl_model.inputs[layer_objs_lst[0]]  # Input layer is Layer 0.
    ndl_mdl_otpt = ndl_model.outputs[layer_objs_lst[-1]]  # Output layer is last.
    with ndl_model.net:
        nengo_dl.configure_settings(stateful=False)  # Optimize simulation speed.
        # Probe for the first Conv layer.
        first_conv_probe = nengo.Probe(ndl_model.layers[layer_objs_lst[1]])
        # Probe for penultimate dense layer.
        penltmt_dense_probe = nengo.Probe(ndl_model.layers[layer_objs_lst[-2]])

    n_steps = 30
    batch_size = 32  # Match your original batch_size
    collect_spikes_output = True
    ndl_mdl_otpt_cls_probs = []

    test_batches = get_nengo_compatible_test_data_generator(
        test_dataset=test_ds,
        batch_size=batch_size,
        n_steps=n_steps
    )

    # Run the simulation.
    with nengo_dl.Simulator(ndl_model.net, minibatch_size=batch_size) as sim:
        # Predict on each batch.
        for batch in test_batches:
            sim_data = sim.predict_on_batch({ndl_mdl_inpt: batch[0]})
            for y_true, y_pred in zip(batch[1], sim_data[ndl_mdl_otpt]):
                # Store the predictions
                ndl_mdl_otpt_cls_probs.append((y_true, y_pred))

    # Calculate accuracy and other metrics from SNN predictions
    y_true_list = []
    y_pred_list = []

    for y_true, y_pred in ndl_mdl_otpt_cls_probs:
        y_true_list.append(np.argmax(y_true))
        y_pred_list.append(np.argmax(y_pred[-1]))  # Last time step prediction

    # Calculate metrics
    final_acc = np.mean(np.array(y_true_list) == np.array(y_pred_list))  # Accuracy in percentage
    prec = precision_score(y_true_list, y_pred_list, average='weighted')
    rec = recall_score(y_true_list, y_pred_list, average='weighted')
    f1 = f1_score(y_true_list, y_pred_list, average='weighted')

    # Create the metrics dictionary
    metrics_dict = {
        "accuracy": final_acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "train_acc_list": train_acc_list,
        "val_acc_list": val_acc_list,
        "train_loss_list": train_loss_list,
        "val_loss_list": val_loss_list,
        "runtime": runtime
    }

    return metrics_dict


# =========================================================
# MAIN EXECUTION & PLOTTING
# =========================================================
if __name__ == "__main__":
    EPOCHS = 50 # Set epoch count

    # Run each model with epoch-based tracking
    cnn_res = run_CNN_Epochs(num_epochs=EPOCHS)
    filename = "cnn_results.json"
    with open(filename, 'w') as file:
        json.dump(cnn_res, file)

    lr_res = run_LogReg_Epochs(num_epochs=EPOCHS)
    filename = "logistic_regression_results.json"
    with open(filename, 'w') as file:
        json.dump(lr_res, file)

    km_res = run_KMeans_Epochs(num_epochs=EPOCHS)
    filename = "kmeans_results.json"
    with open(filename, 'w') as file:
        json.dump(km_res, file)

    gan_res = run_GAN_Epochs(num_epochs=EPOCHS)
    filename = "gan_results.json"
    with open(filename, 'w') as file:
        json.dump(gan_res, file)

    snn_res = run_SNN_Epochs(num_epochs=EPOCHS)
    filename = "snn_results.json"
    with open(filename, 'w') as file:
        json.dump(snn_res, file)


    # cnn_res = {"accuracy": 0.6, "precision": 0.6, "recall": 0.6, "f1_score": 0.6,"train_acc_list": [0]*EPOCHS, "val_acc_list": [0]*EPOCHS,"train_loss_list": [0]*EPOCHS, "val_loss_list": [0]*EPOCHS, "runtime": 0}
    # lr_res = {"accuracy": 0.6, "precision": 0.6, "recall": 0.6, "f1_score": 0.6,"train_acc_list": [0]*EPOCHS, "val_acc_list": [0]*EPOCHS,"train_loss_list": [0]*EPOCHS, "val_loss_list": [0]*EPOCHS, "runtime": 0}
    # km_res = {"accuracy": 0.6, "precision": 0.6, "recall": 0.6, "f1_score": 0.6,"train_acc_list": [0]*EPOCHS, "val_acc_list": [0]*EPOCHS,"train_loss_list": [0]*EPOCHS, "val_loss_list": [0]*EPOCHS, "runtime": 0}
    # gan_res = {"accuracy": 0.6, "precision": 0.6, "recall": 0.6, "f1_score": 0.6,"train_acc_list": [0]*EPOCHS, "val_acc_list": [0]*EPOCHS,"train_loss_list": [0]*EPOCHS, "val_loss_list": [0]*EPOCHS, "runtime": 0}
    # snn_res = {"accuracy": 0.6, "precision": 0.6, "recall": 0.6, "f1_score": 0.6,"train_acc_list": [0]*EPOCHS, "val_acc_list": [0]*EPOCHS,"train_loss_list": [0]*EPOCHS, "val_loss_list": [0]*EPOCHS, "runtime": 0}


    epochs_range = range(1, EPOCHS+1)

    # Create the plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Plot individual model accuracy curves
    plt.figure()
    plt.plot(epochs_range, lr_res["val_acc_list"], label='LogReg Val Acc')
    plt.plot(epochs_range, lr_res["train_acc_list"], label='LogReg Train Acc')
    plt.title("Logistic Regression Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(epochs_range)
    plt.legend()
    plt.savefig('plots/logistic_regression_accuracy.png')  # Save the plot
    plt.close()  # Close the figure

    plt.figure()
    plt.plot(epochs_range, km_res["val_acc_list"], label='K-Means Val Acc')
    plt.plot(epochs_range, km_res["train_acc_list"], label='K-Means Train Acc')
    plt.title("K-Means Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(epochs_range)
    plt.legend()
    plt.savefig('plots/kmeans_accuracy.png')  # Save the plot
    plt.close()  # Close the figure

    plt.figure()
    plt.plot(epochs_range, cnn_res["val_acc_list"], label='CNN Val Acc')
    plt.plot(epochs_range, cnn_res["train_acc_list"], label='CNN Train Acc')
    plt.title("CNN Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(epochs_range)
    plt.legend()
    plt.savefig('plots/cnn_accuracy.png')  # Save the plot
    plt.close()  # Close the figure

    plt.figure()
    plt.plot(epochs_range, gan_res["val_acc_list"], label='GAN Val Acc')
    plt.plot(epochs_range, gan_res["train_acc_list"], label='GAN Train Acc')
    plt.title("GAN Discriminator Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(epochs_range)
    plt.legend()
    plt.savefig('plots/gan_accuracy.png')  # Save the plot
    plt.close()  # Close the figure

    plt.figure()
    plt.plot(epochs_range, snn_res["val_acc_list"], label='SNN Val Acc')
    plt.plot(epochs_range, snn_res["train_acc_list"], label='SNN Train Acc')
    plt.title("SNN Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(epochs_range)
    plt.legend()
    plt.savefig('plots/snn_accuracy.png')  # Save the plot
    plt.close()  # Close the figure


    # Combined validation accuracy plot across all models
    plt.figure()
    plt.plot(epochs_range, cnn_res["val_acc_list"], label='CNN')
    plt.plot(epochs_range, lr_res["val_acc_list"], label='LogReg')
    plt.plot(epochs_range, km_res["val_acc_list"], label='K-Means')
    plt.plot(epochs_range, gan_res["val_acc_list"], label='GAN')
    plt.plot(epochs_range, snn_res["val_acc_list"], label='SNN')
    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(epochs_range)
    plt.legend()
    plt.savefig('plots/validation_accuracy.png')  # Save the plot
    plt.close()  # Close the figure


    # Combined final results bar plot: Accuracy, Precision, Recall, F1, Runtime
    results = {
        "CNN": cnn_res,
        "LogReg": lr_res,
        "KMeans": km_res,
        "GAN": gan_res,
        "SNN": snn_res
    }

    model_names = list(results.keys())
    
    accuracies = [results[m]["accuracy"] for m in model_names]
    precisions = [results[m]["precision"] for m in model_names]
    recalls = [results[m]["recall"] for m in model_names]
    f1_scores = [results[m]["f1_score"] for m in model_names]
    runtimes = [results[m]["runtime"] for m in model_names]

    plt.figure(figsize=(12,8))
    plt.subplot(2, 3, 1)
    plt.bar(model_names, accuracies, color='skyblue')
    plt.title("Accuracy")
    plt.ylim([0, 1])
    for i, v in enumerate(accuracies):
        plt.text(i, v+0.01, f"{v:.2f}", ha='center')

    plt.subplot(2, 3, 2)
    plt.bar(model_names, precisions, color='lightgreen')
    plt.title("Precision")
    plt.ylim([0, 1])
    for i, v in enumerate(precisions):
        plt.text(i, v+0.01, f"{v:.2f}", ha='center')

    plt.subplot(2, 3, 3)
    plt.bar(model_names, recalls, color='salmon')
    plt.title("Recall")
    plt.ylim([0, 1])
    for i, v in enumerate(recalls):
        plt.text(i, v+0.01, f"{v:.2f}", ha='center')

    plt.subplot(2, 3, 4)
    plt.bar(model_names, f1_scores, color='orange')
    plt.title("F1-Score")
    plt.ylim([0, 1])
    for i, v in enumerate(f1_scores):
        plt.text(i, v+0.01, f"{v:.2f}", ha='center')

    plt.subplot(2, 3, 5)
    plt.bar(model_names, runtimes, color='purple')
    plt.title("Runtime (s)")
    for i, v in enumerate(runtimes):
        plt.text(i, v+0.1, f"{v:.2f}s", ha='center')

    plt.tight_layout()
    plt.savefig('plots/final_results.png')  # Save the plot
    plt.close()  # Close the figure

    print("\n=== Final Results ===")
    print(f"CNN -> Acc: {cnn_res['accuracy']:.3f}, F1: {cnn_res['f1_score']:.3f}, Runtime: {cnn_res['runtime']:.2f}s")
    print(f"LogReg -> Acc: {lr_res['accuracy']:.3f}, F1: {lr_res['f1_score']:.3f}, Runtime: {lr_res['runtime']:.2f}s")
    print(f"K-Means -> Acc: {km_res['accuracy']:.3f}, F1: {km_res['f1_score']:.3f}, Runtime: {km_res['runtime']:.2f}s")
    print(f"GAN -> Acc: {gan_res['accuracy']:.3f}, F1: {gan_res['f1_score']:.3f}, Runtime: {gan_res['runtime']:.2f}s")
    print(f"SNN -> Acc: {snn_res['accuracy']:.3f}, F1: {snn_res['f1_score']:.3f}, Runtime: {snn_res['runtime']:.2f}s")