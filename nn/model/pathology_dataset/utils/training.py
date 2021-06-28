from utils.model.ekm import ekm
from utils.model.dense_net import densenet121
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from utils.plot.history_plot import plot_train_history
from utils.plot.confusion_matrix import plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from utils.data.classes_balance import get_augmented_test

def tensorSort(data):
    return sorted(data, key=lambda item: (int(item.partition(' ')[0])
                               if item[0].isdigit() else float('inf'), item))

def train_val(checkpoint_filepaths, args, batch_size, train_size, train_flow_80, val_flow, y_val, y_train, total_train_80, train_flow, total_train, x, y, xS, yS, TEST_CSV, test_dir):
    drops =[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    val_loss =[]
    max_val_loss_epoch = []
    val_acc = []
    max_val_acc_epoch = []
    val_auc = []
    max_val_auc_epoch = []
    histories = []
    epochs_l = []
    i = 0

    for drop in drops:
        model = ekm(drop)
        args["filepath"] = checkpoint_filepaths[i]
        history = model.fit_generator(train_flow_80,
                    steps_per_epoch = total_train_80 // batch_size, #train_size//batch_size
                    epochs=80, # the model never seems to suffer from validation loss increase (even up to 100 epochs)
                    validation_data=val_flow,
                    validation_steps=1,
                    callbacks = [ModelCheckpoint(**args)], # we tried early stopping and learning rate scheduling, but they proved inefficient due to the high loss swipes we had during training.
                    workers=4,
                    verbose = 0 )   
        val_loss.append(np.max(np.array(history.history["val_loss"])))
        val_acc.append(np.max(np.array(history.history['val_categorical_accuracy'])))
        val_auc.append(np.max(np.array(history.history['val_categorical_auc'])))
        epochs_l.append(np.argmax(np.array(history.history['val_categorical_auc'])))
        histories.append(history)
        i = i+1
        print("drop  = ",drop, "done, next...")
        
    history  = histories[np.argmax(np.array(val_auc))]
    drop = drops[np.argmax(np.array(val_auc))]
    epochs = epochs_l[np.argmax(np.array(val_auc))]+1 #epochs = np.argmax(np.array(history.history["val_categorical_auc"]))
    best_weights = checkpoint_filepaths[np.argmax(np.array(val_auc))]
    
    ###############################################
    print("best drop = ", drop,"best epochs = ", epochs, "best_weights = ", best_weights)

    fig, axs = plt.subplots(1,3, figsize = (15,15))
    axs[0].set_title("val_loss")
    axs[0].set_xlabel("dropout")
    axs[0].set_ylabel("val_loss")
    axs[0].plot(drops, val_loss)

    axs[1].set_title("val_acc")
    axs[1].set_xlabel("dropout")
    axs[1].set_ylabel("val_acc")
    axs[1].plot(drops, val_acc)

    axs[2].set_title("val_auc")
    axs[2].set_xlabel("dropout")
    axs[2].set_ylabel("val_loss")
    axs[2].plot(drops, val_auc)
    plt.savefig("../pathology_dataset/results/ekm_training_result.png")
    
    ###############################################
    model_new = ekm(drop)
    history_new = model_new.fit_generator(train_flow_80,
                steps_per_epoch = total_train_80 // batch_size, #train_size//batch_size
                epochs=80,           # the model never seems to suffer from validation loss increase (even up to 100 epochs)
                validation_data=val_flow,
                validation_steps=1,  # we tried early stopping and learning rate scheduling, but they proved inefficient due to the high loss swipes we had during training.
                workers=4)   

    deeper_model = densenet121()
    deeper_history = deeper_model.fit_generator(train_flow_80,
                steps_per_epoch = total_train_80 // batch_size, #train_size//batch_size
                epochs=20,               # the model never seems to suffer from validation loss increase (even up to 100 epochs)
                validation_data=val_flow,
                validation_steps=1,      # we tried early stopping and learning rate scheduling, but they proved inefficient due to the high loss swipes we had during training.
                workers=4)
    
    ###############################################
    # Plot training history
    plot_train_history(history) # EKM

    plot_train_history(history_new) # Stochasticity Evaluation

    plot_train_history(deeper_history) # DenseNet121

    ###############################################
    # Plot confusion matrix - EKM
    Y_pred = model.predict_generator(val_flow, train_size // batch_size + 1) #128 +1
    y_pred = np.argmax(Y_pred, axis=1)
    a = confusion_matrix(y_val, y_pred)

    plot_confusion_matrix(a,["h","d","c","f"], "ekm" ,normalize=True)

    # Plot confusion matrix - DenseNet121
    Y_pred = deeper_model.predict_generator(val_flow, train_size // batch_size +1) #128 +1
    y_pred = np.argmax(Y_pred, axis=1)
    a = confusion_matrix(y_val, y_pred)

    plot_confusion_matrix(a,["h","d","c","f"], "dense_net_121", normalize=True)

    ###############################################
    # Prediction and Submission

    # Save memory
    del y,yS
    del train_flow_80, y_train, val_flow, y_val

    # Load test
    test_imgs = get_augmented_test(test_dir = test_dir, test_generator = test_datagen)
    print(test_imgs.shape)

    # EKM model:
    EKM = ekm(drop) #0.4
    EKM.fit_generator(train_flow,
                steps_per_epoch = total_train // batch_size, #train_size//batch_size
                epochs=epochs,
                #callbacks=[lr_schedule],
                workers=4)
    y_predicted = EKM.predict(test_imgs)
    submission = pd.DataFrame(y_predicted, columns = ["healthy", "multiple_diseases", "rust","scab"],)
    submission.insert(0,"image_id",tensorSort(test_csv["image_id"].tolist()))
    submission.to_csv("../dataset/submission.csv", index = False)
    print(f"Submission is: {submission}")

    model_loaded = ekm(drop)
    model_loaded.load_weights(best_weights)

    y_predicted = model_loaded.predict(test_imgs)
    submission_loaded = pd.DataFrame(y_predicted, columns = ["healthy", "multiple_diseases", "rust","scab"])
    submission_loaded.insert(0,"image_id",tensorSort(test_csv["image_id"].tolist()))
    submission_loaded.to_csv("../dataset/submission_loaded.csv", index = False)
    submission_loaded
    print(f"Submission is: {submission_loaded}")
