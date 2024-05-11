
from keras.datasets import cifar10
import numpy as np


(x_train, y_train), (x_test, y_test) = cifar10.load_data()



import matplotlib.pyplot as plt
plt.imshow(x_train[30])
plt.show()


from keras.utils import to_categorical
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)


x_train=x_train.reshape(-1,32,32,3)
x_test=x_test.reshape(-1,32,32,3)

x_train.shape,x_test.shape,y_train.shape,y_test.shape


x_train=x_train/255
x_test=x_test/255


from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D,Flatten,Dense,Dropout


inputs=Input(shape=(32,32,3))

c1=Conv2D(64,(3,3),padding="same",activation="relu")(inputs)
m1=MaxPooling2D(padding="same")(c1)

drop1=Dropout(0.3)(m1)

c2=Conv2D(64,(3,3),padding="same",activation="relu")(drop1)
m2=MaxPooling2D(padding="same")(c2)

drop2=Dropout(0.3)(m2)

c3=Conv2D(64,(5,5),padding="same",activation="relu")(drop2)
m3=MaxPooling2D(padding="same")(c3)


drop2=Dropout(0.3)(m3)

conv_out=Flatten()(drop2)

d1=Dense(512,activation="relu")(conv_out)

out=Dense(10,activation="softmax")(d1)
     


model = Model(inputs=inputs, outputs=out)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


model.summary()


history = model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_test, y_test)).history


test_accuracy = model.evaluate(x_test, y_test)[1] * 100
##srore the accuracy in a file
with open("accuracy.txt", "w") as file:
    file.write(str(test_accuracy))
    file.close()
print("Model Final Accuracy:", test_accuracy)


plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Training and Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Validation Loss'], loc='lower right')
plt.show()


model.save("Model.h5")


