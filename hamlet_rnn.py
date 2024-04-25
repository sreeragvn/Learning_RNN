import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import load_model

# YOUR CODE
with open('hamlet_1.txt', encoding= 'utf-8') as f:
    hamlet_1_text= f.read()

with open('hamlet_2.txt', encoding= 'utf-8') as f:
    hamlet_2_text= f.read()

with open('hamlet_3.txt', encoding= 'utf-8') as f:
    hamlet_3_text= f.read()


# YOUR CODE
print(hamlet_1_text[:325])

# YOUR CODE
tokenizer= tf.keras.preprocessing.text.Tokenizer(char_level= True)

tokenizer.fit_on_texts([hamlet_1_text,hamlet_2_text,hamlet_3_text])

print(tokenizer.word_index)

# YOUR CODE
max_id= len(tokenizer.word_index)
print(max_id)

# YOUR CODE
hamlet_1_encoded, hamlet_2_encoded, hamlet_3_encoded= tokenizer.texts_to_sequences([hamlet_1_text, hamlet_2_text, hamlet_3_text])
hamlet_1_encoded= np.asarray(hamlet_1_encoded)- 1
hamlet_2_encoded= np.asarray(hamlet_2_encoded)- 1
hamlet_3_encoded= np.asarray(hamlet_3_encoded)- 1
print(np.min(hamlet_3_encoded))


print(hamlet_1_encoded[:325])


# YOUR CODE
hamlet_1_decoded= ''.join(tokenizer.sequences_to_texts([hamlet_1_encoded + 1]))
print(hamlet_1_decoded[:649])



# YOUR CODE
hamlet_1_dataset= tf.data.Dataset.from_tensor_slices(hamlet_1_encoded)
hamlet_2_dataset= tf.data.Dataset.from_tensor_slices(hamlet_2_encoded)
hamlet_3_dataset= tf.data.Dataset.from_tensor_slices(hamlet_3_encoded)

# YOUR CODE
for i in hamlet_1_dataset.take(10):
    print(i)


# YOUR CODE
T = 100
window_length = T+1

# YOUR CODE
hamlet_1_dataset= hamlet_1_dataset.window(size= window_length, shift= 1, drop_remainder=True)
hamlet_2_dataset= hamlet_2_dataset.window(size= window_length, shift= 1, drop_remainder=True)
hamlet_3_dataset= hamlet_3_dataset.window(size= window_length, shift= 1, drop_remainder=True)


for window in hamlet_1_dataset.take(1):
    print(window)
    for item in window.take(10):
        print(item)


# YOUR CODE
hamlet_1_dataset= hamlet_1_dataset.flat_map(lambda window:window.batch(window_length))
hamlet_2_dataset= hamlet_2_dataset.flat_map(lambda window:window.batch(window_length))
hamlet_3_dataset= hamlet_3_dataset.flat_map(lambda window:window.batch(window_length))

# YOUR CODE
for i in hamlet_1_dataset.take(1):
    print(i)

# YOUR CODE
hamlet_dataset= (hamlet_1_dataset.concatenate(hamlet_2_dataset)).concatenate(hamlet_3_dataset)

tf.random.set_seed(0)
# YOUR CODE
batch_size= 32
hamlet_dataset= hamlet_dataset.repeat().shuffle(buffer_size= 10000).batch(batch_size, drop_remainder= True)


# YOUR CODE
hamlet_dataset= hamlet_dataset.map(lambda window_batch: (window_batch[:, 0:100], window_batch[:, 1:101]))


for window_batch in hamlet_dataset.take(1):
    [x] = tokenizer.sequences_to_texts([window_batch[0][0, :].numpy() + 1])
    [y] = tokenizer.sequences_to_texts([window_batch[1][0, :].numpy() + 1])
    print(x)
    print()
    print(y)


# YOUR CODE

hamlet_dataset= hamlet_dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))


# YOUR CODE
hamlet_dataset= hamlet_dataset.prefetch(buffer_size= 1)

# YOUR CODE
steps_per_epoch= int(((len(hamlet_1_encoded)+len(hamlet_2_encoded)+len(hamlet_3_encoded))-3*T)/batch_size)
print(steps_per_epoch)

# YOUR CODE
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# YOUR CODE
model=keras.models.Sequential([keras.layers.GRU(128,return_sequences=True, input_shape=[None,max_id]),
                                keras.layers.GRU(128,return_sequences=True),
                            keras.layers.TimeDistributed(keras.layers.Dense(max_id,activation="softmax"))])


# YOUR CODE
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")


# YOUR CODE
history= model.fit(hamlet_dataset, epochs=20, steps_per_epoch= steps_per_epoch,callbacks=[callback])


model.save('hamlet_model.h5')

model= tf.keras.models.load_model('hamlet_model.h5')

model







