from module import *

filepath=tf.keras.utils.get_file('shakespear.txt','https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text=open(filepath,'rb').read().decode(encoding='utf-8').lower()
text = text[300000:800000]

characters = sorted(set(text))
char_to_index = {c: i for i, c in enumerate(characters)}
index_to_char = {c: i for i, c in enumerate(characters)}


seq_length=40
step_size=3

sentences=[]
next_char=[]

for i in range(0,len(text)-seq_length,step_size):
    sentences.append(text[i:i+seq_length])
    next_char.append(text[i + seq_length])

a=np.zeros((len(sentences), seq_length,len(characters)), dtype=np.bool)
b=np.zeros((len(sentences),len(characters)), dtype=np.bool)

for i,s in enumerate(sentences):
    for j,char in enumerate(s):
        a[i, j, char_to_index[char]] = 1
    b[i,char_to_index[next_char[i]]]=1

model=Sequential()
model.add(LSTM(128,input_shape=(seq_length,len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))
model.fit(a,b,batch_size=256,epochs=4)

model.save('textgenerator.h5')






