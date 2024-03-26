from module import *

filepath=tf.keras.utils.get_file('shakespear.txt','https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text=open(filepath,'rb').read().decode(encoding='utf-8').lower()
text = text[300000:800000]

characters = sorted(set(text))
char_to_index = {c: i for i, c in enumerate(characters)}
index_to_char = {c: i for i, c in enumerate(characters)}


seq_length=40
step_size=3

def sample(prods,temperature=1.0):
    prods=np.asarray(prods).astype('float64')
    prods=np.log(prods)/temperature
    exp_prods=np.exp(prods)
    prods=exp_prods/np.sum(exp_prods)
    probas=np.random.multinomial(1,prods,1)
    return np.argmax(probas)

def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - seq_length - 1)
    generated = ''
    sentence = text[start_index: start_index + seq_length]
    generated += sentence
    for i in range(length):
        x_predictions = np.zeros((1, seq_length, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1

        predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions,
                                 temperature)
        next_character = index_to_char.get(next_index, '')

        generated += next_character
        sentence = sentence[1:] + next_character
    return generated

model =tf.keras.models.load_model('textgenerator.h5')

print("----------0.2--------")
print(generate_text(800, 0.2))
print("----------0.4--------")
print(generate_text(300, 0.4))
print("----------0.5--------")
print(generate_text(300, 0.5))
print("----------0.6--------")
print(generate_text(300, 0.6))
print("----------0.7--------")
print(generate_text(300, 0.7))
print("----------0.8--------")
print(generate_text(300, 0.8))
