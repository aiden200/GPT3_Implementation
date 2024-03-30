map_char_to_int = None
map_int_to_char = None

def information_on_data(file):
    with open(file, 'r', encoding= 'utf-8') as f:
        txt = f.read()
    
    data_len = len(txt)
    vocab = sorted(list(set(txt)))
    set_encodings(vocab)
    return data_len

def set_encodings(vocab):
    #very basic level of encodings
    global map_char_to_int, map_int_to_char
    map_char_to_int = { ch:i for i,ch in enumerate(vocab)} 
    map_int_to_char = { i:ch for i,ch in enumerate(vocab)} 

def encode(text):
    #character level tokenizer, long long tokens but easy and simple
    if map_char_to_int == None:
        print("Encodings not set")
        exit(0)
    encodings = []
    for char in text:
        encodings.append(map_char_to_int[char])
    return encodings

def decode(text):
    if map_int_to_char == None:
        print("Decodings not set")
        exit(0)
    decodings = ""
    for number in text:
        decodings = decodings + map_int_to_char[number]
    return decodings