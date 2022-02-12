import os
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import unicodedata

import tensorflow_datasets as tfds
print("Tensorflow version {}".format(tf.__version__))
tf.random.set_seed(1234)
AUTO = tf.data.experimental.AUTOTUNE

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU {}'.format(tpu.cluster_spec().as_dict()['worker']))
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: {}".format(strategy.num_replicas_in_sync))

# Maximum sentence length
ENCODER_LEN = 100
DECODER_LEN = 100
# MAX_LENGTH = 40

# 텐서플로우 dataset을 이용하여 셔플(shuffle)을 수행하되, 배치 크기로 데이터를 묶는다.
# 또한 이 과정에서 교사 강요(teacher forcing)을 사용하기 위해서 디코더의 입력과 실제값 시퀀스를 구성한다.
BATCH_SIZE = 64 * strategy.num_replicas_in_sync
BUFFER_SIZE = 20000

N_EPOCHS = 200

import urllib3
import zipfile
import shutil
import pandas as pd

pd.set_option('display.max_colwidth', None)

http = urllib3.PoolManager()
url ='http://www.manythings.org/anki/fra-eng.zip'
filename = 'fra-eng.zip'
path = os.getcwd()
zipfilename = os.path.join(path, filename)
with http.request('GET', url, preload_content=False) as r, open(zipfilename, 'wb') as out_file:       
    shutil.copyfileobj(r, out_file)

with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
    zip_ref.extractall(path)

train_df = pd.read_csv('fra.txt', names=['SRC', 'TRG', 'lic'], sep='\t')
del train_df['lic']
print(len(train_df))

train_df = train_df.loc[:, 'SRC':'TRG']
    
train_df.head()

train_df["src_len"] = ""
train_df["trg_len"] = ""
train_df.head()

for idx in range(len(train_df['SRC'])):
    # initialize string
    text_eng = str(train_df.iloc[idx]['SRC'])

    # default separator: space
    result_eng = len(text_eng.split())
    train_df.at[idx, 'src_len'] = int(result_eng)

    text_fra = str(train_df.iloc[idx]['TRG'])
    # default separator: space
    result_fra = len(text_fra.split())
    train_df.at[idx, 'trg_len'] = int(result_fra)

print('Translation Pair :',len(train_df)) # 리뷰 개수 출력

train_df = train_df.drop_duplicates(subset = ["SRC"])
print('Translation Pair :',len(train_df)) # 리뷰 개수 출력

train_df = train_df.drop_duplicates(subset = ["TRG"])
print('Translation Pair :',len(train_df)) # 리뷰 개수 출력

# 그 결과를 새로운 변수에 할당합니다.
is_within_len = (8 < train_df['src_len']) & (train_df['src_len'] < 20) & (8 < train_df['trg_len']) & (train_df['trg_len'] < 20)
# 조건를 충족하는 데이터를 필터링하여 새로운 변수에 저장합니다.
train_df = train_df[is_within_len]

dataset_df_8096 = train_df.sample(n=1024*8, # number of items from axis to return.
          random_state=1234) # seed for random number generator for reproducibility

print('Translation Pair :',len(dataset_df_8096)) # 리뷰 개수 출력

raw_src = []
for sentence in dataset_df_8096['SRC']:
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # removing contractions
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "that is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"n'", "ng", sentence)
    sentence = re.sub(r"'bout", "about", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    raw_src.append(sentence)

raw_trg = []

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')
    
for sentence in dataset_df_8096['TRG']:
    # 위에서 구현한 함수를 내부적으로 호출
    sentence = unicode_to_ascii(sentence.lower())

    # 단어와 구두점 사이에 공백을 만듭니다.
    # Ex) "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,¿])", r" \1", sentence)

    # (a-z, A-Z, ".", "?", "!", ",") 이들을 제외하고는 전부 공백으로 변환합니다.
    sentence = re.sub(r"[^a-zA-Z!.?]+", r" ", sentence)

    sentence = re.sub(r"\s+", " ", sentence)

    raw_trg.append(sentence)

print(raw_src[:5])
print(raw_trg[:5])

print('Translation Pair :',len(raw_src)) # 리뷰 개수 출력


# 서브워드텍스트인코더를 사용하여 질문과 답변을 모두 포함한 단어 집합(Vocabulary) 생성
SRC_tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    raw_src, target_vocab_size=2**13)

# 서브워드텍스트인코더를 사용하여 질문과 답변을 모두 포함한 단어 집합(Vocabulary) 생성
TRG_tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    raw_trg, target_vocab_size=2**13)

# 시작 토큰과 종료 토큰에 대한 정수 부여.
START_TOKEN, END_TOKEN = [TRG_tokenizer.vocab_size], [TRG_tokenizer.vocab_size + 1]

# 시작 토큰과 종료 토큰을 고려하여 단어 집합의 크기를 + 2
n_enc_vocab = SRC_tokenizer.vocab_size
n_dec_vocab = TRG_tokenizer.vocab_size + 2

print('시작 토큰 번호           :',START_TOKEN)
print('종료 토큰 번호           :',END_TOKEN)
print('Encoder 단어 집합의 크기 :',n_enc_vocab)
print('Decoder 단어 집합의 크기 :',n_dec_vocab)

lines = [
  "It is winter and the weather is very cold.",
  "Will this Christmas be a white Christmas?",
  "Be careful not to catch a cold in winter and have a happy new year."
]
for line in lines:
    txt_2_ids = SRC_tokenizer.encode(line)
    ids_2_txt = SRC_tokenizer.decode(txt_2_ids)
    print("Input     :", line)
    print("txt_2_ids :", txt_2_ids)
    print("ids_2_txt :", ids_2_txt,"\n")

lines = [
  "C'est l'hiver et il fait très froid.",
  "Ce Noël sera-t-il un Noël blanc ?",
  "Attention à ne pas attraper froid en hiver et bonne année."
]
for line in lines:
    txt_2_ids = TRG_tokenizer.encode(line)
    ids_2_txt = TRG_tokenizer.decode(txt_2_ids)
    print("Input     :", line)
    print("txt_2_ids :", txt_2_ids)
    print("ids_2_txt :", ids_2_txt,"\n")
    
# 각 정수는 각 단어와 어떻게 mapping되는지 병렬로 출력
# 서브워드텍스트인코더는 의미있는 단위의 서브워드로 토크나이징한다. 띄어쓰기 단위 X 형태소 분석 단위 X
for ts in txt_2_ids:
    print ('{} ----> {}'.format(ts, TRG_tokenizer.decode([ts])))

# 토큰화 / 정수 인코딩 / 시작 토큰과 종료 토큰 추가 / 패딩
tokenized_inputs, tokenized_outputs = [], []

for (sentence1, sentence2) in zip(raw_src, raw_trg):
    sentence1 = SRC_tokenizer.encode(sentence1)
    sentence2 = START_TOKEN + TRG_tokenizer.encode(sentence2) + END_TOKEN

    tokenized_inputs.append(sentence1)
    tokenized_outputs.append(sentence2)

# 패딩
tkn_sources = tf.keras.preprocessing.sequence.pad_sequences(tokenized_inputs, maxlen=ENCODER_LEN, padding='post')
tkn_targets = tf.keras.preprocessing.sequence.pad_sequences(tokenized_outputs, maxlen=DECODER_LEN, padding='post')

tkn_sources = tf.cast(tkn_sources, dtype=tf.int64)
tkn_targets = tf.cast(tkn_targets, dtype=tf.int64)

print('질문 데이터의 크기(shape) :', tkn_sources.shape)
print('답변 데이터의 크기(shape) :', tkn_targets.shape)

# 0번째 샘플을 임의로 출력
print(tkn_sources[0])
print(tkn_targets[0])

# Hyper-parameters
n_layers  = 2     # 6
hid_dim   = 256
pf_dim    = 1024
n_heads   = 8
dropout   = 0.3

# 디코더의 실제값 시퀀스에서는 시작 토큰을 제거해야 한다.
dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': tkn_sources,
        'dec_inputs': tkn_targets[:, :-1] # 디코더의 입력. 마지막 패딩 토큰이 제거된다.
    },
    {
        'outputs': tkn_targets[:, 1:]  # 맨 처음 토큰이 제거된다. 다시 말해 시작 토큰이 제거된다.
    },
))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

# 임의의 샘플에 대해서 [:, :-1]과 [:, 1:]이 어떤 의미를 가지는지 테스트해본다.
print(tkn_targets[0]) # 기존 샘플
print(tkn_targets[:1][:, :-1]) # 마지막 패딩 토큰 제거하면서 길이가 39가 된다.
print(tkn_targets[:1][:, 1:]) # 맨 처음 토큰이 제거된다. 다시 말해 시작 토큰이 제거된다. 길이는 역시 39가 된다.

""" sinusoid position encoding """
class get_sinusoid_encoding_table(tf.keras.layers.Layer):

    def __init__(self, position, hid_dim):
        super(get_sinusoid_encoding_table, self).__init__()
        self.pos_encoding = self.positional_encoding(position, hid_dim)

    def get_angles(self, position, i, hid_dim):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(hid_dim, tf.float32))
        return position * angles

    def positional_encoding(self, position, hid_dim):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(hid_dim, dtype=tf.float32)[tf.newaxis, :],
            hid_dim=hid_dim)

        # 배열의 짝수 인덱스(2i)에는 사인 함수 적용
        sines = tf.math.sin(angle_rads[:, 0::2])

        # 배열의 홀수 인덱스(2i+1)에는 코사인 함수 적용
        cosines = tf.math.cos(angle_rads[:, 1::2])

        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines
        pos_encoding = tf.constant(angle_rads)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        print(pos_encoding.shape)
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

sample_pos_encoding = get_sinusoid_encoding_table(50, 128)

plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 128))
plt.ylabel('Position')
plt.colorbar()
plt.show()

""" attention pad mask """
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # (batch_size, 1, 1, key의 문장 길이)
    return seq[:, tf.newaxis, tf.newaxis, :]

# 디코더의 첫번째 서브층(sublayer)에서 미래 토큰을 Mask하는 함수
def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x) # 패딩 마스크도 포함
    return tf.maximum(look_ahead_mask, padding_mask)

""" scale dot product attention """
def ScaledDotProductAttention(query, key, value, mask):
    """Calculate the attention weights.
    query, key, value must have matching leading dimensions.
    key, value must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
    
    query, key, value의 leading dimensions은 동일해야 합니다.
    key, value 에는 일치하는 끝에서 두 번째 차원이 있어야 합니다(예: seq_len_k = seq_len_v).
    MASK는 유형에 따라 모양이 다릅니다(패딩 혹은 미리보기(=look ahead)).
    그러나 추가하려면 브로드캐스트할 수 있어야 합니다.

    Args:
        query: query shape == (batch_size, n_heads, seq_len_q, depth)
        key: key shape     == (batch_size, n_heads, seq_len_k, depth)
        value: value shape == (batch_size, n_heads, seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
              to (batch_size, n_heads, seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """
    
    matmul_qk = tf.matmul(query, key, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, value)  # (..., seq_len_q, depth_v)

    return output, attention_weights

""" multi head attention """
class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    
    def __init__(self, hid_dim, n_heads, name="multi_head_attention"):
        super(MultiHeadAttentionLayer, self).__init__(name=name)
        self.n_heads = n_heads
        assert hid_dim % self.n_heads == 0
        self.hid_dim = hid_dim
        
        # hid_dim을 n_heads로 나눈 값.
        self.depth = int(hid_dim/self.n_heads)
        
        # WQ, WK, WV에 해당하는 밀집층 정의
        self.q_linear = tf.keras.layers.Dense(hid_dim)
        self.k_linear = tf.keras.layers.Dense(hid_dim)
        self.v_linear = tf.keras.layers.Dense(hid_dim)
        # WO에 해당하는 밀집층 정의
        self.out = tf.keras.layers.Dense(hid_dim)

    def split_heads(self, inputs, batch_size):
        """Split the last dimension into (n_heads, depth).
        Transpose the result such that the shape is (batch_size, n_heads, seq_len, depth)
        """
        inputs = tf.reshape(
            inputs, (batch_size, -1, self.n_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]
        # 1. WQ, WK, WV에 해당하는 밀집층 지나기
        # q : (batch_size, query의 문장 길이, hid_dim)
        # k : (batch_size, key의 문장 길이, hid_dim)
        # v : (batch_size, value의 문장 길이, hid_dim)
        query = self.q_linear(query)
        key   = self.k_linear(key)
        value = self.v_linear(value)
        
        # 2. 헤드 나누기
        # q : (batch_size, n_heads, query의 문장 길이, hid_dim/n_heads)
        # k : (batch_size, n_heads, key의 문장 길이,   hid_dim/n_heads)
        # v : (batch_size, n_heads, value의 문장 길이, hid_dim/n_heads)
        query = self.split_heads(query, batch_size)
        key   = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        # 3. 스케일드 닷 프로덕트 어텐션. 앞서 구현한 함수 사용.
        # (batch_size, n_heads, query의 문장 길이, hid_dim/n_heads)
        # attention_weights.shape == (batch_size, n_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = ScaledDotProductAttention(
            query, key, value, mask)
        
        # (batch_size, query의 문장 길이, n_heads, hid_dim/n_heads)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        
        # 4. 헤드 연결(concatenate)하기
        # (batch_size, query의 문장 길이, hid_dim)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.hid_dim))
        
        # 5. WO에 해당하는 밀집층 지나기
        # (batch_size, query의 문장 길이, hid_dim)
        outputs = self.out(concat_attention)

        return outputs

""" feed forward """
class PositionwiseFeedforwardLayer(tf.keras.layers.Layer):
    def __init__(self, hid_dim, pf_dim):
        super(PositionwiseFeedforwardLayer, self).__init__()
        self.linear_1 = tf.keras.layers.Dense(pf_dim, activation='relu')
        self.linear_2 = tf.keras.layers.Dense(hid_dim)

    def forward(self, attention):
        output = self.linear_1(attention)
        output = self.linear_2(output)
        return output

""" encoder layer """
def encoder_layer(pf_dim, hid_dim, n_heads, dropout, name="encoder_layer"):
    
    inputs = tf.keras.Input(shape=(None, hid_dim), name="inputs")

    # 인코더는 패딩 마스크 사용
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    # 멀티-헤드 어텐션 (첫번째 서브층 / 셀프 어텐션)
    attention = MultiHeadAttentionLayer(
        hid_dim, n_heads, name="attention")({
        'query': inputs, 'key': inputs, 'value': inputs, # Q = K = V
        'mask': padding_mask # 패딩 마스크 사용
    })

    # 드롭아웃 + 잔차 연결과 층 정규화
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization( epsilon=1e-6)(inputs + attention)

    # 포지션 와이즈 피드 포워드 신경망 (두번째 서브층)
    outputs = PositionwiseFeedforwardLayer(hid_dim, pf_dim)(attention)
    
    # 드롭아웃 + 잔차 연결과 층 정규화
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)

""" encoder """
def encoder(n_enc_vocab, n_layers, pf_dim, hid_dim, n_heads, dropout,
            name="encoder"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")

    # 인코더는 패딩 마스크 사용
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    # adding embedding and position encoding.
    emb = tf.keras.layers.Embedding(n_enc_vocab, hid_dim)(inputs)
    emb *= tf.math.sqrt(tf.cast(hid_dim, tf.float32))
    emb = get_sinusoid_encoding_table(n_enc_vocab, hid_dim)(emb)
    outputs = tf.keras.layers.Dropout(rate=dropout)(emb)

    # 인코더를 n_layers개 쌓기
    for i in range(n_layers):
        outputs = encoder_layer(pf_dim=pf_dim, hid_dim=hid_dim, n_heads=n_heads,
            dropout=dropout, name="encoder_layer_{}".format(i),
        )([outputs, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)

    
""" decoder layer """
def DecoderLayer(pf_dim, hid_dim, n_heads, dropout, name="DecoderLayer"):
    inputs = tf.keras.Input(shape=(None, hid_dim), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, hid_dim), name="encoder_outputs")

    # 디코더는 룩어헤드 마스크(첫번째 서브층)와 패딩 마스크(두번째 서브층) 둘 다 사용.
    look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    # 멀티-헤드 어텐션 (첫번째 서브층 / 마스크드 셀프 어텐션)
    attention1 = MultiHeadAttentionLayer(
        hid_dim, n_heads, name="attention_1")(inputs={
        'query': inputs, 'key': inputs, 'value': inputs, # Q = K = V
        'mask': look_ahead_mask # 룩어헤드 마스크
    })

    # 잔차 연결과 층 정규화
    attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)

    # 멀티-헤드 어텐션 (두번째 서브층 / 디코더-인코더 어텐션)
    attention2 = MultiHeadAttentionLayer(
        hid_dim, n_heads, name="attention_2")(inputs={
        'query': attention1, 'key': enc_outputs, 'value': enc_outputs, # Q != K = V
        'mask': padding_mask # 패딩 마스크
    })

    # 드롭아웃 + 잔차 연결과 층 정규화
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + attention1)

    # 포지션 와이즈 피드 포워드 신경망 (세번째 서브층)
    outputs = PositionwiseFeedforwardLayer(hid_dim, pf_dim)(attention2)
    
    # 드롭아웃 + 잔차 연결과 층 정규화
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)

""" decoder """
def decoder(n_dec_vocab, n_layers, pf_dim, hid_dim, n_heads, dropout,
            name='decoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, hid_dim), name='encoder_outputs')

    # 디코더는 룩어헤드 마스크(첫번째 서브층)와 패딩 마스크(두번째 서브층) 둘 다 사용.
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    # adding embedding and position encoding.
    emb = tf.keras.layers.Embedding(n_dec_vocab, hid_dim)(inputs)
    emb *= tf.math.sqrt(tf.cast(hid_dim, tf.float32))
    emb = get_sinusoid_encoding_table(n_dec_vocab, hid_dim)(emb)

    outputs = tf.keras.layers.Dropout(rate=dropout)(emb)

    # 디코더를 n_layers개 쌓기
    for i in range(n_layers):
        outputs = DecoderLayer( pf_dim=pf_dim, hid_dim=hid_dim, n_heads=n_heads,
            dropout=dropout, name='DecoderLayer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)
    
def create_masks(inp, tar):
    # 인코더의 패딩 마스크
    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='enc_padding_mask')(inp)

    # 디코더의 룩어헤드 마스크(첫번째 서브층)
    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask, output_shape=(1, None, None),
        name='look_ahead_mask')(tar)

    # 디코더의 패딩 마스크(두번째 서브층)
    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='dec_padding_mask')(inp)
  
    return enc_padding_mask, look_ahead_mask, dec_padding_mask

# Model Define for Training
""" transformer """
def Transformer(n_enc_vocab, n_dec_vocab, n_layers, pf_dim, hid_dim, n_heads, dropout,
                name="Transformer"):

    # 인코더의 입력
    inputs = tf.keras.Input(shape=(None,), name="inputs")

    # 디코더의 입력
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")
    
    enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inputs, dec_inputs)
    
    # 인코더의 출력은 enc_outputs. 디코더로 전달된다.
    enc_outputs = encoder(n_enc_vocab=n_enc_vocab, n_layers=n_layers, pf_dim=pf_dim,
                          hid_dim=hid_dim, n_heads=n_heads, dropout=dropout,
                         )(inputs=[inputs, enc_padding_mask]) # 인코더의 입력은 입력 문장과 패딩 마스크

    # 디코더의 출력은 dec_outputs. 출력층으로 전달된다.
    dec_outputs = decoder(n_dec_vocab=n_dec_vocab, n_layers=n_layers, pf_dim=pf_dim,
                          hid_dim=hid_dim, n_heads=n_heads, dropout=dropout,
                         )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    # 다음 단어 예측을 위한 출력층
    outputs = tf.keras.layers.Dense(units=n_dec_vocab, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, hid_dim, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.hid_dim = hid_dim
        self.hid_dim = tf.cast(self.hid_dim, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.hid_dim) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(hid_dim)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

temp_learning_rate_schedule = CustomSchedule(hid_dim)

plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")

tf.keras.backend.clear_session()

def accuracy(y_true, y_pred):
    # ensure labels have shape (batch_size, DECODER_LEN - 1)
    y_true = tf.reshape(y_true, shape=(-1, DECODER_LEN - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
    # return tf.keras.metrics.SparseCategoricalCrossentropy(y_true, y_pred)


# initialize and compile model within strategy scope
with strategy.scope():
    model = Transformer(
        n_enc_vocab = n_enc_vocab,
        n_dec_vocab = n_dec_vocab,
        n_layers  = n_layers,
        pf_dim      = pf_dim,
        hid_dim     = hid_dim,
        n_heads     = n_heads,
        dropout     = dropout)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

model.summary()

tf.keras.utils.plot_model(
    model, to_file='transformer.png', show_shapes=True)

checkpoint_path = "./checkpoints/Transformer.h5"
if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)
    print('Latest checkpoint restored!!')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
# tf.keras.callbacks.EarlyStopping( monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

# 콜백 함수 loss
es = EarlyStopping(monitor='loss', min_delta=0.0001, patience = 20)

# mc = ModelCheckpoint(checkpoint_path, save_best_only=True)
# rlr = ReduceLROnPlateau(factor=0.1, patience=5)
# csvlogger = CSVLogger("your_path/file_name.log")
# fit 안에 위의 콜백 함수를 넣어주면 됩니다. 
# model.fit(x_train, y_train, epochs=20, batch_size=128, callbacks=[es, mc, rlr, csvlogger])

model.fit(dataset, epochs=N_EPOCHS, callbacks=[es])
# model.fit(dataset, epochs=N_EPOCHS)


import os
if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")
model.save_weights(checkpoint_path)

def evaluate(text):
    text = preprocess_sentence(text)

    encoder_input = tf.expand_dims(SRC_tokenizer.encode(text), axis=0)

    output = tf.expand_dims(START_TOKEN, 0)

    # 디코더의 예측 시작
    for i in range(DECODER_LEN):
        predictions = model(inputs=[encoder_input, output], training=False)

        # 현재(마지막) 시점의 예측 단어를 받아온다.
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # 만약 마지막 시점의 예측 단어가 종료 토큰이라면 예측을 중단
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # 마지막 시점의 예측 단어를 출력에 연결한다.
        # 이는 for문을 통해서 디코더의 입력으로 사용될 예정이다.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)

def predict(text):
    prediction = evaluate(text)

    predicted_sentence = TRG_tokenizer.decode(
        [i for i in prediction if i < TRG_tokenizer.vocab_size])
    
    return predicted_sentence

def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence

for idx in (11, 21, 31, 41, 51):
    print("Input        :", raw_src[idx])
    print("Prediction   :", predict(raw_src[idx]))
    print("Ground Truth :", raw_trg[idx],"\n")
    