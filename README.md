# LoFAI
> Gerando músicas de LoFi usando Variational Auto Encoders

## Modelos
Desenvolvemos 3 tipos de VAE, baseados em diferentes famílias de camadas:  
`DenseVAE` para VAEs convencionais,  
`Conv1DVAE` para VAEs com convolução 1D, e  
`LSTMVAE` para VAEs com camadas LSTM.

## Como usar
### Treino
Rode o comando `python train.py [args]` dentro da pasta `src`.  
O primeiro argumento após o nome do arquivo informará o tipo do modelo que será treinado: `dense`, `conv` ou `lstm`. Veja a seção de hiperparâmetros para saber os demais argumentos referentes a cada tipo de modelo.

### Sample
Rode o comando `python sample.py [args]` dentro da pasta `src`. Os argumentos terão de ser iguais aos argumentos passados ao modelo treinado que desejas fazer uma sample.

## Hiperparâmetros
### `dense`
1. `num_layers`: Número de camadas tanto no Encoder e Decoder
2. `latent_dim`: Tamanho da dimensão latente
3. `input_neurons`: Número de neurônios na primeira camada do Encoder/penúltima camada do Decoder 
4. `output_nerons`: Número de neurônios na última camada do Encoder/primeira camada do Decoder  
5. `num_epochs`: Número de épocas para treinar o modelo.

### `conv`
1. `num_conv_layers`: Número de camadas convolucionais no Encoder e no Decoder
2. `num_dense_layers`: Número de camadas densas no Encoder e no Decoder
3. `latent_dim`: Tamanho da dimensão latente
4. `input_neurons`: Número de neurônios na primeira camada do Encoder/penúltima camada do Decoder 
5. `output_nerons`: Número de neurônios na última camada do Encoder/primeira camada do Decoder  
6. `initial_channels`: Número de canais gerados pela primeira camada convolucional do Encoder
7. `factor`: Fator multiplicativo para a sequência do número de canais nas camadas convolucionais
8. `num_epochs`: Número de épocas para treinar o modelo.

### `lstm`
1. `num_layers`: Número de camadas tanto no Encoder e Decoder
2. `latent_dim`: Tamanho da dimensão latente
3. `output_nerons`: Número de neurônios na última camada do Encoder/primeira camada do Decoder  
4. `num_epochs`: Número de épocas para treinar o modelo.