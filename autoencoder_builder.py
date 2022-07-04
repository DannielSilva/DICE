import torch
import torch.nn as nn
import enum

class AutoEncoderEnum(enum.Enum):
    ORIGINAL_DICE = "original"
    LSTM = "lstm"
        
class Encoder(nn.Module):
    def __init__(self, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.n_features = n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn1 = nn.LSTM(
        input_size=n_features,
        hidden_size=self.hidden_dim,
        num_layers=1,
        batch_first=True
        )
        self.rnn2 = nn.LSTM(
        input_size=self.hidden_dim,
        hidden_size=embedding_dim,
        num_layers=1,
        batch_first=True
        )
    def forward(self, x):
        self.seq_len = x.shape[1] #shape: [bs, seq_len, input_dim]
        bs = x.shape[0]
        x = x.reshape((bs, self.seq_len, self.n_features))
        x, (_, _) = self.rnn1(x)
        #print('a',x.shape)
        x, (hn, _) = self.rnn2(x)
        #print('hn', hn.shape, x.shape)
        return x, hn.reshape((bs, self.embedding_dim))

class Decoder(nn.Module):
  def __init__(self, input_dim=64, n_features=1):
    super(Decoder, self).__init__()
    self.input_dim =  input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features
    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.input_dim,
      num_layers=1,
      batch_first=True
    )
    #self.output_layer = nn.Linear(self.hidden_dim, n_features)
  def forward(self, x, seq_len):
    self.seq_len = seq_len #shape: [bs, seq_len, input_dim]
    bs = x.shape[0]
    #print(self.seq_len,x.shape)
    x = x.repeat(self.seq_len, 1)
    x = x.reshape((bs, self.seq_len, self.input_dim))
    x, (hn, cn) = self.rnn1(x)
    x, (hn, cn) = self.rnn2(x)
    x = x.reshape((bs,self.seq_len, self.input_dim))
    return x	#return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):
  def __init__(self, n_features, embedding_dim=64):
    super(RecurrentAutoencoder, self).__init__()
    self.encoder = Encoder(n_features, embedding_dim)
    self.decoder = Decoder(embedding_dim, n_features)
  def forward(self, x):
    seq_len = x.shape[1] #shape: [bs, seq_len, input_dim]
    encoded_x, hn = self.encoder(x)
    decoded_x = self.decoder(hn, seq_len)
    return hn, decoded_x

###############

#Original AutoEncoder architecture from DICE

###############

class EncoderRNN(nn.Module):
    def __init__(self, input_size, nhidden, nlayers, dropout, cuda):
        super(EncoderRNN, self).__init__()
        self.nhidden = nhidden
        self.feasize = input_size
        self.nlayers = nlayers
        self.dropout = dropout
        self.cuda = cuda 
        self.lstm = nn.LSTM(input_size=self.feasize,
                               hidden_size=self.nhidden,
                               num_layers=self.nlayers,
                               dropout=self.dropout,
                               batch_first=True)
        self.init_weights()

    def init_weights(self):
        #nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        for p in self.lstm.parameters():
            p.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        batch_size = x.size()[0]
        output, state = self.lstm(x) #output [batch_size, seq_size, hidden_size]
        hn, cn = state
        #hidden = hidden_state[-1]  # get hidden state of last layer of encoder
        output = torch.flip(output, [1])
        newinput = torch.flip(x,[1])        
        zeros = torch.zeros(batch_size, 1, x.shape[-1]) #zeros = torch.zeros(batch_size, 1, x.shape[-1])
        if self.cuda:
            zeros = zeros.cuda()
        newinput = torch.cat((zeros, newinput),1)
        newinput = newinput[:, :-1, :]
        #import IPython; IPython.embed(); import sys; sys.exit(0)
        #print("output.size()=",output.size()) # output.size()= torch.Size([1, 10, 100])
        #print("hn.size()=",hn.size()) # hn.size()= torch.Size([1, 1, 100])
        #print("hn=",hn)
        #print("output[0]=",output[0])
        return output, (hn, cn), newinput

class DecoderRNN(nn.Module):
    def __init__(self, input_size, nhidden, nlayers, dropout):
        super(DecoderRNN, self).__init__()
        self.nhidden = nhidden
        self.feasize = input_size
        self.nlayers = nlayers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size=self.feasize,
                               hidden_size=self.nhidden,
                               num_layers=self.nlayers,
                               dropout=self.dropout,
                               batch_first=True)
        self.init_weights()

    def init_weights(self):
        #nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        for p in self.lstm.parameters():
            p.data.uniform_(-0.1, 0.1)

    def forward(self, x, h):
        output, state = self.lstm(x, h)
        fin = torch.flip(output, [1])
        return fin


class DICEAutoEncoder(nn.Module):
  def __init__(self,input_size, nhidden, nlayers, dropout, cuda):
    super(DICEAutoEncoder,self).__init__()
    self.encoder = EncoderRNN(input_size, nhidden, nlayers, dropout, cuda)
    self.decoder = DecoderRNN(input_size, nhidden, nlayers, dropout)
    self.nhidden = nhidden

  def forward(self, x):
    encoded_x, (hn, cn), newinput = self.encoder(x)
    decoded_x = self.decoder(newinput, (hn, cn))
    return hn.reshape(x.shape[0], self.nhidden), decoded_x

def get_auto_encoder(type,input_size, nhidden, nlayers, dropout, cuda):
    if type == AutoEncoderEnum.ORIGINAL_DICE.value:
        print('Building AutoEncoder from original DICE implementation')
        return DICEAutoEncoder(input_size, nhidden, nlayers, dropout, cuda)
    
    elif type == AutoEncoderEnum.LSTM.value:
        print('Building AutoEncoder with new LSTM implementation')
        return RecurrentAutoencoder(input_size, nhidden)
    
    else:
        print('Wrong AE type!')
    