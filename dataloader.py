import os, pickle, random
from glob import glob

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

def transpose_events(raw_events, n_keys):
  transposed_raw_events = []

  for ev in raw_events:
    if ev['name'] == 'Pitch':
      transposed_raw_events.append(
        {'name': ev['name'], 'value': min(max(int(ev['value']) + n_keys, 21), 108)}
      )
    else:
      transposed_raw_events.append(ev)
  assert len(transposed_raw_events) == len(raw_events)
  return transposed_raw_events
  
def check_extreme_pitch(raw_events):
  low, high = 128, 0
  for ev in raw_events:
    if ev['name'] == 'Note_Pitch':
      low = min(low, int(ev['value']))
      high = max(high, int(ev['value']))

  return low, high

def pickle_load(path):
  return pickle.load(open(path, 'rb'))

def convert_event(event_seq, event2idx, to_ndarr=True):
  if isinstance(event_seq[0], dict):
    event_seq = [event2idx['{}_{}'.format(e['name'], e['value'])] for e in event_seq]
  else:
    event_seq = [event2idx[e] for e in event_seq]

  if to_ndarr:
    return np.array(event_seq)
  else:
    return event_seq

class REMIFullSongTransformerDataset(Dataset):
  def __init__(self, data_dir, vocab_file, 
               model_enc_seqlen=128, model_dec_seqlen=1280, model_max_bars=16,
               pieces=[], do_augment=True, augment_range=range(-6, 7), 
               min_pitch=22, max_pitch=107, pad_to_same=True, use_attr_cls=True,
               appoint_st_bar=None, dec_end_pad_value=None):
    self.vocab_file = vocab_file
    self.read_vocab()

    self.data_dir = data_dir
    self.pieces = pieces
    self.build_dataset()

    self.model_enc_seqlen = model_enc_seqlen
    self.model_dec_seqlen = model_dec_seqlen
    self.model_max_bars = model_max_bars

    self.do_augment = do_augment
    self.augment_range = augment_range
    self.min_pitch, self.max_pitch = min_pitch, max_pitch
    self.pad_to_same = pad_to_same
    self.use_attr_cls = use_attr_cls

    self.appoint_st_bar = appoint_st_bar
    if dec_end_pad_value is None:
      self.dec_end_pad_value = self.pad_token
    elif dec_end_pad_value == 'EOS':
      self.dec_end_pad_value = self.eos_token
    else:
      self.dec_end_pad_value = self.pad_token

  def read_vocab(self):
    vocab = pickle_load(self.vocab_file)[0]
    self.idx2event = pickle_load(self.vocab_file)[1]
    orig_vocab_size = len(vocab)
    self.event2idx = vocab
    self.bar_token = self.event2idx['Bar_None']
    self.eos_token = self.event2idx['EOS_None']
    self.pad_token = orig_vocab_size
    self.vocab_size = self.pad_token + 1
  
  def build_dataset(self):
    if not self.pieces:
      self.pieces = sorted( glob(os.path.join(self.data_dir, '*.pkl')) )
    else:
      self.pieces = sorted( [os.path.join(self.data_dir, p) for p in self.pieces] )

    self.piece_bar_pos = []

    for i, p in enumerate(self.pieces):
      bar_pos, p_evs = pickle_load(p)
      if not i % 200:
        print ('[preparing data] now at #{}'.format(i))
      if bar_pos[-1] == len(p_evs):
        print ('piece {}, got appended bar markers'.format(p))
        bar_pos = bar_pos[:-1]
      if len(p_evs) - bar_pos[-1] == 2:
        # got empty trailing bar
        bar_pos = bar_pos[:-1]

      bar_pos.append(len(p_evs))

      self.piece_bar_pos.append(bar_pos)

  def get_sample_from_file(self, piece_idx):
    piece_evs = pickle_load(self.pieces[piece_idx])[1]
    if len(self.piece_bar_pos[piece_idx]) > self.model_max_bars and self.appoint_st_bar is None:
      picked_st_bar = random.choice(
        range(len(self.piece_bar_pos[piece_idx]) - self.model_max_bars)
      )
    elif self.appoint_st_bar is not None and self.appoint_st_bar < len(self.piece_bar_pos[piece_idx]) - self.model_max_bars:
      picked_st_bar = self.appoint_st_bar
    else:
      picked_st_bar = 0

    piece_bar_pos = self.piece_bar_pos[piece_idx]

    if len(piece_bar_pos) > self.model_max_bars:
      piece_evs = piece_evs[ piece_bar_pos[picked_st_bar] : piece_bar_pos[picked_st_bar + self.model_max_bars] ]
      picked_bar_pos = np.array(piece_bar_pos[ picked_st_bar : picked_st_bar + self.model_max_bars ]) - piece_bar_pos[picked_st_bar]
      n_bars = self.model_max_bars
    else:
      picked_bar_pos = np.array(piece_bar_pos + [piece_bar_pos[-1]] * (self.model_max_bars - len(piece_bar_pos)))
      n_bars = len(piece_bar_pos)
      assert len(picked_bar_pos) == self.model_max_bars

    return piece_evs, picked_st_bar, picked_bar_pos, n_bars

  def pad_sequence(self, seq, maxlen, pad_value=None):
    if pad_value is None:
      pad_value = self.pad_token

    seq.extend( [pad_value for _ in range(maxlen- len(seq))] )

    return seq

  def pitch_augment(self, bar_events):
    bar_min_pitch, bar_max_pitch = check_extreme_pitch(bar_events)
    
    n_keys = random.choice(self.augment_range)
    while bar_min_pitch + n_keys < self.min_pitch or bar_max_pitch + n_keys > self.max_pitch:
      n_keys = random.choice(self.augment_range)

    augmented_bar_events = transpose_events(bar_events, n_keys)
    return augmented_bar_events

  def get_attr_classes(self, piece, st_bar):

    composer_cls_data = pickle_load(os.path.join(self.data_dir, 'attr_cls/composer.pkl'))
    composer_cls = [composer_cls_data[int(piece[0:-4])]] * self.model_max_bars

    assert len(composer_cls) == self.model_max_bars
    return composer_cls

  def get_encoder_input_data(self, bar_positions, bar_events):
    assert len(bar_positions) == self.model_max_bars + 1
    enc_padding_mask = np.ones((self.model_max_bars, self.model_enc_seqlen), dtype=bool)
    enc_padding_mask[:, :2] = False
    padded_enc_input = np.full((self.model_max_bars, self.model_enc_seqlen), dtype=int, fill_value=self.pad_token)
    enc_lens = np.zeros((self.model_max_bars,))

    for b, (st, ed) in enumerate(zip(bar_positions[:-1], bar_positions[1:])):
      enc_padding_mask[b, : (ed-st)] = False
      enc_lens[b] = ed - st
      within_bar_events = self.pad_sequence(bar_events[st : ed], self.model_enc_seqlen, self.pad_token)
      within_bar_events = np.array(within_bar_events)

      padded_enc_input[b, :] = within_bar_events[:self.model_enc_seqlen]

    return padded_enc_input, enc_padding_mask, enc_lens

  def __len__(self):
    return len(self.pieces)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    bar_events, st_bar, bar_pos, enc_n_bars = self.get_sample_from_file(idx)
    if self.do_augment:
      bar_events = self.pitch_augment(bar_events)

    if self.use_attr_cls:
      composer_cls = self.get_attr_classes(os.path.basename(self.pieces[idx]), st_bar)
      composer_cls_expand = np.zeros((self.model_dec_seqlen,), dtype=int)
      for i, (b_st, b_ed) in enumerate(zip(bar_pos[:-1], bar_pos[1:])):
        composer_cls_expand[b_st:b_ed] = composer_cls[i]
    else:
      composer_cls = [0]
      composer_cls_expand = [0]

    bar_tokens = convert_event(bar_events, self.event2idx, to_ndarr=False)
    bar_pos = bar_pos.tolist() + [len(bar_tokens)]

    enc_inp, enc_padding_mask, enc_lens = self.get_encoder_input_data(bar_pos, bar_tokens)

    length = len(bar_tokens)
    if self.pad_to_same:
      inp = self.pad_sequence(bar_tokens, self.model_dec_seqlen + 1) 
    else:
      inp = self.pad_sequence(bar_tokens, len(bar_tokens) + 1, pad_value=self.dec_end_pad_value)
    target = np.array(inp[1:], dtype=int)
    inp = np.array(inp[:-1], dtype=int)
    assert len(inp) == len(target)

    return {
      'id': idx,
      'piece_id': int(os.path.basename(self.pieces[idx]).replace('.pkl', '')),
      'st_bar_id': st_bar,
      'bar_pos': np.array(bar_pos, dtype=int),
      'enc_input': enc_inp,
      'dec_input': inp[:self.model_dec_seqlen],
      'dec_target': target[:self.model_dec_seqlen],
      'composer_cls': composer_cls_expand,
      'composer_cls_bar': np.array(composer_cls),
      'length': min(length, self.model_dec_seqlen),
      'enc_padding_mask': enc_padding_mask,
      'enc_length': enc_lens,
      'enc_n_bars': enc_n_bars
    }

if __name__ == "__main__":
  # codes below are for unit test
  dset = REMIFullSongTransformerDataset(
    './remi_dataset', './pickles/remi_vocab.pkl', do_augment=True, use_attr_cls=True,
    model_max_bars=16, model_dec_seqlen=1280, model_enc_seqlen=128, min_pitch=22, max_pitch=107
  )
  print (dset.bar_token, dset.pad_token, dset.vocab_size)
  print ('length:', len(dset))

  # for i in random.sample(range(len(dset)), 100):
  # for i in range(len(dset)):
  #   sample = dset[i]
    # print (i, len(sample['bar_pos']), sample['bar_pos'])
    # print (i)
    # print ('******* ----------- *******')
    # print ('piece: {}, st_bar: {}'.format(sample['piece_id'], sample['st_bar_id']))
    # print (sample['enc_input'][:8, :16])
    # print (sample['dec_input'][:16])
    # print (sample['dec_target'][:16])
    # print (sample['enc_padding_mask'][:32, :16])
    # print (sample['length'])

  dloader = DataLoader(dset, batch_size=4, shuffle=False, num_workers=24)
  for i, batch in enumerate(dloader):
    for k, v in batch.items():
      if torch.is_tensor(v):
        print (k, ':', v.dtype, v.size())
    print ('=====================================\n')
